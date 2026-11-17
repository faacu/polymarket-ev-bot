"""
Algorithmic trading bot for Polymarket prediction markets.
Strategy: Expected Value (EV) arbitrage using Bayesian probability estimation.
AI Brain: Google Gemini as a Bayesian Analyst for market mispricing detection.
Risk management: Fractional Kelly Criterion for position sizing.
"""

import os
import re
import html
import time
import logging
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field
from typing import Optional
import feedparser
try:
	from py_clob_client.client import ClobClient
	from py_clob_client.clob_types import ApiCreds
except ImportError:
	ClobClient = None  # type: ignore
	ApiCreds = None  # type: ignore

try:
	from google import genai
	from google.genai import types as ai_types
except ImportError:
	genai = None  # type: ignore
	ai_types = None  # type: ignore
from dotenv import load_dotenv
from colorama import Fore, Style, init as colorama_init

colorama_init(autoreset=True)
load_dotenv()

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
	level=logging.INFO,
	format='%(asctime)s [%(levelname)s] %(message)s',
	handlers=[
		logging.StreamHandler(),
		logging.FileHandler('bot.log', encoding='utf-8'),
	],
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
def _parse_watchlist(raw_env: Optional[str]) -> list[str]:
	"""Parses watchlist from JSON array or comma-separated string."""
	if not raw_env or not raw_env.strip():
		return []
	raw = raw_env.strip()
	try:
		parsed = json.loads(raw)
		if isinstance(parsed, list):
			return [str(item).strip() for item in parsed if item]
	except json.JSONDecodeError:
		pass
	return [item.strip() for item in raw.split(',') if item.strip()]


# List of Polymarket condition IDs (market identifiers) to monitor
WATCHLIST: list[str] = _parse_watchlist(os.getenv('POLYMARKET_WATCHLIST'))

KELLY_FRACTION = 0.25            # Fractional Kelly: use 25% of the suggested stake
MIN_EDGE_THRESHOLD = 0.05        # Minimum edge (5%) to place a bet
SCAN_INTERVAL_SECONDS = 300      # Slow loop cadence for Gemini AI scans (5 min)
GEMINI_MODEL = 'gemini-2.5-flash-preview-04-17'
POLYMARKET_HOST = os.getenv('POLYMARKET_HOST', 'https://clob.polymarket.com')
POLYMARKET_CHAIN_ID = int(os.getenv('POLYMARKET_CHAIN_ID', '137'))  # Polygon mainnet


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class MarketSnapshot:
	"""Holds the current state of a Polymarket binary market."""
	condition_id: str
	question: str
	yes_price: float             # Market-implied probability for YES (0.0 – 1.0)
	no_price: float              # Market-implied probability for NO (0.0 – 1.0)
	news_summary: str            # External context fed to Gemini


@dataclass
class BayesianSignal:
	"""Signal produced by the Gemini Bayesian analysis layer."""
	condition_id: str
	true_probability_yes: float  # Gemini's estimated true probability
	market_implied_probability: float
	expected_value_usd: float
	action: str                  # 'BUY_YES' | 'BUY_NO' | 'HOLD'
	reasoning: str


@dataclass
class ActiveBet:
	"""Represents an active bet placed by the bot on a Polymarket market."""
	condition_id: str
	side: str                    # 'YES' or 'NO'
	entry_price: float           # Price paid per share (0.0 – 1.0)
	shares: float                # Number of shares purchased
	stake_usd: float             # Total USD invested
	order_id: str
	placed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Polymarket CLOB connector
# ---------------------------------------------------------------------------
class PolymarketConnector:
	"""
	Wraps py_clob_client to interact with the Polymarket CLOB.
	Provides methods to fetch market data and place orders.
	"""

	def __init__(self, private_key: str, api_key: str, api_secret: str, api_passphrase: str):
		credentials = ApiCreds(
			api_key=api_key,
			api_secret=api_secret,
			api_passphrase=api_passphrase,
		)
		self._client = ClobClient(
			host=POLYMARKET_HOST,
			chain_id=POLYMARKET_CHAIN_ID,
			key=private_key,
			creds=credentials,
		)
		logger.info('Polymarket CLOB connector initialised — host: %s', POLYMARKET_HOST)

	def fetch_usdc_balance(self) -> float:
		"""Returns the available USDC balance in the connected wallet."""
		balance_data = self._client.get_balance()
		return float(balance_data.get('balance', 0.0))

	def fetch_market_prices(self, condition_id: str) -> dict:
		"""
		Fetches the best bid/ask midpoint for YES and NO tokens.
		Returns a dict with keys 'yes_price' and 'no_price'.
		"""
		order_book = self._client.get_order_book(condition_id)
		yes_price = self._extract_midpoint_price(order_book, token_side='YES')
		no_price = round(1.0 - yes_price, 4)
		return {'yes_price': yes_price, 'no_price': no_price}

	def _extract_midpoint_price(self, order_book: dict, token_side: str) -> float:
		"""Calculates the midpoint price from the best bid and ask."""
		bids = order_book.get('bids', [])
		asks = order_book.get('asks', [])
		best_bid = float(bids[0]['price']) if bids else 0.0
		best_ask = float(asks[0]['price']) if asks else 1.0
		return round((best_bid + best_ask) / 2, 4)

	def fetch_market_info(self, condition_id: str) -> dict:
		"""Fetches metadata for a market, including the question text."""
		return self._client.get_market(condition_id)

	def place_market_order(self, condition_id: str, side: str, amount_usd: float) -> dict:
		"""
		Places a market buy order for YES or NO shares.
		amount_usd is the total USD amount to spend.
		"""
		logger.info(
			'Placing order: BUY_%s on %s for $%.2f',
			side, condition_id, amount_usd,
		)
		order = self._client.create_market_order(
			condition_id=condition_id,
			outcome=side,
			amount=amount_usd,
		)
		return order


# ---------------------------------------------------------------------------
# AI Bayesian signal analyzer
# ---------------------------------------------------------------------------
class GeminiSignalAnalyzer:
	"""
	Uses Google Gemini as a Bayesian Analyst to detect EV opportunities
	in Polymarket binary prediction markets.
	"""

	def __init__(self, api_key: str):
		self._client = genai.Client(api_key=api_key)
		self._model_id = GEMINI_MODEL

	def _build_prompt(self, snapshot: MarketSnapshot) -> str:
		"""Builds the structured Bayesian analysis prompt for Gemini."""
		return f"""
You are a senior Bayesian analyst specializing in prediction market arbitrage.

Your task is to analyze the following binary prediction market and determine whether the current market price represents a mispricing opportunity.

MARKET QUESTION:
"{snapshot.question}"

CURRENT MARKET DATA:
- Market-Implied Probability (YES price): {snapshot.yes_price:.4f} ({snapshot.yes_price:.1%})
- Market-Implied Probability (NO price): {snapshot.no_price:.4f} ({snapshot.no_price:.1%})

RECENT NEWS AND CONTEXT:
{snapshot.news_summary}

YOUR ANALYSIS PROCESS:
1. Based on the news and context provided, estimate the TRUE probability that the event resolves as YES.
2. Calculate the edge: edge = true_probability_yes - market_implied_probability
3. Calculate the Expected Value of buying YES shares at the market price:
   EV_yes = (true_probability_yes * (1 - yes_price)) - ((1 - true_probability_yes) * yes_price)
4. Calculate the Expected Value of buying NO shares:
   EV_no  = ((1 - true_probability_yes) * (1 - no_price)) - (true_probability_yes * no_price)
5. Only recommend BUY_YES or BUY_NO if the absolute edge exceeds 5% (0.05).
   Otherwise, recommend HOLD.

EXECUTION RULES:
- BUY_YES: if true_probability_yes > market_implied_probability AND edge > 0.05
- BUY_NO:  if true_probability_yes < market_implied_probability AND abs(edge) > 0.05
- HOLD:    in all other cases (insufficient edge, high uncertainty, insufficient data)

Respond ONLY with a valid JSON object, no markdown, no commentary:
{{
  "true_probability_yes": <float between 0.0 and 1.0>,
  "market_implied_probability": {snapshot.yes_price},
  "expected_value_usd": <float, EV in USD per $1 staked, can be negative>,
  "action": "BUY_YES" | "BUY_NO" | "HOLD",
  "reasoning": "<one concise sentence explaining the key factor>"
}}
"""

	def analyse(self, snapshot: MarketSnapshot) -> BayesianSignal:
		"""Calls Gemini and parses the Bayesian EV signal."""
		prompt = self._build_prompt(snapshot)
		try:
			response = self._client.models.generate_content(
				model=self._model_id,
				contents=prompt,
				config=ai_types.GenerateContentConfig(
					response_mime_type='application/json'
				)
			)

			raw_text = response.text.strip()
			parsed = json.loads(raw_text)

			return BayesianSignal(
				condition_id=snapshot.condition_id,
				true_probability_yes=float(parsed.get('true_probability_yes', 0.5)),
				market_implied_probability=float(parsed.get('market_implied_probability', snapshot.yes_price)),
				expected_value_usd=float(parsed.get('expected_value_usd', 0.0)),
				action=parsed.get('action', 'HOLD').upper(),
				reasoning=parsed.get('reasoning', ''),
			)
		except Exception as error:
			logger.error('Gemini API error for %s: %s', snapshot.condition_id, error)
			return BayesianSignal(
				condition_id=snapshot.condition_id,
				true_probability_yes=0.5,
				market_implied_probability=snapshot.yes_price,
				expected_value_usd=0.0,
				action='HOLD',
				reasoning='API error — defaulting to HOLD',
			)


# ---------------------------------------------------------------------------
# Risk manager — Fractional Kelly Criterion
# ---------------------------------------------------------------------------
class RiskManager:
	"""
	Implements Fractional Kelly Criterion for position sizing on binary markets.
	Formula: f = (p - p_market) / (1 - p_market)
	Where:
	  f        = fraction of capital to stake
	  p        = true probability estimated by Gemini
	  p_market = current market price (implied probability)
	"""

	def __init__(self, kelly_fraction: float = KELLY_FRACTION):
		self._kelly_fraction = kelly_fraction

	def calculate_stake(self, balance: float, signal: BayesianSignal) -> float:
		"""
		Computes the USD amount to bet using Fractional Kelly.
		Returns 0.0 if the Kelly fraction is negative (no edge).
		"""
		p = signal.true_probability_yes
		p_market = signal.market_implied_probability

		if signal.action == 'BUY_NO':
			# Mirror the formula for the NO side
			p = 1.0 - signal.true_probability_yes
			p_market = 1.0 - signal.market_implied_probability

		denominator = 1.0 - p_market
		if denominator <= 0:
			return 0.0

		full_kelly = (p - p_market) / denominator

		if full_kelly <= 0:
			logger.debug('Kelly fraction is negative (%.4f) — no edge, skipping.', full_kelly)
			return 0.0

		fractional_kelly = full_kelly * self._kelly_fraction
		stake = balance * fractional_kelly

		logger.info(
			'Kelly sizing: full_kelly=%.4f | fractional(%.0f%%)=%.4f | stake=$%.2f',
			full_kelly, self._kelly_fraction * 100, fractional_kelly, stake,
		)
		return stake

	def validate_edge(self, signal: BayesianSignal) -> bool:
		"""Returns True if the edge is above the minimum threshold to trade."""
		edge = abs(signal.true_probability_yes - signal.market_implied_probability)
		if edge < MIN_EDGE_THRESHOLD:
			logger.debug(
				'Edge %.2f%% is below minimum threshold %.2f%% — HOLD',
				edge * 100, MIN_EDGE_THRESHOLD * 100,
			)
			return False
		return True


# ---------------------------------------------------------------------------
# Bet manager (replaces PositionManager)
# ---------------------------------------------------------------------------
class BetManager:
	"""Maintains the in-memory registry of active bets placed by the bot."""

	def __init__(self):
		self._bets: dict[str, ActiveBet] = {}

	def has_bet(self, condition_id: str) -> bool:
		return condition_id in self._bets

	def get_bet(self, condition_id: str) -> Optional[ActiveBet]:
		return self._bets.get(condition_id)

	def add_bet(self, bet: ActiveBet) -> None:
		self._bets[bet.condition_id] = bet
		logger.info(
			'Bet placed: BUY_%s on %s | shares=%.2f | stake=$%.2f @ price=%.4f',
			bet.side, bet.condition_id, bet.shares, bet.stake_usd, bet.entry_price,
		)

	def remove_bet(self, condition_id: str) -> None:
		self._bets.pop(condition_id, None)
		logger.info('Bet resolved/removed for market %s', condition_id)

	def all_condition_ids(self) -> list[str]:
		return list(self._bets.keys())


# ---------------------------------------------------------------------------
# News fetcher — RSS feeds
# ---------------------------------------------------------------------------
class NewsFetcher:
	"""
	Fetches and formats the latest macro/crypto news from RSS feeds.
	Designed to be resilient: a failing feed never crashes the bot.
	"""

	# Reliable public RSS feeds covering macro, crypto and prediction markets
	_FEEDS: list[dict] = [
		{
			'name': 'Yahoo Finance',
			'url': 'https://finance.yahoo.com/news/rssindex',
		},
		{
			'name': 'CoinDesk',
			'url': 'https://www.coindesk.com/arc/outboundfeeds/rss/',
		},
		{
			'name': 'Cointelegraph',
			'url': 'https://cointelegraph.com/rss',
		},
		{
			'name': 'Reuters Markets',
			'url': 'https://feeds.reuters.com/reuters/businessNews',
		},
	]

	# Max characters per article summary to avoid flooding the Gemini context window
	_MAX_SUMMARY_CHARS = 300

	@staticmethod
	def _clean_html(raw_text: str) -> str:
		"""Strips HTML tags and decodes HTML entities from a raw string."""
		without_tags = re.sub(r'<[^>]+>', '', raw_text)
		return html.unescape(without_tags).strip()

	@staticmethod
	def _parse_published_date(entry: dict) -> datetime:
		"""Extracts and normalises the publication date from a feed entry."""
		tuple_date = entry.get('published_parsed') or entry.get('updated_parsed')
		if tuple_date:
			return datetime(*tuple_date[:6], tzinfo=timezone.utc)
		return datetime.min.replace(tzinfo=timezone.utc)

	@classmethod
	def _fetch_single_feed(cls, feed_config: dict) -> list[dict]:
		"""
		Parses a single RSS feed and returns a list of article dicts.
		Returns an empty list on any network or parse failure.
		"""
		try:
			parsed = feedparser.parse(feed_config['url'])

			# feedparser does not raise exceptions — check the bozo flag for errors
			if parsed.bozo and not parsed.entries:
				raise ValueError(parsed.bozo_exception)

			articles = []
			for entry in parsed.entries:
				raw_summary = (
					entry.get('summary', '')
					or entry.get('description', '')
					or entry.get('title', '')
				)
				clean_summary = cls._clean_html(raw_summary)[:cls._MAX_SUMMARY_CHARS]
				articles.append({
					'date': cls._parse_published_date(entry),
					'title': cls._clean_html(entry.get('title', 'No title')),
					'summary': clean_summary,
					'source': feed_config['name'],
				})
			return articles

		except Exception as error:
			logger.warning(
				'NewsFetcher: could not load feed "%s" (%s) — %s',
				feed_config['name'], feed_config['url'], error,
			)
			return []

	@classmethod
	def get_latest_news(cls, limit: int = 10) -> str:
		"""
		Aggregates articles from all configured feeds, sorts them by date
		(newest first) and returns a clean, LLM-friendly formatted string.

		Returns a fallback message if all feeds fail, so Gemini can still
		decide to HOLD rather than crashing the bot.
		"""
		all_articles: list[dict] = []
		for feed_config in cls._FEEDS:
			all_articles.extend(cls._fetch_single_feed(feed_config))

		if not all_articles:
			logger.warning('NewsFetcher: all feeds failed — returning fallback message.')
			return 'News data currently unavailable.'

		sorted_articles = sorted(all_articles, key=lambda a: a['date'], reverse=True)
		top_articles = sorted_articles[:limit]

		formatted_lines = [
			f"[{article['date'].strftime('%Y-%m-%d %H:%M UTC')}] "
			f"({article['source']}) {article['title']}: {article['summary']}"
			for article in top_articles
		]
		return '\n'.join(formatted_lines)


# ---------------------------------------------------------------------------
# Console reporter
# ---------------------------------------------------------------------------
class ConsoleReporter:
	"""Prints colored, human-friendly status lines to the terminal."""

	def scan_start(self) -> None:
		print(f'\n{Fore.CYAN}[SCANNING MARKETS]{Style.RESET_ALL}', flush=True)

	def print_balance(self, balance: float) -> None:
		print(f'  {Fore.YELLOW}[BALANCE]{Style.RESET_ALL} ${balance:.2f} USDC')

	def print_signal(self, signal: BayesianSignal, stake: float) -> None:
		color = Fore.GREEN if signal.action == 'BUY_YES' else Fore.RED
		edge = abs(signal.true_probability_yes - signal.market_implied_probability)
		print(
			f'  {color}[{signal.action}]{Style.RESET_ALL} '
			f'{signal.condition_id[:12]}… | '
			f'true_p={signal.true_probability_yes:.2%} | '
			f'mkt_p={signal.market_implied_probability:.2%} | '
			f'edge={edge:.2%} | EV=${signal.expected_value_usd:.4f} | '
			f'stake=${stake:.2f}'
			f'\n    {Fore.WHITE}reason:{Style.RESET_ALL} {signal.reasoning}'
		)

	def print_hold(self, signal: BayesianSignal) -> None:
		edge = signal.true_probability_yes - signal.market_implied_probability
		print(
			f'  {Fore.WHITE}[HOLD]{Style.RESET_ALL} '
			f'{signal.condition_id[:12]}… | '
			f'edge={edge:+.2%} | {signal.reasoning}'
		)

	def print_active_bet(self, bet: ActiveBet) -> None:
		print(
			f'  {Fore.MAGENTA}[ACTIVE BET]{Style.RESET_ALL} '
			f'BUY_{bet.side} on {bet.condition_id[:12]}… | '
			f'stake=${bet.stake_usd:.2f} | shares={bet.shares:.2f} @ {bet.entry_price:.4f}'
		)

	def scan_done(self) -> None:
		print()


# ---------------------------------------------------------------------------
# Core bot
# ---------------------------------------------------------------------------
class TradingBot:
	"""
	Orchestrates the full EV arbitrage loop:
	  1. Fetch USDC balance
	  2. Report active bets
	  3. For each market in the watchlist:
	     a. Fetch current CLOB prices
	     b. Run Gemini Bayesian analysis with news context
	     c. If EV is positive and edge > 5%: size via Kelly and place order
	"""

	def __init__(self):
		self._clob = PolymarketConnector(
			private_key=os.getenv('POLYMARKET_PRIVATE_KEY', ''),
			api_key=os.getenv('POLYMARKET_API_KEY', ''),
			api_secret=os.getenv('POLYMARKET_API_SECRET', ''),
			api_passphrase=os.getenv('POLYMARKET_API_PASSPHRASE', ''),
		)
		self._ai = GeminiSignalAnalyzer(api_key=os.getenv('GEMINI_API_KEY', ''))
		self._risk = RiskManager(kelly_fraction=KELLY_FRACTION)
		self._bets = BetManager()
		self._reporter = ConsoleReporter()
		self._last_scan_time = 0.0  # Force scan on first tick

		initial_balance = self._clob.fetch_usdc_balance()
		logger.info('Bot started — initial USDC balance: $%.2f', initial_balance)

	# ------------------------------------------------------------------
	# Trade lifecycle helpers
	# ------------------------------------------------------------------

	def _build_market_snapshot(self, condition_id: str) -> MarketSnapshot:
		"""Fetches live market data and assembles a MarketSnapshot."""
		market_info = self._clob.fetch_market_info(condition_id)
		prices = self._clob.fetch_market_prices(condition_id)

		question = market_info.get('question', condition_id)

		# Aggregate global macro/crypto news from RSS feeds.
		# NewsFetcher is resilient: returns a fallback string if all feeds fail,
		# so Gemini can still decide to HOLD rather than crashing the bot.
		market_description = market_info.get('description', '')
		latest_news = NewsFetcher.get_latest_news(limit=10)
		news_summary = (
			f'MARKET DESCRIPTION:\n{market_description}\n\n'
			f'LATEST MACRO/CRYPTO NEWS:\n{latest_news}'
		)

		return MarketSnapshot(
			condition_id=condition_id,
			question=question,
			yes_price=prices['yes_price'],
			no_price=prices['no_price'],
			news_summary=news_summary,
		)

	def _execute_bet(self, signal: BayesianSignal, balance: float) -> None:
		"""Validates the signal, sizes via Kelly and places the order."""
		if not self._risk.validate_edge(signal):
			self._reporter.print_hold(signal)
			return

		if signal.expected_value_usd <= 0:
			logger.info('EV is negative (%.4f) for %s — skipping.', signal.expected_value_usd, signal.condition_id)
			self._reporter.print_hold(signal)
			return

		stake = self._risk.calculate_stake(balance, signal)
		if stake <= 0:
			logger.warning('Calculated stake is $0 for %s — skipping.', signal.condition_id)
			return

		# DO NOT bet more than the available balance
		stake = min(stake, balance)

		side = 'YES' if signal.action == 'BUY_YES' else 'NO'
		price = signal.market_implied_probability if side == 'YES' else (1.0 - signal.market_implied_probability)

		order = self._clob.place_market_order(signal.condition_id, side, stake)

		shares = stake / price if price > 0 else 0.0
		order_id = str(order.get('orderID') or order.get('id') or order.get('order_id') or 'unknown') if isinstance(order, dict) else str(order)
		bet = ActiveBet(
			condition_id=signal.condition_id,
			side=side,
			entry_price=price,
			shares=shares,
			stake_usd=stake,
			order_id=order_id,
		)
		self._bets.add_bet(bet)
		self._reporter.print_signal(signal, stake)

	def _scan_market(self, condition_id: str, balance: float) -> None:
		"""Runs the full analysis pipeline for a single market."""
		if self._bets.has_bet(condition_id):
			bet = self._bets.get_bet(condition_id)
			if bet:
				self._reporter.print_active_bet(bet)
			return

		snapshot = self._build_market_snapshot(condition_id)
		signal = self._ai.analyse(snapshot)

		logger.info(
			'AI signal for %s → %s | true_p=%.2f%% | mkt_p=%.2f%% | EV=%.4f',
			condition_id[:16], signal.action,
			signal.true_probability_yes * 100,
			signal.market_implied_probability * 100,
			signal.expected_value_usd,
		)

		if signal.action in ('BUY_YES', 'BUY_NO'):
			self._execute_bet(signal, balance)
		else:
			self._reporter.print_hold(signal)

	# ------------------------------------------------------------------
	# Main loop
	# ------------------------------------------------------------------

	def run(self) -> None:
		"""Starts the EV arbitrage scanning loop."""
		logger.info('=== Polymarket EV Bot started ===')

		if not WATCHLIST:
			logger.error(
				'WATCHLIST is empty. Add Polymarket condition IDs to POLYMARKET_WATCHLIST in .env'
			)
			return

		while True:
			try:
				current_time = time.time()
				is_scan_time = (current_time - self._last_scan_time) >= SCAN_INTERVAL_SECONDS

				if is_scan_time:
					balance = self._clob.fetch_usdc_balance()
					self._reporter.print_balance(balance)
					self._reporter.scan_start()

					for condition_id in WATCHLIST:
						self._scan_market(condition_id, balance)

					self._reporter.scan_done()
					self._last_scan_time = time.time()

			except ConnectionError as error:
				logger.error('Network error: %s — retrying in %ds', error, SCAN_INTERVAL_SECONDS)
			except Exception as error:
				logger.exception('Unexpected error: %s', error)

			time.sleep(SCAN_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
	bot = TradingBot()
	bot.run()
