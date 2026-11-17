"""
Spot Paper Trading Bot — Hybrid Technical + Fundamental Strategy.
Exchange: Binance Spot (public, read-only via CCXT).
AI Brain:  Google Gemini as a Hybrid Trader (TA + news confluence).
Wallet:    PaperWallet — fully simulated, no real orders placed.
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
import numpy as np
import feedparser
import ccxt
from google import genai
from google.genai import types as ai_types
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
WATCHLIST: list[str] = ['BTC/USDT', 'SOL/USDT', 'ETH/USDT']

GEMINI_MODEL         = 'gemini-2.5-flash-preview-04-17'
CANDLE_TIMEFRAME     = '15m'       # OHLCV timeframe fed to indicators
CANDLE_LIMIT         = 50          # Number of candles to fetch
SMA_SHORT            = 10          # Short SMA period
SMA_LONG             = 20          # Long SMA period
RSI_PERIOD           = 14          # RSI lookback period
MIN_CONFIDENCE       = 0.60        # Minimum Gemini confidence to execute a trade
LOOP_INTERVAL_SECONDS: int = int(os.getenv('LOOP_INTERVAL_SECONDS', '300'))

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class Candle:
	"""Represents a single OHLCV candle."""
	timestamp: datetime
	open: float
	high: float
	low: float
	close: float
	volume: float


@dataclass
class TechnicalSnapshot:
	"""Aggregated TA context built from raw OHLCV data."""
	symbol: str
	last_price: float
	candles: list[Candle]
	sma_short: float
	sma_long: float
	rsi: float
	news_summary: str


@dataclass
class TradeSignal:
	"""Signal produced by the Gemini Hybrid Trader."""
	symbol: str
	action: str                   # 'BUY' | 'SELL' | 'HOLD'
	confidence: float             # 0.0 – 1.0
	allocation_pct: float         # Fraction of available balance to use
	reasoning: str


@dataclass
class Transaction:
	"""Records a single simulated trade execution."""
	timestamp: datetime
	symbol: str
	side: str                     # 'BUY' | 'SELL'
	price: float
	amount_usdt: float
	amount_crypto: float
	note: str = ''


# ---------------------------------------------------------------------------
# PaperWallet — simulated balance & order execution
# ---------------------------------------------------------------------------
class PaperWallet:
	"""
	In-memory paper trading wallet.
	Simulates BUY/SELL execution against live market prices
	without placing any real orders.
	"""

	def __init__(self, initial_usdt: float):
		self._usdt: float = initial_usdt
		self._holdings: dict[str, float] = {}   # symbol -> crypto amount
		self._history: list[Transaction] = []
		logger.info('PaperWallet initialised — balance: $%.2f USDT', initial_usdt)

	# ------------------------------------------------------------------
	# Public accessors
	# ------------------------------------------------------------------

	@property
	def usdt_balance(self) -> float:
		return self._usdt

	def crypto_balance(self, symbol: str) -> float:
		return self._holdings.get(symbol, 0.0)

	@property
	def history(self) -> list[Transaction]:
		return list(self._history)

	# ------------------------------------------------------------------
	# Trade execution
	# ------------------------------------------------------------------

	def buy(self, symbol: str, price: float, amount_usdt: float) -> Optional[Transaction]:
		"""
		Buys crypto using USDT at the given price.
		Returns the Transaction or None if insufficient funds.
		"""
		if amount_usdt <= 0:
			logger.warning('PaperWallet.buy: amount_usdt must be positive.')
			return None

		if amount_usdt > self._usdt:
			logger.warning(
				'PaperWallet.buy: insufficient USDT (have $%.2f, need $%.2f). Capping.',
				self._usdt, amount_usdt,
			)
			amount_usdt = self._usdt

		if amount_usdt <= 0:
			return None

		amount_crypto = amount_usdt / price
		self._usdt -= amount_usdt
		self._holdings[symbol] = self._holdings.get(symbol, 0.0) + amount_crypto

		tx = Transaction(
			timestamp=datetime.now(timezone.utc),
			symbol=symbol,
			side='BUY',
			price=price,
			amount_usdt=amount_usdt,
			amount_crypto=amount_crypto,
		)
		self._history.append(tx)
		logger.info(
			'[PAPER BUY]  %s | %.6f crypto @ $%.4f | spent $%.2f | USDT left: $%.2f',
			symbol, amount_crypto, price, amount_usdt, self._usdt,
		)
		return tx

	def sell(self, symbol: str, price: float, amount_crypto: float) -> Optional[Transaction]:
		"""
		Sells crypto for USDT at the given price.
		Returns the Transaction or None if insufficient holdings.
		"""
		if amount_crypto <= 0:
			logger.warning('PaperWallet.sell: amount_crypto must be positive.')
			return None

		held = self._holdings.get(symbol, 0.0)
		if amount_crypto > held:
			logger.warning(
				'PaperWallet.sell: not enough %s (have %.6f, need %.6f). Capping.',
				symbol, held, amount_crypto,
			)
			amount_crypto = held

		if amount_crypto <= 0:
			return None

		amount_usdt = amount_crypto * price
		self._holdings[symbol] = held - amount_crypto
		self._usdt += amount_usdt

		tx = Transaction(
			timestamp=datetime.now(timezone.utc),
			symbol=symbol,
			side='SELL',
			price=price,
			amount_usdt=amount_usdt,
			amount_crypto=amount_crypto,
		)
		self._history.append(tx)
		logger.info(
			'[PAPER SELL] %s | %.6f crypto @ $%.4f | received $%.2f | USDT balance: $%.2f',
			symbol, amount_crypto, price, amount_usdt, self._usdt,
		)
		return tx


# ---------------------------------------------------------------------------
# Technical Indicator Engine (pure numpy — no TA-Lib dependency)
# ---------------------------------------------------------------------------
class IndicatorEngine:
	"""Computes SMA and RSI from a numpy close-price array."""

	@staticmethod
	def sma(closes: np.ndarray, period: int) -> float:
		"""Returns the most recent Simple Moving Average value."""
		if len(closes) < period:
			return float('nan')
		return float(np.mean(closes[-period:]))

	@staticmethod
	def rsi(closes: np.ndarray, period: int = RSI_PERIOD) -> float:
		"""
		Computes RSI using Wilder's Smoothed Moving Average method.
		Returns NaN if there are insufficient data points.
		"""
		if len(closes) < period + 1:
			return float('nan')

		deltas = np.diff(closes)
		gains  = np.where(deltas > 0, deltas, 0.0)
		losses = np.where(deltas < 0, -deltas, 0.0)

		# Initial averages (simple mean for the seed window)
		avg_gain = np.mean(gains[:period])
		avg_loss = np.mean(losses[:period])

		# Wilder's smoothing for remaining periods
		for gain, loss in zip(gains[period:], losses[period:]):
			avg_gain = (avg_gain * (period - 1) + gain) / period
			avg_loss = (avg_loss * (period - 1) + loss) / period

		if avg_loss == 0:
			return 100.0

		rs = avg_gain / avg_loss
		return float(100.0 - (100.0 / (1.0 + rs)))


# ---------------------------------------------------------------------------
# Market Data Provider — CCXT Binance Spot (public, read-only)
# ---------------------------------------------------------------------------
class MarketDataProvider:
	"""
	Fetches public OHLCV and price data from Binance Spot
	using CCXT in anonymous (no API key) mode.
	"""

	def __init__(self):
		self._exchange = ccxt.binance({
			'options': {'defaultType': 'spot'},
		})
		logger.info('MarketDataProvider: Binance Spot (public) initialised.')

	def fetch_candles(self, symbol: str, timeframe: str = CANDLE_TIMEFRAME, limit: int = CANDLE_LIMIT) -> list[Candle]:
		"""Fetches OHLCV candles and returns them as Candle dataclass instances."""
		try:
			raw = self._exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
			candles = []
			for row in raw:
				ts, o, h, l, c, v = row
				candles.append(Candle(
					timestamp=datetime.fromtimestamp(ts / 1000, tz=timezone.utc),
					open=float(o),
					high=float(h),
					low=float(l),
					close=float(c),
					volume=float(v),
				))
			return candles
		except Exception as error:
			logger.error('MarketDataProvider.fetch_candles(%s): %s', symbol, error)
			return []

	def fetch_last_price(self, symbol: str) -> float:
		"""Returns the latest ticker price for a symbol."""
		try:
			ticker = self._exchange.fetch_ticker(symbol)
			return float(ticker.get('last', 0.0))
		except Exception as error:
			logger.error('MarketDataProvider.fetch_last_price(%s): %s', symbol, error)
			return 0.0


# ---------------------------------------------------------------------------
# News fetcher — RSS feeds
# ---------------------------------------------------------------------------
class NewsFetcher:
	"""
	Fetches and formats the latest macro/crypto news from RSS feeds.
	Designed to be resilient: a failing feed never crashes the bot.
	"""

	_FEEDS: list[dict] = [
		{'name': 'Yahoo Finance',    'url': 'https://finance.yahoo.com/news/rssindex'},
		{'name': 'CoinDesk',         'url': 'https://www.coindesk.com/arc/outboundfeeds/rss/'},
		{'name': 'Cointelegraph',    'url': 'https://cointelegraph.com/rss'},
		{'name': 'Reuters Markets',  'url': 'https://feeds.reuters.com/reuters/businessNews'},
	]

	_MAX_SUMMARY_CHARS = 300

	@staticmethod
	def _clean_html(raw_text: str) -> str:
		without_tags = re.sub(r'<[^>]+>', '', raw_text)
		return html.unescape(without_tags).strip()

	@staticmethod
	def _parse_published_date(entry: dict) -> datetime:
		tuple_date = entry.get('published_parsed') or entry.get('updated_parsed')
		if tuple_date:
			return datetime(*tuple_date[:6], tzinfo=timezone.utc)
		return datetime.min.replace(tzinfo=timezone.utc)

	@classmethod
	def _fetch_single_feed(cls, feed_config: dict) -> list[dict]:
		try:
			parsed = feedparser.parse(feed_config['url'])
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
					'date':    cls._parse_published_date(entry),
					'title':   cls._clean_html(entry.get('title', 'No title')),
					'summary': clean_summary,
					'source':  feed_config['name'],
				})
			return articles

		except Exception as error:
			logger.warning(
				'NewsFetcher: feed "%s" failed — %s',
				feed_config['name'], error,
			)
			return []

	@classmethod
	def get_latest_news(cls, limit: int = 10) -> str:
		"""
		Aggregates articles from all configured feeds, sorts them newest-first,
		and returns a clean LLM-friendly string.
		"""
		all_articles: list[dict] = []
		for feed_config in cls._FEEDS:
			all_articles.extend(cls._fetch_single_feed(feed_config))

		if not all_articles:
			logger.warning('NewsFetcher: all feeds failed — returning fallback.')
			return 'News data currently unavailable.'

		sorted_articles = sorted(all_articles, key=lambda a: a['date'], reverse=True)
		top_articles    = sorted_articles[:limit]

		lines = [
			f"[{a['date'].strftime('%Y-%m-%d %H:%M UTC')}] ({a['source']}) {a['title']}: {a['summary']}"
			for a in top_articles
		]
		return '\n'.join(lines)


# ---------------------------------------------------------------------------
# Gemini Hybrid Signal Analyzer
# ---------------------------------------------------------------------------
class GeminiSignalAnalyzer:
	"""
	Uses Google Gemini as a Hybrid Spot Trader.
	Combines technical indicators (SMA, RSI) with live news to
	produce a structured BUY / SELL / HOLD decision.
	"""

	def __init__(self, api_key: str):
		self._client   = genai.Client(api_key=api_key)
		self._model_id = GEMINI_MODEL

	def _format_candles(self, candles: list[Candle], last_n: int = 10) -> str:
		"""Returns the last N candles as a readable table for the prompt."""
		rows = []
		for c in candles[-last_n:]:
			rows.append(
				f"  {c.timestamp.strftime('%m-%d %H:%M')} | "
				f"O:{c.open:.4f}  H:{c.high:.4f}  L:{c.low:.4f}  "
				f"C:{c.close:.4f}  V:{c.volume:.2f}"
			)
		return '\n'.join(rows)

	def _build_prompt(self, snapshot: TechnicalSnapshot) -> str:
		candle_table = self._format_candles(snapshot.candles)

		# Trend bias derived from SMA crossover
		if not (np.isnan(snapshot.sma_short) or np.isnan(snapshot.sma_long)):
			trend_bias = 'BULLISH (SMA10 > SMA20)' if snapshot.sma_short > snapshot.sma_long else 'BEARISH (SMA10 < SMA20)'
		else:
			trend_bias = 'INSUFFICIENT DATA'

		# RSI interpretation
		if np.isnan(snapshot.rsi):
			rsi_label = 'N/A'
		elif snapshot.rsi >= 70:
			rsi_label = f'{snapshot.rsi:.1f} (OVERBOUGHT)'
		elif snapshot.rsi <= 30:
			rsi_label = f'{snapshot.rsi:.1f} (OVERSOLD)'
		else:
			rsi_label = f'{snapshot.rsi:.1f} (NEUTRAL)'

		return f"""You are a Senior Spot Trader executing a hybrid Technical Analysis + Fundamental Analysis strategy.
You must look for CONFLUENCE between both signals before acting. If signals conflict, choose HOLD.

=== SYMBOL: {snapshot.symbol} ===

--- TECHNICAL DATA ({CANDLE_TIMEFRAME} candles) ---
Last 10 candles (Open | High | Low | Close | Volume):
{candle_table}

Current Price : ${snapshot.last_price:.4f}
SMA-{SMA_SHORT:02d}        : {snapshot.sma_short:.4f}
SMA-{SMA_LONG:02d}        : {snapshot.sma_long:.4f}
Trend Bias    : {trend_bias}
RSI-{RSI_PERIOD:02d}       : {rsi_label}

--- FUNDAMENTAL DATA (latest news) ---
{snapshot.news_summary}

=== CONFLUENCE RULES ===
BUY  → RSI oversold OR SMA bullish crossover   AND news are neutral-to-bullish for this asset.
SELL → RSI overbought OR SMA bearish crossover  AND news are neutral-to-bearish for this asset.
HOLD → Any conflicting signals, extreme uncertainty, or insufficient data.

=== OUTPUT FORMAT ===
Respond ONLY with a valid JSON object — no markdown, no commentary:
{{
  "action": "BUY" | "SELL" | "HOLD",
  "confidence": <float 0.0-1.0>,
  "allocation_pct": <float 0.0-1.0, fraction of available balance to use; 0.0 if HOLD>,
  "reasoning": "<one concise sentence combining TA + news rationale>"
}}
"""

	def analyse(self, snapshot: TechnicalSnapshot) -> TradeSignal:
		"""Calls Gemini and parses the hybrid trade signal."""
		prompt = self._build_prompt(snapshot)
		try:
			response = self._client.models.generate_content(
				model=self._model_id,
				contents=prompt,
				config=ai_types.GenerateContentConfig(
					response_mime_type='application/json'
				),
			)

			parsed = json.loads(response.text.strip())

			return TradeSignal(
				symbol=snapshot.symbol,
				action=parsed.get('action', 'HOLD').upper(),
				confidence=float(parsed.get('confidence', 0.0)),
				allocation_pct=float(parsed.get('allocation_pct', 0.0)),
				reasoning=parsed.get('reasoning', ''),
			)

		except Exception as error:
			logger.error('Gemini API error for %s: %s', snapshot.symbol, error)
			return TradeSignal(
				symbol=snapshot.symbol,
				action='HOLD',
				confidence=0.0,
				allocation_pct=0.0,
				reasoning=f'API error — defaulting to HOLD ({error})',
			)


# ---------------------------------------------------------------------------
# Console reporter
# ---------------------------------------------------------------------------
class ConsoleReporter:
	"""Renders colored, human-friendly cycle reports to the terminal."""

	def print_header(self, cycle: int) -> None:
		ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
		print(
			f'\n{Fore.CYAN}{"="*65}{Style.RESET_ALL}\n'
			f'{Fore.CYAN}  📊  SPOT PAPER TRADING BOT  |  Cycle #{cycle}  |  {ts}{Style.RESET_ALL}\n'
			f'{Fore.CYAN}{"="*65}{Style.RESET_ALL}',
			flush=True,
		)

	def print_wallet(self, wallet: PaperWallet, watchlist: list[str]) -> None:
		print(f'\n{Fore.YELLOW}  💰  PAPER WALLET{Style.RESET_ALL}')
		print(f'     USDT Balance : ${wallet.usdt_balance:>10.2f}')
		for symbol in watchlist:
			crypto_bal = wallet.crypto_balance(symbol)
			if crypto_bal > 0:
				print(f'     {symbol:<12}: {crypto_bal:.6f}')

	def print_signal(self, signal: TradeSignal, last_price: float) -> None:
		if signal.action == 'BUY':
			color = Fore.GREEN
			icon  = '🟢'
		elif signal.action == 'SELL':
			color = Fore.RED
			icon  = '🔴'
		else:
			color = Fore.WHITE
			icon  = '⚪'

		print(
			f'\n{color}  {icon}  [{signal.action}] {signal.symbol}{Style.RESET_ALL}'
			f' | price=${last_price:.4f}'
			f' | conf={signal.confidence:.0%}'
			f' | alloc={signal.allocation_pct:.0%}'
			f'\n     {Fore.WHITE}→{Style.RESET_ALL} {signal.reasoning}'
		)

	def print_trade_executed(self, tx: Transaction) -> None:
		color = Fore.GREEN if tx.side == 'BUY' else Fore.RED
		print(
			f'     {color}✔ EXECUTED:{Style.RESET_ALL} {tx.side} {tx.amount_crypto:.6f} {tx.symbol}'
			f' @ ${tx.price:.4f}  (${tx.amount_usdt:.2f} USDT)'
		)

	def print_skipped(self, symbol: str, reason: str) -> None:
		print(f'     {Fore.YELLOW}⚠  SKIPPED:{Style.RESET_ALL} {symbol} — {reason}')

	def print_footer(self) -> None:
		print(f'\n{Fore.CYAN}{"="*65}{Style.RESET_ALL}\n')


# ---------------------------------------------------------------------------
# Core bot orchestrator
# ---------------------------------------------------------------------------
class SpotPaperTradingBot:
	"""
	Orchestrates the full paper trading loop:
	  1. Fetch OHLCV candles from Binance Spot (public).
	  2. Compute technical indicators (SMA, RSI).
	  3. Fetch latest news from RSS feeds.
	  4. Send the hybrid snapshot to Gemini for a BUY/SELL/HOLD signal.
	  5. Execute simulated trades on the PaperWallet.
	  6. Print a detailed cycle report.
	"""

	def __init__(self):
		initial_balance = float(os.getenv('INITIAL_PAPER_BALANCE', '1000.0'))
		gemini_api_key  = os.getenv('GEMINI_API_KEY', '')

		self._wallet    = PaperWallet(initial_balance)
		self._market    = MarketDataProvider()
		self._indicators = IndicatorEngine()
		self._ai        = GeminiSignalAnalyzer(api_key=gemini_api_key)
		self._reporter  = ConsoleReporter()
		self._cycle     = 0

	# ------------------------------------------------------------------
	# Snapshot builder
	# ------------------------------------------------------------------

	def _build_snapshot(self, symbol: str, news: str) -> Optional[TechnicalSnapshot]:
		"""Fetches candles and computes indicators, returns None on failure."""
		candles = self._market.fetch_candles(symbol)
		if len(candles) < SMA_LONG + 1:
			logger.warning('Not enough candles for %s (%d returned).', symbol, len(candles))
			return None

		closes = np.array([c.close for c in candles])
		sma_short  = self._indicators.sma(closes, SMA_SHORT)
		sma_long   = self._indicators.sma(closes, SMA_LONG)
		rsi_value  = self._indicators.rsi(closes, RSI_PERIOD)
		last_price = candles[-1].close

		return TechnicalSnapshot(
			symbol=symbol,
			last_price=last_price,
			candles=candles,
			sma_short=sma_short,
			sma_long=sma_long,
			rsi=rsi_value,
			news_summary=news,
		)

	# ------------------------------------------------------------------
	# Trade execution
	# ------------------------------------------------------------------

	def _execute_signal(self, signal: TradeSignal, snapshot: TechnicalSnapshot) -> None:
		"""Routes a BUY/SELL signal to the PaperWallet and prints the result."""
		price = snapshot.last_price

		if signal.action == 'BUY':
			amount_usdt = self._wallet.usdt_balance * signal.allocation_pct
			if amount_usdt < 1.0:
				self._reporter.print_skipped(signal.symbol, f'insufficient USDT (${self._wallet.usdt_balance:.2f})')
				return
			tx = self._wallet.buy(signal.symbol, price, amount_usdt)

		elif signal.action == 'SELL':
			held_crypto = self._wallet.crypto_balance(signal.symbol)
			if held_crypto <= 0:
				self._reporter.print_skipped(signal.symbol, f'no {signal.symbol} holdings to sell')
				return
			amount_crypto = held_crypto * signal.allocation_pct
			tx = self._wallet.sell(signal.symbol, price, amount_crypto)

		else:
			return

		if tx:
			self._reporter.print_trade_executed(tx)

	# ------------------------------------------------------------------
	# Per-symbol pipeline
	# ------------------------------------------------------------------

	def _process_symbol(self, symbol: str, news: str) -> None:
		"""Runs the full analysis + execution pipeline for a single symbol."""
		snapshot = self._build_snapshot(symbol, news)
		if snapshot is None:
			self._reporter.print_skipped(symbol, 'insufficient market data')
			return

		signal = self._ai.analyse(snapshot)

		logger.info(
			'[%s] action=%s conf=%.2f alloc=%.2f | %s',
			symbol, signal.action, signal.confidence, signal.allocation_pct, signal.reasoning,
		)

		self._reporter.print_signal(signal, snapshot.last_price)

		if signal.action in ('BUY', 'SELL') and signal.confidence >= MIN_CONFIDENCE:
			self._execute_signal(signal, snapshot)
		elif signal.action in ('BUY', 'SELL'):
			self._reporter.print_skipped(
				symbol,
				f'confidence {signal.confidence:.0%} below threshold {MIN_CONFIDENCE:.0%}',
			)

	# ------------------------------------------------------------------
	# Main loop
	# ------------------------------------------------------------------

	def run(self) -> None:
		"""Starts the continuous paper trading loop."""
		logger.info('=== Spot Paper Trading Bot started ===')
		logger.info('Watchlist: %s | Interval: %ds', WATCHLIST, LOOP_INTERVAL_SECONDS)

		while True:
			try:
				self._cycle += 1
				self._reporter.print_header(self._cycle)
				self._reporter.print_wallet(self._wallet, WATCHLIST)

				# Fetch news once per cycle — shared across all symbols
				news = NewsFetcher.get_latest_news(limit=10)

				print(f'\n{Fore.CYAN}  📡  ANALYZING MARKETS ...{Style.RESET_ALL}')
				for symbol in WATCHLIST:
					self._process_symbol(symbol, news)

				self._reporter.print_footer()

			except KeyboardInterrupt:
				logger.info('Bot stopped by user.')
				break
			except Exception as error:
				logger.exception('Unexpected error in main loop: %s', error)

			logger.info('Sleeping %ds until next cycle...', LOOP_INTERVAL_SECONDS)
			time.sleep(LOOP_INTERVAL_SECONDS)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
	bot = SpotPaperTradingBot()
	bot.run()
