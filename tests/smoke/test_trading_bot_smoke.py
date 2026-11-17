import unittest
from unittest.mock import MagicMock, patch
from bot import TradingBot, BayesianSignal


class TestTradingBotSmoke(unittest.TestCase):
	"""Smoke test verifying critical end-to-end bot pipeline without live network calls."""

	@patch("bot.PolymarketConnector")
	@patch("bot.GeminiSignalAnalyzer")
	def test_full_pipeline_scan_and_execution_smoke(self, mock_analyzer_cls, mock_clob_cls):
		# Setup CLOB mock
		mock_clob = MagicMock()
		mock_clob.fetch_usdc_balance.return_value = 500.0
		mock_clob.fetch_market_info.return_value = {
			"question": "Will BTC break $120k in Q4?",
			"description": "Market resolving based on index price."
		}
		mock_clob.fetch_market_prices.return_value = {
			"yes_price": 0.40,
			"no_price": 0.60
		}
		mock_clob.place_market_order.return_value = {"orderID": "test_order_999"}
		mock_clob_cls.return_value = mock_clob

		# Setup Gemini Signal mock
		mock_analyzer = MagicMock()
		mock_analyzer.analyse.return_value = BayesianSignal(
			condition_id="0xmarket_test",
			true_probability_yes=0.65,
			market_implied_probability=0.40,
			expected_value_usd=0.25,
			action="BUY_YES",
			reasoning="Catalyst confirms upside skew",
		)
		mock_analyzer_cls.return_value = mock_analyzer

		# Instantiate Bot
		bot = TradingBot()

		# Run market scan on condition
		bot._scan_market("0xmarket_test", balance=500.0)

		# Verify order was placed
		mock_clob.place_market_order.assert_called_once()
		args, _ = mock_clob.place_market_order.call_args
		self.assertEqual(args[0], "0xmarket_test")
		self.assertEqual(args[1], "YES")
		self.assertGreater(args[2], 0.0)

		# Verify bet registered in BetManager
		self.assertTrue(bot._bets.has_bet("0xmarket_test"))

		# Second scan should recognize active bet and avoid duplicate orders
		mock_clob.place_market_order.reset_mock()
		bot._scan_market("0xmarket_test", balance=500.0)
		mock_clob.place_market_order.assert_not_called()


if __name__ == "__main__":
	unittest.main()
