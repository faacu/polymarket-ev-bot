import unittest
from unittest.mock import MagicMock, patch
from bot import GeminiSignalAnalyzer, MarketSnapshot


class TestGeminiSignalAnalyzer(unittest.TestCase):
	"""Validates Gemini Bayesian output parsing and fail-safe fallbacks."""

	@patch("google.genai.Client")
	def test_analyse_successful_json_response(self, mock_client_cls):
		mock_client = MagicMock()
		mock_client_cls.return_value = mock_client

		mock_response = MagicMock()
		mock_response.text = (
			'{"true_probability_yes": 0.65, "market_implied_probability": 0.45, '
			'"expected_value_usd": 0.15, "action": "BUY_YES", '
			'"reasoning": "Strong geopolitical tailwinds"}'
		)
		mock_client.models.generate_content.return_value = mock_response

		analyzer = GeminiSignalAnalyzer(api_key="test_api_key")
		snapshot = MarketSnapshot(
			condition_id="0x111",
			question="Will event X happen?",
			yes_price=0.45,
			no_price=0.55,
			news_summary="Sample news context",
		)

		signal = analyzer.analyse(snapshot)

		self.assertEqual(signal.condition_id, "0x111")
		self.assertEqual(signal.action, "BUY_YES")
		self.assertEqual(signal.true_probability_yes, 0.65)
		self.assertEqual(signal.market_implied_probability, 0.45)
		self.assertEqual(signal.expected_value_usd, 0.15)
		self.assertIn("Strong geopolitical", signal.reasoning)

	@patch("google.genai.Client")
	def test_analyse_api_error_returns_safe_hold(self, mock_client_cls):
		mock_client = MagicMock()
		mock_client_cls.return_value = mock_client
		mock_client.models.generate_content.side_effect = RuntimeError("Quota exhausted or connection dropped")

		analyzer = GeminiSignalAnalyzer(api_key="test_api_key")
		snapshot = MarketSnapshot(
			condition_id="0x222",
			question="Will event Y happen?",
			yes_price=0.40,
			no_price=0.60,
			news_summary="Sample context",
		)

		signal = analyzer.analyse(snapshot)

		self.assertEqual(signal.condition_id, "0x222")
		self.assertEqual(signal.action, "HOLD")
		self.assertEqual(signal.true_probability_yes, 0.50)
		self.assertEqual(signal.expected_value_usd, 0.0)


if __name__ == "__main__":
	unittest.main()
