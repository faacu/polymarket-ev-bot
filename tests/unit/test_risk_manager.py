import unittest
from bot import RiskManager, BayesianSignal


class TestRiskManager(unittest.TestCase):
	"""Validates Fractional Kelly Criterion and edge threshold logic."""

	def setUp(self):
		self.risk_manager = RiskManager(kelly_fraction=0.25)

	def test_calculate_stake_positive_edge_yes(self):
		signal = BayesianSignal(
			condition_id="0x123",
			true_probability_yes=0.70,
			market_implied_probability=0.50,
			expected_value_usd=0.20,
			action="BUY_YES",
			reasoning="Bullish news alignment",
		)
		# Kelly formula: (p - p_market) / (1 - p_market) = (0.7 - 0.5) / (1 - 0.5) = 0.2 / 0.5 = 0.40
		# Fractional Kelly (25%): 0.40 * 0.25 = 0.10 (10% of balance)
		balance = 1000.0
		stake = self.risk_manager.calculate_stake(balance, signal)
		self.assertAlmostEqual(stake, 100.0, places=2)

	def test_calculate_stake_positive_edge_no(self):
		signal = BayesianSignal(
			condition_id="0x456",
			true_probability_yes=0.20,
			market_implied_probability=0.40,
			expected_value_usd=0.20,
			action="BUY_NO",
			reasoning="Bearish catalyst detected",
		)
		# For NO side:
		# p_no = 1 - 0.20 = 0.80
		# p_market_no = 1 - 0.40 = 0.60
		# full_kelly = (0.80 - 0.60) / (1 - 0.60) = 0.20 / 0.40 = 0.50
		# fractional_kelly = 0.50 * 0.25 = 0.125
		balance = 1000.0
		stake = self.risk_manager.calculate_stake(balance, signal)
		self.assertAlmostEqual(stake, 125.0, places=2)

	def test_calculate_stake_no_edge_returns_zero(self):
		signal = BayesianSignal(
			condition_id="0x789",
			true_probability_yes=0.45,
			market_implied_probability=0.50,
			expected_value_usd=-0.05,
			action="BUY_YES",
			reasoning="Underpriced market",
		)
		stake = self.risk_manager.calculate_stake(1000.0, signal)
		self.assertEqual(stake, 0.0)

	def test_validate_edge_above_threshold(self):
		signal = BayesianSignal(
			condition_id="0xabc",
			true_probability_yes=0.60,
			market_implied_probability=0.50,
			expected_value_usd=0.10,
			action="BUY_YES",
			reasoning="Edge >= 5%",
		)
		self.assertTrue(self.risk_manager.validate_edge(signal))

	def test_validate_edge_below_threshold(self):
		signal = BayesianSignal(
			condition_id="0xdef",
			true_probability_yes=0.52,
			market_implied_probability=0.50,
			expected_value_usd=0.02,
			action="BUY_YES",
			reasoning="Edge < 5%",
		)
		self.assertFalse(self.risk_manager.validate_edge(signal))


if __name__ == "__main__":
	unittest.main()
