# Polymarket EV Arbitrage & Bayesian Trading Bot

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Platform](https://img.shields.io/badge/platform-Polymarket%20CLOB%20%7C%20Polygon-purple.svg)](https://polymarket.com)
[![AI Engine](https://img.shields.io/badge/AI-Google%20Gemini%202.5%20Flash-orange.svg)](https://ai.google.dev/)
[![Risk Engine](https://img.shields.io/badge/Risk-Fractional%20Kelly%20Criterion-success.svg)]()
[![Tests](https://img.shields.io/badge/tests-13%20passed%20(unit%20%2B%20smoke)-brightgreen.svg)]()

An institutional-grade algorithmic trading bot engineered to detect and exploit market mispricings on **Polymarket** binary prediction markets. The system combines real-time order book analysis on the Polygon Central Limit Order Book (CLOB), resilient multi-source macro/crypto news aggregation, Bayesian probability estimation via **Google Gemini 2.5 Flash**, and mathematically rigorous position sizing using the **Fractional Kelly Criterion**.

---

## Architecture Overview

```
                                      +--------------------------+
                                      |   RSS Feeds Aggregator   |
                                      | (Yahoo, CoinDesk, etc.)  |
                                      +------------+-------------+
                                                   |
+--------------------------+                       v
|  Polymarket CLOB API     |            +--------------------+
|  - Midpoint Price Engine | ---------> |   MarketSnapshot   |
|  - Depth & Order Book    |            +---------+----------+
+--------------------------+                      |
                                                  v
                                    +----------------------------+
                                    |   Gemini Signal Analyzer   |
                                    | (Bayesian True Probability)|
                                    +-------------+--------------+
                                                  |
                                                  v
                                      +------------------------+
                                      |      RiskManager       |
                                      | - Edge Check (>= 5%)   |
                                      | - Fractional Kelly     |
                                      +-----------+------------+
                                                  |
                         +------------------------+------------------------+
                         | (Action: BUY_YES / BUY_NO & Positive EV)       | (Action: HOLD / No Edge)
                         v                                                 v
            +--------------------------+                         +-------------------+
            | Polymarket CLOB Executor |                         |  ConsoleReporter  |
            | - EIP-712 Order Signing  |                         |  - Log & Skip     |
            | - Market Order Executed  |                         +-------------------+
            +------------+-------------+
                         |
                         v
            +--------------------------+
            |        BetManager        |
            | - In-Memory Deduplication|
            | - Position Lifecycle     |
            +--------------------------+
```

---

## Core Capabilities & Engineering Highlights

* **CLOB Execution Engine**: Direct programmatic integration with Polymarket's native Central Limit Order Book via `py-clob-client`. Extracts real-time bid/ask midpoints, validates liquidity, and signs execution payloads on Polygon (Chain ID 137).
* **Bayesian AI Probability Engine**: Employs Google Gemini 2.5 Flash conditioned on live financial news to infer the true posterior probability ($p_{\text{true}}$) of binary outcomes, enforcing strict JSON output schemas with deterministic fail-safe fallbacks.
* **Fractional Kelly Criterion Sizing**: Prevents ruin and optimizes long-term compound growth by dynamically rightsizing exposure ($25\%$ Kelly) based on calculated quantitative edge and current portfolio balance.
* **Resilient Multi-Feed News Aggregator**: Ingests, strips HTML, and sanitizes live macroeconomic and crypto headlines from multiple independent RSS feeds (Yahoo Finance, CoinDesk, Cointelegraph, Reuters). Designed with defensive exception boundaries to ensure news outages never crash the trading daemon.
* **State & Idempotency Management**: `BetManager` maintains in-memory tracking of deployed capital and active market condition IDs, preventing duplicate order placement and over-exposure across scan intervals.
* **Zero Trust & Cybersecurity Best Practices**: Zero hardcoded secrets or environment variables committed to VCS. Environment isolation via `.env.example`, strict input sanitization, and defensive exception handling across all external endpoints.

---

## Quantitative & Risk Management Models

### 1. Expected Value (EV) Formulation
Binary prediction market contracts pay out $\$1.00$ upon affirmative resolution and $\$0.00$ on negative resolution. The bot calculates expected value per dollar staked:

$$\text{EV}_{\text{YES}} = \left(p_{\text{true}} \cdot (1 - p_{\text{market}})\right) - \left((1 - p_{\text{true}}) \cdot p_{\text{market}}\right)$$

$$\text{EV}_{\text{NO}} = \left((1 - p_{\text{true}}) \cdot (1 - (1 - p_{\text{market}}))\right) - \left(p_{\text{true}} \cdot (1 - p_{\text{market}})\right)$$

Orders are filtered out unless the absolute statistical edge exceeds the safety threshold:
$$\text{Edge} = |p_{\text{true}} - p_{\text{market}}| \ge 0.05 \quad (5\%)$$

### 2. Fractional Kelly Criterion for Binary Contracts
Given payout odds $b = \frac{1 - p_{\text{market}}}{p_{\text{market}}}$ and probability of success $p$:

$$f^* = \frac{b \cdot p - (1 - p)}{b} = \frac{\frac{1 - p_{\text{market}}}{p_{\text{market}}} \cdot p - (1 - p)}{\frac{1 - p_{\text{market}}}{p_{\text{market}}}} = \frac{p - p_{\text{market}}}{1 - p_{\text{market}}}$$

To account for parameter uncertainty and protect capital against black-swan tail risk, position sizing is scaled down using a fractional multiplier ($c = 0.25$):

$$\text{Stake} = \min\left(\text{Balance} \times \left(f^* \times 0.25\right),\, \text{Balance}\right)$$

---

## Project Structure

```
polymarket-ev-bot/
├── bot.py                     # Main trading engine & orchestrator
├── requirements.txt           # Core and development dependencies
├── .env.example               # Environment variable configuration template
├── .gitignore                 # VCS hygiene & secret protection rules
├── tests/
│   ├── unit/
│   │   ├── test_risk_manager.py     # Kelly calculations, edge gates & limits
│   │   ├── test_news_fetcher.py     # RSS parsing, HTML stripping & resilience
│   │   └── test_gemini_analyzer.py  # Structured prompt output & fallback safety
│   └── smoke/
│       └── test_trading_bot_smoke.py# End-to-end simulated scan & order cycle
└── README.md                  # Project architecture & documentation
```

---

## Getting Started

### Prerequisites
* Python 3.10+ (tested on Python 3.10 through 3.14)
* Polymarket account with Polygon wallet credentials (API key, secret, passphrase)
* Google AI Studio Gemini API key

### 1. Installation
Clone the repository and set up a virtual environment:

```bash
git clone https://github.com/your-username/polymarket-ev-bot.git
cd polymarket-ev-bot
python -m venv .venv

# On Linux/macOS:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Configuration
Copy `.env.example` to `.env` and fill in your API credentials:

```bash
cp .env.example .env
```

```ini
# Google Gemini API Key
GEMINI_API_KEY=your_gemini_api_key_here

# Polymarket CLOB Credentials
POLYMARKET_PRIVATE_KEY=your_polygon_wallet_private_key_here
POLYMARKET_API_KEY=your_clob_api_key_here
POLYMARKET_API_SECRET=your_clob_api_secret_here
POLYMARKET_API_PASSPHRASE=your_clob_api_passphrase_here

# Network & Market Configuration
POLYMARKET_HOST=https://clob.polymarket.com
POLYMARKET_CHAIN_ID=137
POLYMARKET_WATCHLIST=["0x1234567890abcdef...", "0xabcdef1234567890..."]
```

### 3. Running the Bot
Run the main execution loop:

```bash
python bot.py
```

Console output provides real-time telemetry:
```text
[BALANCE] $1,250.00 USDC
[SCANNING MARKETS]
  [BUY_YES] 0x7b4a2e1c9f… | true_p=68.00% | mkt_p=48.00% | edge=20.00% | EV=$0.2000 | stake=$96.15
    reason: Regulatory filings indicate unanimous approval probability skew.
  [ACTIVE BET] BUY_YES on 0x7b4a2e1c9f… | stake=$96.15 | shares=200.31 @ 0.4800
  [HOLD] 0x3d9c8a1b2e… | edge=+1.20% | Insufficient edge, market reasonably priced.
```

---

## Testing & Quality Assurance

The codebase includes comprehensive unit and smoke tests validating core algorithms, external mocking, and resilient fallback paths:

```bash
# Run tests with Python standard unittest runner
python -m unittest discover tests

# Or run tests using pytest
pytest
```

### Test Coverage Highlights
* **Risk Engine**: Validates positive edge, zero/negative edge rejection, mirror calculations for `BUY_NO`, and capital boundary constraints.
* **News Fetcher**: Verifies tag stripping, entity decoding, timestamp normalization, and non-blocking failure tolerance.
* **AI Signal Analyzer**: Confirms strict JSON schema extraction and graceful fallback to `HOLD` on API timeouts or quota exceptions.
* **End-to-End Smoke**: Simulates an entire market scan loop, verifying execution idempotency and zero duplicate bets.

---

## Contractor Skillset Demonstrated

| Domain | Implemented Concepts |
|---|---|
| **Quantitative Engineering** | Expected Value (EV) arbitrage, Bayesian probability updating, Fractional Kelly Criterion. |
| **Web3 & Prediction Markets** | Polygon Network (Chain ID 137), Polymarket CLOB, EIP-712 signing, order book midpoint liquidity extraction. |
| **Production Architecture** | Modular OOP design (Single Responsibility Principle), Dataclass state models, resilient RSS data pipelines. |
| **Generative AI Systems** | LLM-as-a-Judge / Bayesian analyst integration, strict JSON schema output enforcement, fallback error boundaries. |
| **Cybersecurity & DevOps** | Zero Trust credential isolation, environment hygiene, deterministic test suite (unit + smoke). |
