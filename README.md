🤖 Bot de Trading Algorítmico — Binance Futures + Gemini AI
Resumen del proyecto
Bot de scalping de reversión a la media para Binance Futures, escrito en Python 3. Capital inicial: $50 USD. Objetivo: escalar a $500. Usa 10x de apalancamiento con gestión de riesgo estricta.

Stack tecnológico
Componente	Tecnología
Ejecución de órdenes	ccxt v4.3+ (binanceusdm)
Cerebro analítico	Google Gemini Flash (gemini-2.0-flash) via google-generativeai
Configuración	python-dotenv + archivo .env
Exchange	Binance USDM Futures
Modo actual	Testnet (TESTNET_MODE=true en .env)
Arquitectura de clases (
bot.py
)
TradingBot                  ← Orquestador principal (loop cada 60s)
├── ExchangeConnector       ← Wrapper de ccxt para Binance Futures
├── GeminiSignalAnalyzer    ← Llama a Gemini y parsea el JSON de señal
├── RiskManager             ← Kill switch + trailing stop + position sizing
└── PositionManager         ← Registro en memoria de posiciones abiertas
Dataclasses
TradingSignal
 → { symbol, action: buy|sell|hold, confidence: 0-1, reasoning }
Position
 → { symbol, side: long|short, entry_price, size, order_id, trailing_stop }
TrailingStop
 → { entry_price, side, stop_price, highest_price, lowest_price }
Flujo de decisión (por símbolo, por ciclo)
1. fetch_balance() → check_kill_switch()
2. ¿Posición abierta?
   SÍ → update_trailing_stop() → ¿stop tocado? → close_position()
   NO → fetch_ohlcv() → GeminiSignalAnalyzer.analyse()
        → confidence >= 0.70? → calculate_position_size() → open_market_order()
Parámetros de riesgo
Constante	Valor	Descripción
LEVERAGE	10x	Apalancamiento en Binance
RISK_PER_TRADE	2%	Del balance por operación
KILL_SWITCH_THRESHOLD	20%	Drawdown máximo antes de parar todo
TRAILING_STEP_PCT	0.3%	Paso de avance del trailing stop
OHLCV_TIMEFRAME	5m	Temporalidad de las velas
OHLCV_LIMIT	100	Velas históricas enviadas a Gemini
LOOP_INTERVAL_SECONDS	60	Cadencia del loop principal
Umbral de confianza AI	70%	Señales de Gemini por debajo → HOLD
Watchlist
python
WATCHLIST = ['BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT']
# Nota: formato :USDT al final es obligatorio en ccxt para futuros perpetuos
Prompt enviado a Gemini (estructura)
Rol: "Expert quantitative trader specialising in crypto futures scalping"
Datos: Últimas 30 velas OHLCV en formato [ts, O, H, L, C, V]
Tarea: Detectar oportunidad de reversión a la media
Respuesta esperada (JSON estricto):
{
  "action": "buy" | "sell" | "hold",
  "confidence": float 0.0-1.0,
  "reasoning": "una oración"
}
Variables de entorno requeridas (.env)
env
BINANCE_API_KEY=...        # Key de Binance Futures Testnet
BINANCE_API_SECRET=...     # Secret de Binance Futures Testnet
GEMINI_API_KEY=...         # Google AI Studio API key
TESTNET_MODE=true          # Cambiar a false SOLO para producción real
Gestión de riesgo — detalles técnicos
Kill Switch
python
drawdown = (initial_balance - current_balance) / initial_balance
if drawdown >= 0.20:
    # Cierra todas las posiciones abiertas
    # Detiene el loop permanentemente
Trailing Stop (dinámico)
python
# Para posición LONG:
# - Stop inicial: entry_price * (1 - 0.003)
# - Si precio sube: stop sube (nunca baja)
# - Se cierra si current_price <= stop_price
# Para posición SHORT:
# - Stop inicial: entry_price * (1 + 0.003)
# - Si precio baja: stop baja (nunca sube)
# - Se cierra si current_price >= stop_price
Position Sizing (fixed-fractional)
python
risk_amount = balance * 0.02
distance = abs(entry_price - stop_price)
size = min(risk_amount / distance, (balance * LEVERAGE) / entry_price)
Activar modo LIVE (producción)
Crear keys en Binance Futures REAL (no testnet)
En .env: cambiar TESTNET_MODE=false
El bot imprime 🔴 LIVE MODE ACTIVE al iniciar como confirmación
Dependencias (
requirements.txt
)
ccxt>=4.3.0
google-generativeai>=0.8.0
python-dotenv>=1.0.0
Cómo ejecutar
bash
# 1. Instalar dependencias
pip install -r requirements.txt
# 2. Copiar y completar variables de entorno
cp .env.example .env
# 3. Ejecutar
python bot.py
Archivos del proyecto
bot-trading-futuros/
├── bot.py              ← Script principal completo (~465 líneas)
├── .env                ← API keys (no commitear al repo)
├── .env.example        ← Template de variables de entorno
├── requirements.txt    ← Dependencias Python
└── bot.log             ← Log generado en runtime
