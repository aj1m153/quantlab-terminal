# ⬡ QuantLab Terminal

A professional quantitative research Streamlit app with four integrated modules.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

<<<<<<< HEAD
### Streamlit Cloud

1. Push to a public GitHub repo.
2. Connect at [share.streamlit.io](https://share.streamlit.io).
3. Set `app.py` as the entry point.

---

## Key Design Choices

- **No statsmodels dependency** — OLS is hand-rolled with NumPy (`lstsq` + manual t-stats).
- **No pandas-datareader** — only `yfinance` + a session spoof to reduce rate-limit hits.
- **Caching** — all heavy computations are wrapped in `@st.cache_data(ttl=3600)`.
- **Graceful fallback** — if yfinance with a spoofed session fails, retries with the plain API.

---

---Link to app: 

## Extending the Lab

- **Add a 5-factor model**: include Ψ₉ (IV spread) and Ψ₁₀ (earnings distance) for options data.
- **Real-time mode**: swap `yfinance` for a WebSocket feed and call `st.rerun()` every N seconds.
- **Multi-asset entanglement**: project two tickers into the same Hilbert space and measure `|⟨ψ_A|ψ_B⟩|²` as correlation.
- **Regime labelling**: cluster eigenstates with k-means → label epochs as "ground state", "excited", "tunnel event".
=======
## Modules

### 01 · Market Regime Clustering
- **Model**: Gaussian Hidden Markov Model (hmmlearn)
- **Features**: Rolling log-returns + realised volatility
- **Output**: Bull / Bear / Sideways regime labels per trading day
- **Config**: Ticker, history, HMM states (2–4), rolling window, EM iterations

### 02 · Statistical Pairs Trading
- **Test**: Engle-Granger cointegration (statsmodels)
- **Signal**: Rolling z-score of OLS-hedged spread
- **Output**: Entry/exit signals, cumulative P&L vs buy-and-hold
- **Config**: Two tickers, entry/exit z-score thresholds, z-score window

### 03 · Volatility Surface Builder
- **Data**: Live options chain via yfinance
- **Model**: Black-Scholes inversion (scipy.optimize.brentq)
- **Output**: 3D IV surface (Plotly) + volatility smile cross-sections
- **Config**: Ticker, risk-free rate, min open interest, option type

### 04 · CVaR Portfolio Optimisation
- **Solver**: CVXPY with SCS backend
- **Objective**: Minimise Conditional Value-at-Risk (Expected Shortfall)
- **Benchmarks**: Equal-weight, Max-Sharpe (scipy)
- **Output**: Optimal weights, cumulative return chart, stats table

## Notes
- Options data best with liquid US underlyings: SPY, QQQ, AAPL, TSLA
- CVaR solver uses linear programming reformulation (Rockafellar & Uryasev 2000)
- HMM states are mapped to regimes by mean rolling return (lowest = Bear, highest = Bull)
>>>>>>> d0479b8 (first commit)
