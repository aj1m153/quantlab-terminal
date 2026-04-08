"""
QuantLab Terminal — Rewrite
Modules: Market Regime · Pairs Trading · Vol Surface · CVaR Optimisation
"""

import time
import random
import warnings
import datetime as dt
import io
import urllib.request

import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# RATE-LIMIT RESILIENT yfinance HELPERS
# ──────────────────────────────────────────────────────────────────────────────

try:
    from yfinance.exceptions import YFRateLimitError as _YFRateErr
except ImportError:
    _YFRateErr = None

def _ticker(symbol: str) -> yf.Ticker:
    """Return a plain Ticker — yfinance manages its own curl_cffi session internally."""
    return yf.Ticker(symbol)


def _is_rate_limit(exc: Exception) -> bool:
    if _YFRateErr and isinstance(exc, _YFRateErr):
        return True
    msg = str(exc).lower()
    return any(k in msg for k in ("rate limit", "too many requests", "429", "ratelimit"))


def _backoff(attempt: int) -> None:
    wait = min(2 ** attempt + random.uniform(0, 2), 60)
    time.sleep(wait)


@st.cache_data(ttl=3600, show_spinner=False)
def yf_download(tickers, period: str, auto_adjust: bool = True) -> pd.DataFrame:
    for attempt in range(6):
        try:
            data = yf.download(
                tickers, period=period,
                auto_adjust=auto_adjust,
                progress=False,
                threads=False,
            )
            if not data.empty:
                return data
        except Exception as exc:
            if _is_rate_limit(exc):
                _backoff(attempt)
                continue
            raise
    raise RuntimeError("Yahoo Finance is rate-limiting this IP. Wait 1–2 min and retry.")


@st.cache_data(ttl=3600, show_spinner=False)
def yf_options_expiries(symbol: str):
    """
    Fetch option expiry dates using the spoofed session.
    Retries on both exceptions AND silent empty returns.
    """
    for attempt in range(6):
        try:
            exps = _ticker(symbol).options
            if exps:
                return exps
            if attempt < 4:
                _backoff(attempt)
                continue
            return ()
        except Exception as exc:
            if _is_rate_limit(exc):
                _backoff(attempt)
                continue
            raise
    return ()


@st.cache_data(ttl=3600, show_spinner=False)
def yf_option_chain(symbol: str, expiry: str):
    """Returns (calls_df, puts_df) — plain DataFrames are always serialisable."""
    for attempt in range(5):
        try:
            chain = _ticker(symbol).option_chain(expiry)
            return chain.calls.copy(), chain.puts.copy()
        except Exception as exc:
            if _is_rate_limit(exc):
                _backoff(attempt)
                continue
            raise
    return None, None


@st.cache_data(ttl=3600, show_spinner=False)
def yf_spot(symbol: str) -> float | None:
    for attempt in range(5):
        try:
            t   = _ticker(symbol)
            fi  = t.fast_info
            price = getattr(fi, "last_price", None) or getattr(fi, "regular_market_price", None)
            if price:
                return float(price)
            hist = t.history(period="1d")
            if not hist.empty:
                return float(hist["Close"].iloc[-1])
        except Exception as exc:
            if _is_rate_limit(exc):
                _backoff(attempt)
                continue
            raise
    return None


# ──────────────────────────────────────────────────────────────────────────────
# TICKER UNIVERSE
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def load_ticker_universe():
    urls = [
        "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt",
        "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt",
    ]
    rows = []
    try:
        for url in urls:
            with urllib.request.urlopen(url, timeout=8) as r:
                text = r.read().decode("utf-8")
            df = pd.read_csv(io.StringIO(text), sep="|")
            sym_col  = "Symbol" if "Symbol" in df.columns else "ACT Symbol"
            name_col = "Security Name"
            df = df[[sym_col, name_col]].dropna()
            df.columns = ["symbol", "name"]
            df = df[~df["symbol"].str.contains(r"[\$\^\~\s]", regex=True)]
            df = df[df["symbol"].str.match(r"^[A-Z]{1,5}$")]
            rows.append(df)
        all_df = pd.concat(rows, ignore_index=True).drop_duplicates("symbol").sort_values("symbol")
        symbols = all_df["symbol"].tolist()
        labels  = (all_df["symbol"] + " — " + all_df["name"].str[:50]).tolist()
    except Exception:
        fallback = sorted(set([
            "AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","JPM","V","PG",
            "MA","HD","XOM","CVX","MRK","ABBV","PEP","KO","AVGO","COST","LLY",
            "TMO","MCD","ACN","ABT","WMT","BAC","CSCO","NEE","PM","BMY","RTX",
            "TXN","QCOM","HON","UPS","AMGN","SBUX","MS","GS","BLK","CAT","INTU",
            "AMAT","MDT","DE","NOW","ISRG","ADP","REGN","VRTX","ZTS","SPGI",
            "SPY","QQQ","IWM","DIA","GLD","SLV","USO","TLT","HYG","LQD",
            "WFC","C","USB","PNC","TFC","COF","AXP","T","VZ","TMUS","CMCSA",
            "NFLX","DIS","F","GM","RIVN","SQ","PYPL","COIN","UBER","ABNB",
            "CRM","ORCL","ADBE","SNOW","DDOG","AMD","INTC","MRVL","TSM","ASML",
        ]))
        symbols = fallback
        labels  = fallback
    return labels, symbols


_TICKER_LABELS, _TICKER_SYMBOLS = load_ticker_universe()


def ticker_select(label: str, default: str, key: str) -> str:
    try:
        idx = _TICKER_SYMBOLS.index(default)
    except ValueError:
        idx = 0
    chosen = st.selectbox(label, options=_TICKER_LABELS, index=idx, key=key)
    return chosen.split(" — ")[0].strip()


# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG & THEME
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="QuantLab Terminal",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Reset cache on version bump
_CACHE_VER = "v4"
if st.session_state.get("_cv") != _CACHE_VER:
    st.cache_data.clear()
    st.session_state["_cv"] = _CACHE_VER

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:      #07090d;
    --surf:    #0d1117;
    --surf2:   #161b22;
    --border:  #21262d;
    --accent:  #00d4aa;
    --gold:    #f7c948;
    --red:     #ff6b6b;
    --text:    #e6edf3;
    --muted:   #7d8590;
    --bull:    #26a641;
    --bear:    #da3633;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stSidebar"] {
    background: var(--surf) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .block-container { padding: 1.25rem 1rem; }

/* sidebar nav buttons */
div[data-testid="stSidebar"] .stButton button {
    background: transparent !important;
    border: 1px solid var(--border) !important;
    color: var(--muted) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.7rem !important;
    text-align: left !important;
    width: 100% !important;
    padding: 0.55rem 0.9rem !important;
    border-radius: 4px !important;
    transition: all 0.15s !important;
    margin-bottom: 3px;
    letter-spacing: 0.02em;
}
div[data-testid="stSidebar"] .stButton button:hover,
div[data-testid="stSidebar"] .stButton button:focus {
    border-color: var(--accent) !important;
    color: var(--accent) !important;
    background: rgba(0,212,170,0.06) !important;
}

/* main CTA buttons */
[data-testid="stMainBlockContainer"] .stButton button {
    background: var(--accent) !important;
    color: #000 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 3px !important;
    padding: 0.5rem 1.4rem !important;
    letter-spacing: 0.04em;
}
[data-testid="stMainBlockContainer"] .stButton button:hover { opacity: 0.82 !important; }

/* form inputs */
input, textarea,
.stTextInput input,
.stNumberInput input {
    background: var(--surf2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.78rem !important;
    border-radius: 3px !important;
}
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--surf2) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
}

/* metrics */
[data-testid="stMetric"] {
    background: var(--surf) !important;
    border: 1px solid var(--border) !important;
    border-radius: 6px !important;
    padding: 1rem !important;
}
[data-testid="stMetricLabel"] {
    color: var(--muted) !important;
    font-size: 0.68rem !important;
    font-family: 'Space Mono', monospace !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
[data-testid="stMetricValue"] {
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
}

h1, h2, h3 { font-family: 'Space Mono', monospace !important; }

.pill {
    display: inline-block;
    background: rgba(0,212,170,0.1);
    border: 1px solid rgba(0,212,170,0.28);
    color: var(--accent);
    font-family: 'Space Mono', monospace;
    font-size: 0.66rem;
    padding: 2px 10px;
    border-radius: 20px;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.page-title {
    font-family: 'Space Mono', monospace;
    font-size: 1.3rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 2px;
}
.page-desc {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: var(--muted);
    margin-bottom: 1.5rem;
    line-height: 1.65;
    max-width: 780px;
}
hr.div { border: none; border-top: 1px solid var(--border); margin: 1.4rem 0; }

.stats-table {
    width: 100%;
    border-collapse: collapse;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
}
.stats-table th {
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    padding: 5px 10px;
    text-align: left;
    font-weight: 400;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.65rem;
}
.stats-table td {
    color: var(--text);
    padding: 5px 10px;
    border-bottom: 1px solid rgba(33,38,45,0.45);
}
.stats-table tr:last-child td { border-bottom: none; }
.pos { color: #26a641 !important; }
.neg { color: #da3633 !important; }

.placeholder {
    border: 1px dashed var(--border);
    border-radius: 8px;
    padding: 3rem;
    text-align: center;
    margin-top: 1rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    color: var(--muted);
}

/* slider label */
[data-testid="stSlider"] label { font-family: 'Space Mono',monospace; font-size: 0.73rem; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# SHARED PLOTLY THEME
# ──────────────────────────────────────────────────────────────────────────────

PLOT_LAYOUT = dict(
    paper_bgcolor="#07090d",
    plot_bgcolor="#07090d",
    font=dict(family="Space Mono", color="#7d8590", size=10),
    hovermode="x unified",
    margin=dict(l=10, r=10, t=36, b=10),
)

LEGEND_DEFAULTS = dict(
    orientation="h", y=1.04, bgcolor="rgba(0,0,0,0)",
    font=dict(color="#e6edf3", size=10),
)

def _style_axes(fig, rows=1, cols=1):
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            fig.update_xaxes(gridcolor="#161b22", showgrid=True, zeroline=False, row=r, col=c)
            fig.update_yaxes(gridcolor="#161b22", showgrid=True, zeroline=False, row=r, col=c)
    if hasattr(fig, "layout") and fig.layout.annotations:
        fig.update_annotations(font=dict(family="Space Mono", color="#7d8590", size=9))
    return fig


def placeholder(hint: str) -> None:
    st.markdown(
        f'<div class="placeholder">{hint}</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:1.05rem;font-weight:700;
         color:#00d4aa;letter-spacing:0.06em;padding:0.4rem 0 0.15rem;">
    ⬡ QUANTLAB
    </div>
    <div style="font-family:'DM Sans',sans-serif;font-size:0.7rem;color:#7d8590;
         margin-bottom:1.4rem;letter-spacing:0.02em;">
    Quantitative Research Terminal
    </div>
    <div style="font-family:'Space Mono',monospace;font-size:0.6rem;color:#7d8590;
         text-transform:uppercase;letter-spacing:0.12em;margin-bottom:6px;">Modules</div>
    """, unsafe_allow_html=True)

    PAGES = {
        "01 · Market Regime":    "regime",
        "02 · Pairs Trading":   "pairs",
        "03 · Vol Surface":     "vol_surface",
        "04 · CVaR Optimise":   "cvar",
    }

    if "page" not in st.session_state:
        st.session_state.page = "regime"

    for label, key in PAGES.items():
        if st.button(label, key=f"nav_{key}"):
            st.session_state.page = key

    st.markdown("<hr style='border-color:#21262d;margin:1.4rem 0 0.6rem;'>", unsafe_allow_html=True)
    if st.button("↺  Clear Cache", key="nav_clear"):
        st.cache_data.clear()
        st.success("Cache cleared — retry your request.")
    st.markdown("<hr style='border-color:#21262d;margin:0.6rem 0 1rem;'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-family:Space Mono,monospace;font-size:0.58rem;color:#7d8590;line-height:1.9;'>"
        "Data · yfinance<br>Models · hmmlearn · statsmodels<br>"
        "Solver · CVXPY / SCS<br>Vis · Plotly</div>",
        unsafe_allow_html=True,
    )


page = st.session_state.page


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — MARKET REGIME CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════

if page == "regime":
    st.markdown('<div class="pill">Module 01</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Market Regime Clustering</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">Gaussian Hidden Markov Model trained on rolling returns and '
        'realised volatility to detect latent market states — Bull, Bear, or Sideways — for each trading day.</div>',
        unsafe_allow_html=True,
    )

    col_cfg, col_out = st.columns([1, 2.8], gap="large")

    with col_cfg:
        st.markdown("**Configuration**")
        ticker_hmm   = ticker_select("Ticker", "SPY", key="hmm_tick")
        period_hmm   = st.selectbox("History", ["2y", "3y", "5y", "10y"], index=2, key="hmm_per")
        n_states     = st.selectbox("HMM States", [2, 3, 4], index=1, key="hmm_st")
        roll_win     = st.slider("Rolling Window (days)", 5, 30, 10, key="hmm_roll")
        n_iter       = st.slider("EM Iterations", 50, 300, 150, key="hmm_iter")
        run_hmm      = st.button("Run HMM", key="btn_hmm")

    with col_out:
        if run_hmm:
            with st.spinner("Fetching data & fitting HMM…"):
                try:
                    from hmmlearn.hmm import GaussianHMM
                    from sklearn.preprocessing import StandardScaler

                    raw = yf_download(ticker_hmm, period=period_hmm)
                    if raw.empty:
                        st.error("No data returned — check ticker symbol.")
                        st.stop()

                    close = raw["Close"].squeeze().dropna()
                    log_ret  = np.log(close / close.shift(1)).dropna()
                    roll_ret = log_ret.rolling(roll_win).mean()
                    roll_vol = log_ret.rolling(roll_win).std()
                    feats = pd.DataFrame({"ret": roll_ret, "vol": roll_vol}).dropna()

                    X_sc = StandardScaler().fit_transform(feats.values)
                    model = GaussianHMM(
                        n_components=n_states, covariance_type="full",
                        n_iter=n_iter, random_state=42,
                    )
                    model.fit(X_sc)
                    states = model.predict(X_sc)
                    feats["state"] = states

                    # Map states → regime labels by mean return rank
                    ranked = feats.groupby("state")["ret"].mean().sort_values()
                    keys = list(ranked.index)
                    if n_states == 2:
                        label_map = {keys[0]: "Bear", keys[-1]: "Bull"}
                    else:
                        label_map = {keys[0]: "Bear", keys[-1]: "Bull"}
                        for k in keys[1:-1]:
                            label_map[k] = "Sideways"
                    feats["regime"] = feats["state"].map(label_map)

                    price = close.loc[feats.index]
                    COLORS = {"Bull": "#26a641", "Bear": "#da3633", "Sideways": "#f7c948"}

                    fig = make_subplots(
                        rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.25, 0.2],
                        vertical_spacing=0.025,
                    )

                    # Price
                    fig.add_trace(go.Scatter(
                        x=price.index, y=price.values,
                        mode="lines", line=dict(color="#7d8590", width=1),
                        name="Price", showlegend=False,
                    ), row=1, col=1)

                    # Shaded regime bands
                    seg_id = feats["regime"].ne(feats["regime"].shift()).cumsum()
                    for gid in seg_id.unique():
                        seg = feats[seg_id == gid]
                        fig.add_vrect(
                            x0=seg.index[0], x1=seg.index[-1],
                            fillcolor=COLORS.get(seg["regime"].iloc[0], "#555"),
                            opacity=0.17, layer="below", line_width=0,
                        )

                    # Regime dots
                    for rname, col in COLORS.items():
                        mask = feats["regime"] == rname
                        if mask.any():
                            fig.add_trace(go.Scatter(
                                x=feats.index[mask], y=price[mask],
                                mode="markers",
                                marker=dict(color=col, size=3, opacity=0.55),
                                name=rname,
                            ), row=1, col=1)

                    # Rolling vol
                    fig.add_trace(go.Scatter(
                        x=feats.index, y=feats["vol"] * np.sqrt(252),
                        mode="lines", line=dict(color="#f7c948", width=1.2),
                        name="Ann. Vol", showlegend=False,
                    ), row=2, col=1)

                    # Regime colour bar
                    state_cols = [COLORS.get(r, "#555") for r in feats["regime"]]
                    fig.add_trace(go.Bar(
                        x=feats.index, y=[1] * len(feats),
                        marker_color=state_cols, showlegend=False, name="Regime",
                    ), row=3, col=1)

                    fig.update_layout(
                        **PLOT_LAYOUT, height=560,
                        legend=dict(orientation="h", y=1.03, bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="#e6edf3", size=10)),
                    )
                    _style_axes(fig, rows=3)
                    fig.update_yaxes(title_text="Price",    row=1, col=1)
                    fig.update_yaxes(title_text="Ann. Vol", row=2, col=1)
                    fig.update_yaxes(showticklabels=False,  row=3, col=1)
                    st.plotly_chart(fig, use_container_width=True)

                    # Regime stat cards
                    st.markdown("<hr class='div'>", unsafe_allow_html=True)
                    stat_cols = st.columns(len(COLORS))
                    for i, (rname, col) in enumerate(COLORS.items()):
                        mask = feats["regime"] == rname
                        if not mask.any():
                            continue
                        days    = mask.sum()
                        avg_ret = feats.loc[mask, "ret"].mean() * 252 * 100
                        avg_vol = feats.loc[mask, "vol"].mean() * np.sqrt(252) * 100
                        pct     = days / len(feats) * 100
                        ret_col = "#26a641" if avg_ret > 0 else "#da3633"
                        with stat_cols[i]:
                            st.markdown(f"""
                            <div style="background:#0d1117;border:1px solid #21262d;
                                 border-top:2px solid {col};border-radius:6px;padding:1rem;
                                 font-family:'Space Mono',monospace;">
                                <div style="color:{col};font-size:0.67rem;text-transform:uppercase;
                                     letter-spacing:0.1em;margin-bottom:0.45rem;">{rname}</div>
                                <div style="color:#e6edf3;font-size:1.35rem;font-weight:700;">{pct:.0f}%</div>
                                <div style="color:#7d8590;font-size:0.62rem;margin-top:3px;">of trading days</div>
                                <hr style="border-color:#21262d;margin:0.55rem 0;">
                                <div style="display:flex;justify-content:space-between;font-size:0.65rem;">
                                    <span style="color:#7d8590;">Ann. Ret</span>
                                    <span style="color:{ret_col};">{avg_ret:+.1f}%</span>
                                </div>
                                <div style="display:flex;justify-content:space-between;font-size:0.65rem;margin-top:3px;">
                                    <span style="color:#7d8590;">Ann. Vol</span>
                                    <span style="color:#e6edf3;">{avg_vol:.1f}%</span>
                                </div>
                            </div>""", unsafe_allow_html=True)

                    converged = model.monitor_.converged
                    ll = model.monitor_.history[-1]
                    st.markdown(
                        f"<div style='margin-top:1rem;font-family:Space Mono,monospace;font-size:0.65rem;color:#7d8590;'>"
                        f"Log-likelihood <span style='color:#00d4aa;'>{ll:.2f}</span> &nbsp;·&nbsp; "
                        f"Converged <span style='color:{'#26a641' if converged else '#da3633'};'>"
                        f"{'Yes' if converged else 'No'}</span> &nbsp;·&nbsp; "
                        f"States {n_states} &nbsp;·&nbsp; Window {roll_win}d</div>",
                        unsafe_allow_html=True,
                    )

                except Exception as exc:
                    st.error(f"Error: {exc}")
        else:
            placeholder("Configure parameters and click <strong style='color:#00d4aa;'>Run HMM</strong> to detect market regimes")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — PAIRS TRADING
# ══════════════════════════════════════════════════════════════════════════════

elif page == "pairs":
    st.markdown('<div class="pill">Module 02</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Statistical Pairs Trading</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">Engle-Granger cointegration test identifies structurally linked assets. '
        'Deviations in the spread beyond a z-score threshold generate long/short signals that profit as the spread mean-reverts.</div>',
        unsafe_allow_html=True,
    )

    col_cfg, col_out = st.columns([1, 2.8], gap="large")

    with col_cfg:
        st.markdown("**Configuration**")
        tick1        = ticker_select("Asset A", "KO",  key="p_t1")
        tick2        = ticker_select("Asset B", "PEP", key="p_t2")
        period_p     = st.selectbox("History", ["2y", "3y", "5y"], index=1, key="p_per")
        z_entry      = st.slider("Entry |Z-Score|", 1.0, 3.0, 2.0, 0.25, key="p_ze")
        z_exit       = st.slider("Exit |Z-Score|",  0.0, 1.5, 0.5, 0.25, key="p_zx")
        roll_z       = st.slider("Rolling Window (days)", 20, 120, 60, key="p_rz")
        run_pairs    = st.button("Run Backtest", key="btn_pairs")

    with col_out:
        if run_pairs:
            with st.spinner("Running pairs analysis…"):
                try:
                    from statsmodels.tsa.stattools import coint
                    from statsmodels.regression.linear_model import OLS
                    import statsmodels.api as sm

                    raw = yf_download([tick1, tick2], period=period_p)["Close"].dropna()

                    # Align columns to requested tickers (yfinance may reorder)
                    col_map = {}
                    for req in [tick1, tick2]:
                        for av in raw.columns:
                            if av.upper() == req.upper() and av not in col_map.values():
                                col_map[av] = req
                                break
                    raw = raw.rename(columns=col_map)
                    if tick1 not in raw.columns or tick2 not in raw.columns:
                        st.error(f"Tickers not found in data. Got: {list(raw.columns)}")
                        st.stop()

                    _, pvalue, _ = coint(raw[tick1], raw[tick2])
                    X = sm.add_constant(raw[tick2])
                    res = OLS(raw[tick1], X).fit()
                    hedge = res.params[tick2]

                    spread    = raw[tick1] - hedge * raw[tick2]
                    roll_mean = spread.rolling(roll_z).mean()
                    roll_std  = spread.rolling(roll_z).std()
                    zscore    = (spread - roll_mean) / roll_std

                    # Stateful signal — ffill carries open position
                    sig_raw = pd.Series(np.nan, index=raw.index)
                    sig_raw[zscore >  z_entry] = -1
                    sig_raw[zscore < -z_entry] =  1
                    sig_raw[zscore.abs() <= z_exit] = 0
                    signal = sig_raw.ffill().fillna(0)

                    ret_a   = raw[tick1].pct_change().fillna(0)
                    ret_b   = raw[tick2].pct_change().fillna(0)
                    sp_ret  = ret_a - hedge * ret_b
                    st_ret  = signal.shift(1) * sp_ret
                    cum_st  = (1 + st_ret).cumprod()
                    cum_a   = (1 + ret_a).cumprod()
                    cum_b   = (1 + ret_b).cumprod()

                    sharpe    = (st_ret.mean() / st_ret.std() * np.sqrt(252)) if st_ret.std() > 0 else 0
                    tot_ret   = (cum_st.iloc[-1] - 1) * 100
                    max_dd    = ((cum_st / cum_st.cummax()) - 1).min() * 100
                    n_trades  = int((signal.diff() != 0).sum())

                    fig = make_subplots(
                        rows=4, cols=1, shared_xaxes=True,
                        row_heights=[0.3, 0.22, 0.28, 0.2],
                        vertical_spacing=0.024,
                        subplot_titles=["Normalised Prices", "Spread", "Z-Score", "Cumulative P&L"],
                    )

                    for t, col in [(tick1, "#00d4aa"), (tick2, "#f7c948")]:
                        fig.add_trace(go.Scatter(
                            x=raw.index, y=raw[t] / raw[t].iloc[0],
                            mode="lines", line=dict(color=col, width=1.4), name=t,
                        ), row=1, col=1)

                    fig.add_trace(go.Scatter(x=spread.index, y=spread, mode="lines",
                        line=dict(color="#7d8590", width=1), showlegend=False, name="Spread"), row=2, col=1)
                    fig.add_trace(go.Scatter(x=roll_mean.index, y=roll_mean, mode="lines",
                        line=dict(color="#00d4aa", width=1, dash="dash"), showlegend=False, name="Mean"), row=2, col=1)

                    fig.add_trace(go.Scatter(x=zscore.index, y=zscore, mode="lines",
                        line=dict(color="#e6edf3", width=1), showlegend=False, name="Z"), row=3, col=1)
                    for lvl, col, dash in [
                        (z_entry, "#26a641", "dash"), (-z_entry, "#da3633", "dash"),
                        (z_exit, "#7d8590", "dot"),   (-z_exit, "#7d8590", "dot"),
                        (0, "#7d8590", "solid"),
                    ]:
                        fig.add_hline(y=lvl, line_color=col, line_dash=dash, line_width=0.8, row=3, col=1)

                    long_idx  = zscore[zscore < -z_entry]
                    short_idx = zscore[zscore >  z_entry]
                    fig.add_trace(go.Scatter(x=long_idx.index, y=long_idx, mode="markers",
                        marker=dict(color="#26a641", size=5, symbol="triangle-up"),
                        showlegend=False, name="Long"), row=3, col=1)
                    fig.add_trace(go.Scatter(x=short_idx.index, y=short_idx, mode="markers",
                        marker=dict(color="#da3633", size=5, symbol="triangle-down"),
                        showlegend=False, name="Short"), row=3, col=1)

                    fig.add_trace(go.Scatter(x=cum_st.index, y=cum_st, mode="lines",
                        line=dict(color="#00d4aa", width=2), showlegend=False, name="Strategy"), row=4, col=1)
                    fig.add_trace(go.Scatter(x=cum_a.index, y=cum_a, mode="lines",
                        line=dict(color="#7d8590", width=1, dash="dot"), showlegend=False, name=tick1), row=4, col=1)
                    fig.add_trace(go.Scatter(x=cum_b.index, y=cum_b, mode="lines",
                        line=dict(color="#444", width=1, dash="dot"), showlegend=False, name=tick2), row=4, col=1)
                    fig.add_hline(y=1, line_color="#21262d", line_width=0.8, row=4, col=1)

                    fig.update_layout(**PLOT_LAYOUT, height=640,
                        legend=dict(orientation="h", y=1.03, bgcolor="rgba(0,0,0,0)",
                                    font=dict(color="#e6edf3", size=10)))
                    _style_axes(fig, rows=4)
                    st.plotly_chart(fig, use_container_width=True)

                    st.markdown("<hr class='div'>", unsafe_allow_html=True)
                    coint_ok  = pvalue < 0.05
                    coint_col = "#26a641" if coint_ok else "#da3633"
                    coint_lbl = "Cointegrated ✓" if coint_ok else "Not Cointegrated ✗"

                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric("Coint. p-value", f"{pvalue:.4f}", delta=coint_lbl)
                    m2.metric("Hedge Ratio",    f"{hedge:.4f}")
                    m3.metric("Total Return",   f"{tot_ret:+.1f}%")
                    m4.metric("Sharpe Ratio",   f"{sharpe:.2f}")
                    m5.metric("Max Drawdown",   f"{max_dd:.1f}%")
                    st.caption(
                        f"Trades: {n_trades}  ·  Entry |Z| > {z_entry}  ·  "
                        f"Exit |Z| < {z_exit}  ·  Window: {roll_z}d"
                    )

                except Exception as exc:
                    st.error(f"Error: {exc}")
                    import traceback; st.code(traceback.format_exc())
        else:
            placeholder("Set your pair and parameters, then click <strong style='color:#00d4aa;'>Run Backtest</strong>")


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — IMPLIED VOLATILITY SURFACE
# ══════════════════════════════════════════════════════════════════════════════

elif page == "vol_surface":
    st.markdown('<div class="pill">Module 03</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Implied Volatility Surface</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">Fetches live options chains, inverts Black-Scholes analytically for each strike '
        '& expiry, and renders the full IV surface in 3D — revealing the volatility smile and term structure.</div>',
        unsafe_allow_html=True,
    )

    col_cfg, col_out = st.columns([1, 2.8], gap="large")

    with col_cfg:
        st.markdown("**Configuration**")
        ticker_vs   = ticker_select("Ticker (options-liquid)", "SPY", key="vs_tick")
        rf_rate     = st.number_input("Risk-Free Rate (%)", value=5.0, step=0.25, key="vs_rf") / 100
        min_oi      = st.number_input("Min Open Interest", value=100, step=50, key="vs_oi")
        colorscale  = st.selectbox("Colorscale", ["Plasma", "Viridis", "Turbo", "RdYlGn"], key="vs_cs")
        opt_type    = st.selectbox("Option Type", ["calls", "puts", "both"], key="vs_ot")
        run_vs      = st.button("Build Surface", key="btn_vs")

    with col_out:
        if run_vs:
            with st.spinner("Fetching options chain & computing IV…"):
                try:
                    from scipy.optimize import brentq
                    from scipy.stats import norm
                    from scipy.interpolate import griddata

                    def bs_price(S, K, T, r, sigma, flag="c"):
                        if T <= 0 or sigma <= 0:
                            return max(S - K, 0) if flag == "c" else max(K - S, 0)
                        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
                        d2 = d1 - sigma * np.sqrt(T)
                        if flag == "c":
                            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
                        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

                    def implied_vol(price, S, K, T, r, flag="c"):
                        if T <= 1e-6:
                            return np.nan
                        intrinsic = max(S - K, 0) if flag == "c" else max(K - S, 0)
                        if price <= intrinsic + 1e-6:
                            return np.nan
                        try:
                            return brentq(
                                lambda s: bs_price(S, K, T, r, s, flag) - price,
                                1e-6, 10.0, maxiter=200, xtol=1e-6,
                            )
                        except Exception:
                            return np.nan

                    spot = yf_spot(ticker_vs)
                    if not spot:
                        st.error("Could not fetch spot price — check ticker.")
                        st.stop()

                    exps = yf_options_expiries(ticker_vs)
                    if not exps:
                        st.error(
                            f"**No options data returned for `{ticker_vs}`.**  \n"
                            "This usually means one of:\n"
                            "- Yahoo Finance is rate-limiting this IP — wait 60 s and click **Build Surface** again\n"
                            "- The ticker has no listed options (try SPY · QQQ · AAPL · TSLA · AMZN)\n"
                            "- A stale empty result is cached — click **↺ Clear Cache** in the sidebar, then retry"
                        )
                        st.stop()

                    today = dt.date.today()
                    flags = (
                        ["calls"] if opt_type == "calls" else
                        ["puts"]  if opt_type == "puts"  else
                        ["calls", "puts"]
                    )
                    records = []
                    prog = st.progress(0, text="Loading expiries…")

                    for idx, exp in enumerate(exps):
                        prog.progress((idx + 1) / len(exps), text=f"Expiry {exp}…")
                        calls_df, puts_df = yf_option_chain(ticker_vs, exp)
                        if calls_df is None:
                            continue
                        exp_date = dt.datetime.strptime(exp, "%Y-%m-%d").date()
                        T = (exp_date - today).days / 365.0
                        if T <= 0:
                            continue
                        chain_map = {"calls": calls_df, "puts": puts_df}
                        for flag in flags:
                            df_opt = chain_map[flag].copy()
                            df_opt["openInterest"] = pd.to_numeric(
                                df_opt["openInterest"], errors="coerce"
                            ).fillna(0)
                            df_opt = df_opt[df_opt["openInterest"] >= min_oi]
                            for _, row in df_opt.iterrows():
                                bid = row.get("bid") or 0
                                ask = row.get("ask") or 0
                                mid = (bid + ask) / 2 or float(row.get("lastPrice") or 0)
                                if mid <= 0:
                                    continue
                                K = float(row["strike"])
                                if K <= 0 or not (spot * 0.5 < K < spot * 2.0):
                                    continue
                                iv = implied_vol(mid, spot, K, T, rf_rate, "c" if flag == "calls" else "p")
                                if iv and 0.01 < iv < 5.0:
                                    records.append({
                                        "strike": K, "T": T, "iv": iv * 100,
                                        "expiry": exp, "type": flag, "moneyness": K / spot,
                                    })
                    prog.empty()

                    if not records:
                        st.error("No valid IV computed. Try lower min-OI or a more liquid ticker.")
                        st.stop()

                    iv_df = pd.DataFrame(records)
                    iv_df = iv_df[(iv_df["iv"] > 1) & (iv_df["iv"] < 150)]

                    # Interpolate to grid for 3-D surface
                    xi = np.linspace(iv_df["moneyness"].min(), iv_df["moneyness"].max(), 60)
                    yi = np.linspace(iv_df["T"].min(), iv_df["T"].max(), 40)
                    xi_g, yi_g = np.meshgrid(xi, yi)
                    zi_g = griddata(
                        (iv_df["moneyness"], iv_df["T"]), iv_df["iv"],
                        (xi_g, yi_g), method="linear",
                    )

                    surf = go.Surface(
                        x=xi_g, y=yi_g * 365, z=zi_g,
                        colorscale=colorscale, opacity=0.92, showscale=True,
                        contours=dict(z=dict(show=True, usecolormap=True, project_z=True)),
                        colorbar=dict(
                            title=dict(text="IV (%)", font=dict(family="Space Mono", size=10)),
                            tickfont=dict(family="Space Mono", size=9),
                            len=0.6, thickness=12,
                        ),
                    )
                    scatter = go.Scatter3d(
                        x=iv_df["moneyness"], y=iv_df["T"] * 365, z=iv_df["iv"],
                        mode="markers",
                        marker=dict(size=2, color=iv_df["iv"], colorscale=colorscale, opacity=0.4),
                        showlegend=False, name="Data pts",
                    )

                    fig = go.Figure(data=[surf, scatter])
                    ax_style = dict(gridcolor="#21262d", backgroundcolor="#0d1117")
                    fig.update_layout(
                        paper_bgcolor="#07090d",
                        scene=dict(
                            bgcolor="#0d1117",
                            xaxis=dict(title=dict(text="Moneyness (K/S)", font=dict(family="Space Mono", size=10)), **ax_style),
                            yaxis=dict(title=dict(text="Days to Expiry",  font=dict(family="Space Mono", size=10)), **ax_style),
                            zaxis=dict(title=dict(text="Implied Vol (%)",  font=dict(family="Space Mono", size=10)), **ax_style),
                        ),
                        font=dict(family="Space Mono", color="#7d8590", size=9),
                        height=600, margin=dict(l=0, r=0, t=36, b=0),
                        title=dict(
                            text=f"IV Surface · {ticker_vs} · Spot ${spot:.2f}",
                            font=dict(family="Space Mono", color="#e6edf3", size=12),
                        ),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # Smile cross-sections
                    st.markdown("<hr class='div'>", unsafe_allow_html=True)
                    st.markdown(
                        "<div style='font-family:Space Mono,monospace;font-size:0.72rem;color:#7d8590;"
                        "margin-bottom:0.5rem;text-transform:uppercase;letter-spacing:0.1em;'>"
                        "Volatility Smile — Nearest Expiries</div>",
                        unsafe_allow_html=True,
                    )
                    smile_colors = ["#00d4aa", "#f7c948", "#ff6b6b", "#a78bfa", "#60a5fa"]
                    smile_fig = go.Figure()
                    for i, exp in enumerate(sorted(iv_df["expiry"].unique())[:5]):
                        sub = iv_df[iv_df["expiry"] == exp].sort_values("moneyness")
                        smile_fig.add_trace(go.Scatter(
                            x=sub["moneyness"], y=sub["iv"],
                            mode="lines+markers",
                            line=dict(color=smile_colors[i % len(smile_colors)], width=1.5),
                            marker=dict(size=4), name=exp,
                        ))
                    smile_fig.add_vline(x=1.0, line_color="#7d8590", line_dash="dot", line_width=0.8)
                    smile_fig.update_layout(
                        **PLOT_LAYOUT, height=300,
                        xaxis=dict(title="Moneyness", gridcolor="#161b22"),
                        yaxis=dict(title="IV (%)",    gridcolor="#161b22"),
                    )
                    st.plotly_chart(smile_fig, use_container_width=True)

                    atm_iv = iv_df.loc[(iv_df["moneyness"] - 1).abs().idxmin(), "iv"]
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("ATM IV (nearest)", f"{atm_iv:.1f}%")
                    m2.metric("Data Points",      len(iv_df))
                    m3.metric("Expiries",          iv_df["expiry"].nunique())
                    m4.metric("Spot Price",        f"${spot:.2f}")

                except Exception as exc:
                    st.error(f"Error: {exc}")
                    import traceback; st.code(traceback.format_exc())
        else:
            placeholder(
                "Click <strong style='color:#00d4aa;'>Build Surface</strong> to compute the IV surface.<br>"
                "<span style='font-size:0.62rem;'>Best with liquid underlyings: SPY · QQQ · AAPL · TSLA</span>"
            )


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 4 — CVaR PORTFOLIO OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════════

elif page == "cvar":
    st.markdown('<div class="pill">Module 04</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-title">CVaR Portfolio Optimisation</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-desc">Minimises Conditional Value-at-Risk (Expected Shortfall) — the average loss '
        'in the worst α% of scenarios — via convex optimisation (CVXPY/SCS). '
        'Compared against equal-weight and max-Sharpe benchmarks.</div>',
        unsafe_allow_html=True,
    )

    col_cfg, col_out = st.columns([1, 2.8], gap="large")

    _CVAR_DEFAULTS = ["AAPL", "MSFT", "GOOGL", "AMZN", "JPM", "GS", "XOM", "JNJ"]
    _CVAR_DEF_LABELS = [
        lbl for lbl in _TICKER_LABELS
        if lbl.split(" — ")[0].strip() in _CVAR_DEFAULTS
    ]

    with col_cfg:
        st.markdown("**Configuration**")
        tickers_sel = st.multiselect(
            "Tickers (search & select)",
            options=_TICKER_LABELS,
            default=_CVAR_DEF_LABELS,
            key="cv_ticks",
            help="Type to search all US-listed stocks",
        )
        period_cv   = st.selectbox("History", ["2y", "3y", "5y"], index=1, key="cv_per")
        alpha_cv    = st.slider("CVaR Confidence α", 0.90, 0.99, 0.95, 0.01, key="cv_a",
                                help="Minimise expected loss in worst (1-α)% of days")
        max_wt      = st.slider("Max Weight per Asset (%)", 10, 60, 40, 5, key="cv_mw") / 100
        run_cvar    = st.button("Optimise Portfolio", key="btn_cvar")

    with col_out:
        if run_cvar:
            with st.spinner("Fetching data & solving CVaR optimisation…"):
                try:
                    import cvxpy as cp
                    from scipy.optimize import minimize as sp_minimize

                    tlist = [lbl.split(" — ")[0].strip() for lbl in tickers_sel if lbl]
                    if len(tlist) < 2:
                        st.error("Select at least 2 tickers.")
                        st.stop()

                    raw = yf_download(tlist, period=period_cv)["Close"]
                    if isinstance(raw, pd.Series):
                        raw = raw.to_frame()
                    raw   = raw.dropna(axis=1, how="all").dropna()
                    tlist = list(raw.columns)
                    n     = len(tlist)
                    if n < 2:
                        st.error("Insufficient data for selected tickers.")
                        st.stop()

                    rets    = raw.pct_change().dropna().values   # (T, n)
                    T_rows  = rets.shape[0]
                    k       = int(np.ceil((1 - alpha_cv) * T_rows))

                    # ── CVaR minimisation ──────────────────────────────
                    w   = cp.Variable(n)
                    aux = cp.Variable()
                    z   = cp.Variable(T_rows)
                    pr  = rets @ w
                    constraints = [
                        cp.sum(w) == 1, w >= 0, w <= max_wt,
                        z >= 0, z >= -pr - aux,
                    ]
                    prob = cp.Problem(
                        cp.Minimize(aux + (1 / float(k)) * cp.sum(z)),
                        constraints,
                    )
                    prob.solve(solver=cp.SCS, verbose=False)

                    if prob.status not in ("optimal", "optimal_inaccurate"):
                        st.error(f"Solver status: {prob.status}")
                        st.stop()

                    w_cvar = np.maximum(np.array(w.value).flatten(), 0)
                    w_cvar /= w_cvar.sum()

                    # Equal weight
                    w_eq = np.ones(n) / n

                    # Max Sharpe via SLSQP
                    mu  = rets.mean(axis=0) * 252
                    cov = np.cov(rets.T) * 252

                    def neg_sharpe(ww, mu, cov, rf=0.05):
                        pret = ww @ mu
                        pvol = np.sqrt(ww @ cov @ ww)
                        return -(pret - rf) / pvol if pvol > 0 else 1e9

                    res_ms = sp_minimize(
                        neg_sharpe, w_eq, args=(mu, cov), method="SLSQP",
                        bounds=[(0, max_wt)] * n,
                        constraints=[{"type": "eq", "fun": lambda ww: ww.sum() - 1}],
                        options={"maxiter": 500},
                    )
                    w_ms = np.maximum(res_ms.x, 0); w_ms /= w_ms.sum()

                    def port_stats(ww):
                        pr_s = rets @ ww
                        ann_ret = (1 + pr_s).prod() ** (252 / T_rows) - 1
                        ann_vol = pr_s.std() * np.sqrt(252)
                        sharpe  = (ann_ret - 0.05) / ann_vol if ann_vol > 0 else 0
                        k2      = max(1, int(np.floor((1 - alpha_cv) * T_rows)))
                        cvar_v  = -np.sort(pr_s)[:k2].mean()
                        cum     = (1 + pr_s).cumprod()
                        max_dd  = ((cum / np.maximum.accumulate(cum)) - 1).min()
                        return dict(
                            ann_ret=ann_ret * 100, ann_vol=ann_vol * 100,
                            sharpe=sharpe, cvar=cvar_v * 100, max_dd=max_dd * 100,
                        )

                    s_cvar = port_stats(w_cvar)
                    s_eq   = port_stats(w_eq)
                    s_ms   = port_stats(w_ms)

                    dates    = raw.index[1:]
                    cum_cvar = (1 + rets @ w_cvar).cumprod()
                    cum_eq   = (1 + rets @ w_eq).cumprod()
                    cum_ms   = (1 + rets @ w_ms).cumprod()

                    fig = make_subplots(
                        rows=1, cols=2, column_widths=[0.62, 0.38],
                        subplot_titles=["Cumulative Return", "CVaR-Optimal Weights"],
                    )

                    for cum, name, col, dash in [
                        (cum_cvar, "CVaR-Optimal", "#00d4aa", "solid"),
                        (cum_eq,   "Equal Weight", "#7d8590", "dot"),
                        (cum_ms,   "Max Sharpe",   "#f7c948", "dash"),
                    ]:
                        fig.add_trace(go.Scatter(
                            x=dates, y=cum, mode="lines",
                            line=dict(color=col, width=2, dash=dash), name=name,
                        ), row=1, col=1)
                    fig.add_hline(y=1, line_color="#21262d", line_width=0.8)

                    order = np.argsort(w_cvar)[::-1]
                    bar_colors = ["#00d4aa" if w_cvar[i] > 0.001 else "#21262d" for i in order]
                    fig.add_trace(go.Bar(
                        x=[tlist[i] for i in order],
                        y=[w_cvar[i] * 100 for i in order],
                        marker_color=bar_colors, showlegend=False, name="Weights",
                    ), row=1, col=2)

                    fig.update_layout(**PLOT_LAYOUT, legend=dict(**LEGEND_DEFAULTS), height=420)
                    _style_axes(fig, rows=1, cols=2)
                    fig.update_yaxes(title_text="Growth of $1", row=1, col=1)
                    fig.update_yaxes(title_text="Weight (%)",   row=1, col=2)
                    st.plotly_chart(fig, use_container_width=True)

                    # Comparison table
                    st.markdown("<hr class='div'>", unsafe_allow_html=True)
                    st.markdown(
                        "<div style='font-family:Space Mono,monospace;font-size:0.68rem;color:#7d8590;"
                        "text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.75rem;'>"
                        "Performance Comparison</div>",
                        unsafe_allow_html=True,
                    )

                    def fmt_val(v, pct=True, pos_good=True):
                        s   = f"{v:+.2f}%" if pct else f"{v:.3f}"
                        col = "#26a641" if (v > 0) == pos_good else "#da3633"
                        return f'<span style="color:{col};font-family:Space Mono,monospace;font-size:0.75rem;">{s}</span>'

                    table_rows = [
                        ("Ann. Return",      [s_cvar["ann_ret"], s_eq["ann_ret"], s_ms["ann_ret"]], True,  True),
                        ("Ann. Volatility",  [s_cvar["ann_vol"], s_eq["ann_vol"], s_ms["ann_vol"]], True,  False),
                        ("Sharpe Ratio",     [s_cvar["sharpe"],  s_eq["sharpe"],  s_ms["sharpe"]],  False, True),
                        (f"CVaR {int(alpha_cv*100)}%", [s_cvar["cvar"], s_eq["cvar"], s_ms["cvar"]], True, False),
                        ("Max Drawdown",     [s_cvar["max_dd"], s_eq["max_dd"], s_ms["max_dd"]],    True,  False),
                    ]

                    tbl = (
                        "<table class='stats-table'>"
                        "<tr><th>Metric</th>"
                        "<th style='color:#00d4aa;'>CVaR-Optimal</th>"
                        "<th>Equal Weight</th>"
                        "<th style='color:#f7c948;'>Max Sharpe</th></tr>"
                    )
                    for lbl, vals, pct, pos_good in table_rows:
                        tbl += f"<tr><td style='color:#7d8590;'>{lbl}</td>"
                        for v in vals:
                            tbl += f"<td>{fmt_val(v, pct, pos_good)}</td>"
                        tbl += "</tr>"
                    tbl += "</table>"
                    st.markdown(tbl, unsafe_allow_html=True)

                    # Per-asset detail
                    st.markdown("<hr class='div'>", unsafe_allow_html=True)
                    st.markdown(
                        "<div style='font-family:Space Mono,monospace;font-size:0.68rem;color:#7d8590;"
                        "text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.75rem;'>"
                        "CVaR-Optimal Allocation Detail</div>",
                        unsafe_allow_html=True,
                    )

                    alloc = (
                        "<table class='stats-table'>"
                        "<tr><th>Ticker</th><th>CVaR Wt</th><th>Equal Wt</th>"
                        "<th>Max Sharpe</th><th>Ann. Ret</th><th>Ann. Vol</th></tr>"
                    )
                    for i in order:
                        cr = rets[:, i]
                        ar = ((1 + cr).prod() ** (252 / T_rows) - 1) * 100
                        av = cr.std() * np.sqrt(252) * 100
                        cls_r = "pos" if ar > 0 else "neg"
                        alloc += (
                            f"<tr>"
                            f"<td style='color:#e6edf3;font-weight:500;'>{tlist[i]}</td>"
                            f"<td style='color:#00d4aa;'>{w_cvar[i]*100:.1f}%</td>"
                            f"<td style='color:#7d8590;'>{w_eq[i]*100:.1f}%</td>"
                            f"<td style='color:#f7c948;'>{w_ms[i]*100:.1f}%</td>"
                            f"<td class='{cls_r}'>{ar:+.1f}%</td>"
                            f"<td style='color:#7d8590;'>{av:.1f}%</td>"
                            f"</tr>"
                        )
                    alloc += "</table>"
                    st.markdown(alloc, unsafe_allow_html=True)
                    st.caption(
                        f"α = {alpha_cv:.0%}  ·  Max single weight = {max_wt:.0%}  ·  "
                        f"Solver: CVXPY/SCS  ·  Assets: {n}"
                    )

                except Exception as exc:
                    st.error(f"Error: {exc}")
                    import traceback; st.code(traceback.format_exc())
        else:
            placeholder(
                "Enter tickers and click <strong style='color:#00d4aa;'>Optimise Portfolio</strong><br>"
                "<span style='font-size:0.62rem;'>CVaR minimisation via convex optimisation (CVXPY)</span>"
            )
