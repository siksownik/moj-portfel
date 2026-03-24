import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import minimize
from scipy.stats import norm

# --- CONFIGURATION & STYLING ---
st.set_page_config(page_title="Siksu Portfolio Advisor", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #1E88E5;'>📊 SIKSU PORTFOLIO ADVISOR</h1>
    <h3 style='text-align: center; color: #555;'>Automated Financial Intelligence</h3>
    <hr>
""", unsafe_allow_html=True)

st.markdown("### 🚀 PROJECT BY: SIKSU")

# --- MODE SELECTION ---
st.markdown("#### 🛠️ Select Operation Mode:")
# FIXED: Label matches the logic checks below exactly
manual_label = "Custom portfolio with predetermined weights (Manual)"
app_mode = st.radio(
    label="Choose mode:",
    options=["Optimization (Auto)", manual_label],
    horizontal=True,
    label_visibility="collapsed"
)

# --- SECTION 1: RISK AVERSION ---
st.header("🛡️ STEP 1: Determine Your Risk Aversion (A)")

with st.expander("📝 How to determine these values? (Instructions)", expanded=True):
    st.markdown("""
    **Part A: Vanguard Investor Questionnaire** https://investor.vanguard.com/tools-calculators/investor-questionnaire Select your 'A' value based on your recommended stock allocation:
    * **100% Stocks** → Choose **1.5**
    * **80% Stocks** → Choose **2.2**
    * **60% Stocks** → Choose **3.0**
    * **40% Stocks** → Choose **4.5**
    * **20% Stocks** → Choose **8.0**

    **Part B: Lottery Test (Psychology of Loss)** At what loss amount (in a 50/50 double-or-nothing bet of $1,000) would you **STOP** playing?
    * Stop at **$100 loss** → Choose **7.5**
    * Stop at **$200 loss** → Choose **3.8**
    * Stop at **$330 loss** → Choose **2.0**
    * Stop at **$500 loss** → Choose **1.0**
    """)

col1, col2 = st.columns(2)
v_options = {"1.5 (100% Stocks)": 1.5, "2.2 (80% Stocks)": 2.2, "3.0 (60% Stocks)": 3.0, "4.5 (40% Stocks)": 4.5, "8.0 (20% Stocks)": 8.0}
a_vanguard = v_options[col1.selectbox("Vanguard Result (A1):", list(v_options.keys()), index=2)]

l_options = {"7.5 ($100 max loss)": 7.5, "3.8 ($200 max loss)": 3.8, "2.0 ($330 max loss)": 2.0, "1.0 ($500 max loss)": 1.0}
a_lottery = l_options[col2.selectbox("Lottery Test Result (A2):", list(l_options.keys()), index=2)]

risk_aversion = (a_vanguard + a_lottery) / 2
st.success(f"🔍 Combined Risk Aversion (A): **{risk_aversion:.2f}**")

# --- SECTION 2: INPUT DATA ---
st.header("⚙️ STEP 2: Input Parameters")
col_n1, col_n2 = st.columns(2)
net_worth = col_n1.number_input("Total Portfolio Value (USD)", value=300000.0, step=1000.0)
rf_rate = col_n2.number_input("Risk-free Rate (e.g., 0.056 for 5.6%)", value=0.056, format="%.4f")

period_input = st.selectbox("Historical Analysis Period", ["1y", "2y", "5y", "10y", "max"], index=2)
tickers_input = st.text_input("Tickers (space separated)", value="SPY GLD NVDA MSFT").upper().split()

# --- SECTION 3: ASSET WEIGHTS ---
st.header("⚖️ STEP 3: Asset Weights & Constraints")
user_weights = {}

if tickers_input:
    if app_mode == manual_label:
        cols = st.columns(len(tickers_input))
        for i, t in enumerate(tickers_input):
            user_weights[t] = cols[i].number_input(f"Weight {t}", -1.0, 1.0, 1.0/len(tickers_input), key=f"Manual_{t}")
    else:
        for t in tickers_input:
            with st.expander(f"Constraints for {t}"):
                st.number_input(f"Min Weight {t}", -1.0, 1.0, 0.0, key=f"min_{t}")
                st.number_input(f"Max Weight {t}", -1.0, 1.0, 1.0, key=f"max_{t}")

# --- SECTION 4: STRATEGY & RISK SETTINGS ---
st.header("📉 STEP 4: Strategy & Risk Settings")
method_choice = "MAX SHARPE RATIO (Classic)"
allow_leverage = False

if app_mode == "Optimization (Auto)":
    c_p1, c_p2 = st.columns(2)
    allow_short = c_p1.checkbox("Allow Asset Short-Selling? - Set negative min. weights manually.")
    allow_leverage = c_p2.checkbox("Allow Cash Leveraging?")
    method_choice = st.radio("Optimization Method:", ["MAX SHARPE RATIO (Classic)", "MINIMUM VARIANCE (Safe)", "EQUAL WEIGHTS"])

st.divider()
c_v1, c_v2, c_v3 = st.columns(3)
var_method = c_v1.selectbox("VaR Method", ["Parametric (Normal Dist)", "Historical Simulation", "Monte Carlo"])
var_conf = c_v2.select_slider("Confidence Level", options=[0.90, 0.95, 0.99], value=0.95)
var_days = c_v3.number_input("Time Horizon (Days)", min_value=1, max_value=30, value=1)

market_crash = st.slider("Simulate Market Drop (S&P 500) %", 0, 80, 20)

# --- CALCULATIONS ---
if st.button("🚀 GENERATE FULL REPORT"):
    with st.spinner('Fetching data and calculating...'):
        all_data = {}
        to_fetch = list(set(tickers_input + ["SPY"]))
        for t in to_fetch:
            try:
                all_data[t] = yf.download(t, period="max", progress=False)['Close'].squeeze()
            except: st.error(f"Error fetching: {t}")

    if all_data:
        df_f = pd.DataFrame(all_data).ffill().dropna()
        if period_input != 'max':
            cutoff = pd.Timestamp.now() - pd.DateOffset(years=int(period_input.replace('y','')))
            df_f = df_f[df_f.index >= cutoff]
        
        returns = df_f[tickers_input].pct_change().dropna()
        spy_ret = df_f["SPY"].pct_change().dropna()
        mu = returns.mean() * 252
        cov = returns.cov() * 252

        # FIXED: Logic check matches the variable label
        if app_mode == manual_label:
            weights = np.array([user_weights[t] for t in tickers_input])
            weights /= weights.sum()
        else:
            bounds = [(st.session_state[f"min_{t}"], st.session_state[f"max_{t}"]) for t in tickers_input]
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
            if "SHARPE" in method_choice:
                res = minimize(lambda w: -(np.sum(mu*w)-rf_rate)/np.sqrt(np.dot(w.T, np.dot(cov,w))), len(tickers_input)*[1./len(tickers_input)], bounds=bounds, constraints=cons)
                weights = res.x
            elif "VARIANCE" in method_choice:
                res = minimize(lambda w: np.sqrt(np.dot(w.T, np.dot(cov,w))), len(tickers_input)*[1./len(tickers_input)], bounds=bounds, constraints=cons)
                weights = res.x
            else: weights = np.array(len(tickers_input)*[1./len(tickers_input)])

        p_ret_annual = np.sum(mu * weights)
        p_vol_annual = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        opt_risky_w = max(0, min((p_ret_annual - rf_rate) / (risk_aversion * (p_vol_annual**2)), 1.5 if allow_leverage else 1.0))
        
        portfolio_returns = returns @ weights

        # --- VaR CALCULATION LOGIC ---
        if var_method == "Parametric (Normal Dist)":
            z_score = norm.ppf(1 - var_conf)
            m_h = (p_ret_annual / 252) * var_days
            v_h = (p_vol_annual / np.sqrt(252)) * np.sqrt(var_days)
            var_pct = m_h + z_score * v_h
        elif var_method == "Historical Simulation":
            h_returns = portfolio_returns * np.sqrt(var_days)
            var_pct = np.percentile(h_returns, (1 - var_conf) * 100)
        else: # Monte Carlo
            sim_returns = np.random.normal(portfolio_returns.mean(), portfolio_returns.std(), 10000)
            h_sim = sim_returns * np.sqrt(var_days)
            var_pct = np.percentile(h_sim, (1 - var_conf) * 100)

        # --- RESULTS ---
        st.divider()
        st.header("📊 ANALYSIS RESULTS")
        
        st.subheader("1️⃣ Risky Portfolio Structure")
        st.table(pd.DataFrame({"Ticker": tickers_input, "Weight": [f"{v:.2%}" for v in weights]}).set_index("Ticker"))
        
        st.subheader("2️⃣ Total Allocation (Tobin's Model)")
        c1, c2 = st.columns(2)
        c1.metric("RISKY ASSETS", f"{opt_risky_w:.2%}", f"${net_worth * opt_risky_w:,.2f}")
        c2.metric("CASH / SAFE ASSETS", f"{1-opt_risky_w:.2%}", f"${net_worth * (1-opt_risky_w):,.2f}")

        # Backtest
        st.subheader("📈 3️⃣ Performance Backtest")
        port_vals = (1 + portfolio_returns).cumprod() * (net_worth * opt_risky_w)
        st.plotly_chart(px.line(port_vals, title="Historical Value Growth (Risky Segment)"))

        # Stress Test & VaR
        st.subheader("🔥 4️⃣ Risk & Stress Test")
        common = portfolio_returns.index.intersection(spy_ret.index)
        beta = np.cov(portfolio_returns.loc[common], spy_ret.loc[common])[0,1] / np.var(spy_ret.loc[common])
        
        st.error(f"**Market Crash Scenario:** If S&P 500 drops by {market_crash}%, your portfolio (Beta: {beta:.2f}) is estimated to drop by **{(market_crash/100*beta):.2%}**, losing **${(net_worth * opt_risky_w * market_crash/100*beta):,.2f}**.")
        
        res_v1, res_v2 = st.columns(2)
        res_v1.metric(f"{var_days}-Day VaR (%)", f"{abs(var_pct):.2%}")
        res_v2.metric(f"{var_days}-Day VaR ($)", f"${(net_worth * opt_risky_w * abs(var_pct)):,.2f}")
        
        st.warning(f"**Interpretation:** Using the **{var_method}**, there is a {(1-var_conf):.0%} probability that your risky portfolio will lose more than **{abs(var_pct):.2%}** over a **{var_days}-day** period.")