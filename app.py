import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import minimize

# --- KONFIGURACJA (IDENTYCZNA Z TWOIM SCREENEM) ---
st.set_page_config(page_title="Siksu Portfolio Advisor", layout="centered")

st.markdown("""
    <h1 style='text-align: center; color: #1E88E5;'>📊 SIKSU PORTFOLIO ADVISOR</h1>
    <h3 style='text-align: center; color: #555;'>Automated Financial Intelligence</h3>
    <hr>
""", unsafe_allow_html=True)

st.markdown("### 🚀 PROJECT BY: SIKSU")

# --- CENTRALNY WYBÓR TRYBU ---
st.markdown("#### 🛠️ Wybierz metodę działania:")
app_mode = st.radio(
    label="Wybierz tryb pracy:",
    options=["Optymalizacja (Auto)", "Własny Portfel (Manual)"],
    horizontal=True,
    label_visibility="collapsed"
)

# --- STEP 1: RISK AVERSION ---
st.header("🛡️ STEP 1: Determine Your Risk Aversion (A)")
with st.expander("📝 How to determine these values? (Instructions)", expanded=True):
    st.markdown("""
    **Part A: Vanguard Investor Questionnaire** Please complete the official test here: [Vanguard Investor Questionnaire](https://ownyourfuture.vanguard.com/home/investor-questionnaire/en/resultPage).  
    Select your 'A' value based on your recommended stock allocation result:
    * **100% Stocks** → Choose **1.5**
    * **80% Stocks** → Choose **2.2**
    * **60% Stocks** → Choose **3.0**
    * **40% Stocks** → Choose **4.5**
    * **20% Stocks** → Choose **8.0**

    **Part B: Lottery Test (Psychology of Loss)** Imagine you put **$1,000** on the table. You have a 50% chance to double it to **$2,000**.  
    At what loss amount (if the coin flip fails) would you **STOP** playing?
    * Stop at **$100 loss** → Choose **7.5**
    * Stop at **$200 loss** → Choose **3.8**
    * Stop at **$330 loss** → Choose **2.0**
    * Stop at **$500 loss** → Choose **1.0**
    """)

col1, col2 = st.columns(2)
v_opt = {"1.5 (100% Stocks)": 1.5, "2.2 (80% Stocks)": 2.2, "3.0 (60% Stocks)": 3.0, "4.5 (40% Stocks)": 4.5, "8.0 (20% Stocks)": 8.0}
a_vanguard = v_opt[col1.selectbox("Vanguard Result (A1):", list(v_opt.keys()), index=2)]
l_opt = {"7.5 (\$100 loss)": 7.5, "3.8 (\$200 loss)": 3.8, "2.0 (\$330 loss)": 2.0, "1.0 (\$500 loss)": 1.0}
a_lottery = l_opt[col2.selectbox("Lottery Test Result (A2):", list(l_opt.keys()), index=2)]

risk_aversion = (a_vanguard + a_lottery) / 2
st.success(f"🔍 Combined Risk Aversion (A): **{risk_aversion:.2f}**")

# --- STEP 2: INPUTS ---
st.header("⚙️ STEP 2: Input Parameters")
col_n1, col_n2 = st.columns(2)
net_worth = col_n1.number_input("Total Portfolio Value (USD)", value=300000.0)
rf_rate = col_n2.number_input("Risk-free Rate (e.g., 0.056)", value=0.056, format="%.4f")
period_input = st.selectbox("Historical Analysis Period", ["1y", "2y", "5y", "10y", "max"], index=2)
tickers_input = st.text_input("Tickers (space separated)", value="SPY GLD NVDA MSFT").upper().split()

# --- STEP 3: WEIGHTS ---
st.header("⚖️ STEP 3: Asset Weights")
user_weights = {}
if tickers_input:
    if app_mode == "Własny Portfel (Manual)":
        cols = st.columns(len(tickers_input))
        for i, t in enumerate(tickers_input):
            user_weights[t] = cols[i].number_input(f"Weight {t}", 0.0, 1.0, 1.0/len(tickers_input))
    else:
        for t in tickers_input:
            with st.expander(f"Constraints for {t}"):
                c1, c2 = st.columns(2)
                st.number_input(f"Min Weight {t}", 0.0, 1.0, 0.0, key=f"min_{t}")
                st.number_input(f"Max Weight {t}", 0.0, 1.0, 1.0, key=f"max_{t}")

# --- STEP 4: STRATEGY & STRESS TEST ---
st.header("🛡️ STEP 4: Strategy & Stress Test")
col_str1, col_str2 = st.columns(2)
with col_str1:
    method = st.radio("Optimization Method:", ["MAX SHARPE", "MIN VARIANCE", "EQUAL"])
    lev = st.checkbox("Allow Leverage (1.5x)?")
with col_str2:
    crash_sim = st.slider("Symuluj krach S&P 500 (%)", 0, 80, 20)

# --- OBLICZENIA I RAPORT ---
if st.button("🚀 GENERATE FULL REPORT"):
    with st.spinner('Pobieranie danych...'):
        data = yf.download(tickers_input + ["SPY"], period="5y", progress=False)['Close']
        if isinstance(data.columns, pd.MultiIndex): data.columns = data.columns.get_level_values(1)
        df = data.ffill().dropna()
        returns = df[tickers_input].pct_change().dropna()
        spy_ret = df["SPY"].pct_change().dropna()

        # Optymalizacja
        n = len(tickers_input)
        if app_mode == "Optymalizacja (Auto)":
            bounds = [(st.session_state[f"min_{t}"], st.session_state[f"max_{t}"]) for t in tickers_input]
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
            mu, cov = returns.mean() * 252, returns.cov() * 252
            if "SHARPE" in method:
                res = minimize(lambda w: -(np.sum(mu*w)-rf_rate)/np.sqrt(w.T@cov@w), n*[1./n], bounds=bounds, constraints=cons)
                weights = res.x
            else: weights = np.array(n*[1./n])
        else:
            weights = np.array([user_weights[t] for t in tickers_input])
            weights /= weights.sum()

        # Model Tobina
        p_vol = np.sqrt(weights.T @ (returns.cov()*252) @ weights)
        p_ret = np.sum(returns.mean()*252 * weights)
        opt_w = max(0, min((p_ret - rf_rate)/(risk_aversion * p_vol**2), 1.5 if lev else 1.0))

        # --- WYNIKI ---
        st.divider()
        st.subheader("1️⃣ Risky Portfolio Structure")
        st.table(pd.DataFrame({"Ticker": tickers_input, "Weight": [f"{v:.2%}" for v in weights]}).set_index("Ticker"))
        
        st.subheader("2️⃣ Total Allocation (Tobin Model)")
        c_r1, c_r2 = st.columns(2)
        c_r1.metric("RISKY ASSETS", f"{opt_w:.2%}", f"${net_worth * opt_w:,.2f}")
        c_r2.metric("CASH / SAFE", f"{1-opt_w:.2%}", f"${net_worth * (1-opt_w):,.2f}")

        st.subheader("📈 3️⃣ Backtesting (Equity Curve)")
        equity = (1 + (returns @ weights)).cumprod() * (net_worth * opt_w)
        st.plotly_chart(px.line(equity, title="Historical Growth (USD)"))

        st.subheader("🔥 4️⃣ Stress Test")
        beta = np.cov(returns @ weights, spy_ret.loc[returns.index])[0,1] / np.var(spy_ret)
        st.error(f"Przy spadku rynku o {crash_sim}%, szacowana strata to: **${(net_worth * opt_w * crash_sim/100 * beta):,.2f}**")