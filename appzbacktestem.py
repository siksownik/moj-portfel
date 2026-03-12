import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.optimize import minimize

# --- KONFIGURACJA I STYLIZACJA (IDENTYCZNA Z TWOIM SCREENEM) ---
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

# --- SECTION 1: RISK AVERSION ---
st.header("🛡️ STEP 1: Determine Your Risk Aversion (A)")

with st.expander("📝 How to determine these values? (Instructions)", expanded=True):
    st.markdown(f"""
    **Part A: Vanguard Investor Questionnaire** Please complete the official test here: [Vanguard Investor Questionnaire](https://ownyourfuture.vanguard.com/home/investor-questionnaire/en/resultPage).  
    Select your 'A' value based on your recommended stock allocation result:
    * **100% Stocks** → Choose **1.5**
    * **80% Stocks** → Choose **2.2**
    * **60% Stocks** → Choose **3.0**
    * **40% Stocks** → Choose **4.5**
    * **20% Stocks** → Choose **8.0**

    **Part B: Lottery Test (Psychology of Loss)** Imagine you put 1,000 USD on the table. You have a fifty-fifty chance to double it to 2,000 USD.  
    At what loss amount (if the coin flip fails) would you **STOP** playing?
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
rf_rate = col_n2.number_input("Risk-free Rate (np. 0.056 dla 5.6%)", value=0.056, format="%.4f")

period_input = st.selectbox("Historical Analysis Period", ["1y", "2y", "5y", "10y", "max"], index=2)
tickers_input = st.text_input("Tickers (space separated)", value="SPY GLD NVDA MSFT").upper().split()

# --- SECTION 3: ASSET WEIGHTS ---
st.header("⚖️ STEP 3: Asset Weights")
user_weights = {}

if tickers_input:
    if app_mode == "Własny Portfel (Manual)":
        cols = st.columns(len(tickers_input))
        for i, t in enumerate(tickers_input):
            user_weights[t] = cols[i].number_input(f"Weight {t}", -1.0, 1.0, 1.0/len(tickers_input), key=f"manual_{t}")
    else:
        for t in tickers_input:
            with st.expander(f"Constraints for {t}"):
                c1, c2 = st.columns(2)
                # NAPRAWIONE: Używamy tylko key, bez ręcznego przypisywania do session_state przed widgetem
                st.number_input(f"Min Weight {t}", -1.0, 1.0, 0.0, key=f"min_{t}")
                st.number_input(f"Max Weight {t}", -1.0, 1.0, 1.0, key=f"max_{t}")

# --- SECTION 4: STRATEGY PREFERENCES ---
method_choice = "MAX SHARPE RATIO (Classic)"
allow_leverage = False
if app_mode == "Optymalizacja (Auto)":
    st.header("🛡️ STEP 4: Strategy Preferences")
    c_p1, c_p2 = st.columns(2)
    allow_short = c_p1.checkbox("Allow Asset Short-Selling?")
    allow_leverage = c_p2.checkbox("Allow Cash Leveraging?")
    method_choice = st.radio("Optimization Method:", ["MAX SHARPE RATIO (Classic)", "MINIMUM VARIANCE (Safe)", "EQUAL WEIGHTS"])

# --- STEP 5: STRESS TEST ---
st.header("📉 STEP 5: Stress Test Parameters")
market_crash = st.slider("Symuluj spadek rynku (S&P 500) o %", 0, 80, 20)

# --- CALCULATIONS ---
if st.button("🚀 GENERATE FULL REPORT"):
    with st.spinner('Pobieranie danych...'):
        all_data = {}
        to_fetch = list(set(tickers_input + ["SPY"]))
        for t in to_fetch:
            try:
                if t.endswith(".WA"):
                    symbol = t.replace(".WA", "").lower()
                    df = pd.read_csv(f"https://stooq.pl/q/d/l/?s={symbol}&i=d", index_col='Data', parse_dates=True)
                    all_data[t] = df['Zamkniecie']
                else:
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

        if app_mode == "Optymalizacja (Auto)":
            bounds = [(st.session_state[f"min_{t}"], st.session_state[f"max_{t}"]) for t in tickers_input]
            cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1.0})
            if "SHARPE" in method_choice:
                res = minimize(lambda w: -(np.sum(mu*w)-rf_rate)/np.sqrt(np.dot(w.T, np.dot(cov,w))), len(tickers_input)*[1./len(tickers_input)], bounds=bounds, constraints=cons)
                weights = res.x
            elif "VARIANCE" in method_choice:
                res = minimize(lambda w: np.sqrt(np.dot(w.T, np.dot(cov,w))), len(tickers_input)*[1./len(tickers_input)], bounds=bounds, constraints=cons)
                weights = res.x
            else: weights = np.array(len(tickers_input)*[1./len(tickers_input)])
        else:
            weights = np.array([user_weights[t] for t in tickers_input])
            weights /= weights.sum()

        p_ret = np.sum(mu * weights)
        p_vol = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        opt_risky_w = max(0, min((p_ret - rf_rate) / (risk_aversion * (p_vol**2)), 1.5 if allow_leverage else 1.0))

        # --- RESULTS ---
        st.divider()
        st.header("📊 RESULTS")
        st.subheader("1️⃣ Risky Portfolio Structure")
        st.table(pd.DataFrame({"Ticker": tickers_input, "Weight": [f"{v:.2%}" for v in weights]}).set_index("Ticker"))
        
        st.subheader("2️⃣ Total Allocation (Tobin Model)")
        c1, c2 = st.columns(2)
        c1.metric("RISKY ASSETS", f"{opt_risky_w:.2%}", f"${net_worth * opt_risky_w:,.2f}")
        c2.metric("CASH / SAFE", f"{1-opt_risky_w:.2%}", f"${net_worth * (1-opt_risky_w):,.2f}")

        # Backtest
        st.subheader("📈 3️⃣ Backtesting")
        port_vals = (1 + (returns @ weights)).cumprod() * (net_worth * opt_risky_w)
        st.plotly_chart(px.line(port_vals, title="Historical Value (Risky Portion)"))

        # Stress Test
        st.subheader("🔥 4️⃣ Stress Test")
        common = (returns @ weights).index.intersection(spy_ret.index)
        beta = np.cov((returns @ weights).loc[common], spy_ret.loc[common])[0,1] / np.var(spy_ret.loc[common])
        st.error(f"Przy spadku S&P 500 o {market_crash}%, Twój portfel (Beta: {beta:.2f}) spadnie o ok. {(market_crash/100*beta):.2%}, tracąc ${(net_worth * opt_risky_w * market_crash/100*beta):,.2f}.")