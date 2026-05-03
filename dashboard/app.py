import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import math
import time

st.set_page_config(page_title="Fraud Detection Dashboard", page_icon="🔍", layout="wide")

API_BASE = "http://localhost:8000"

st.markdown("""<style>
.explanation-box{background:#f0f4ff;border-left:4px solid #4361ee;padding:16px;border-radius:0 8px 8px 0;margin:10px 0;line-height:1.6}
</style>""", unsafe_allow_html=True)

def check_api_status():
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200, r.json()
    except:
        return False, {}

def get_alerts(limit=50, risk_level=None):
    try:
        params = {"limit": limit}
        if risk_level and risk_level != "All":
            params["risk_level"] = risk_level
        r = requests.get(f"{API_BASE}/alerts", params=params, timeout=5)
        return r.json().get("alerts", [])
    except:
        return []

def get_stats():
    try:
        r = requests.get(f"{API_BASE}/stats", timeout=5)
        return r.json()
    except:
        return {}

def predict_transaction(transaction, include_llm=True):
    try:
        r = requests.post(f"{API_BASE}/predict", json=transaction, params={"include_llm": include_llm}, timeout=30)
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def risk_color(level):
    return {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}.get(level, "⚪")

def action_badge(action):
    return {"BLOCK_IMMEDIATELY": "🚫 BLOCK", "REVIEW_URGENT": "⚠️ REVIEW", "MONITOR": "👁️ MONITOR", "CLEAR": "✅ CLEAR"}.get(action, action)

def build_full_transaction(txn_id, amount, hour, day, d1, c1, card_type, email_domain):
    card_map  = {"Visa": 2, "Mastercard": 1, "Discover": 3, "Amex": 4}
    email_map = {"gmail.com": 0, "yahoo.com": 1, "hotmail.com": 2, "Other/Unknown": 8}
    day_map   = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,"Friday":4,"Saturday":5,"Sunday":6}

    is_night     = hour < 6 or hour > 22
    is_weekend   = day_map[day] in [5, 6]
    is_large     = amount > 1000
    is_new       = d1 < 2
    is_high_freq = c1 > 20
    is_regular   = d1 > 30 and c1 < 5

    risk_count = sum([is_night, is_weekend, is_large, is_new, is_high_freq])
    is_risky   = risk_count >= 2

    log_amt = math.log1p(amount)

    if is_risky:
        c2, c6, c13, c14 = c1*3, c1*2, c1*4, c1*2
    else:
        c2  = max(1, c1-1)
        c6  = max(1, c1//3)
        c13 = max(1, c1//2)
        c14 = max(1, c1//4)

    d10 = max(0.5, d1*0.8)
    d15 = max(0.5, d1*1.2)

    if is_regular:
        v12,v13,v29,v30,v33,v34 = 1.0,1.0,0.0,0.0,1.0,1.0
    elif is_risky:
        v12,v13,v29,v30,v33,v34 = 0.0,0.0,1.0,1.0,0.0,0.0
    else:
        v12,v13,v29,v30,v33,v34 = 1.0,1.0,0.0,0.0,1.0,1.0

    m4 = m5 = m6 = 0 if is_risky else 2
    card1 = 500.0  if is_risky else 7200.0
    card2 = 100.0  if is_risky else 452.0
    card6 = 0      if is_risky else 1
    addr1 = 100.0  if is_risky else 299.0
    addr2 = 10.0   if is_risky else 87.0
    dist1 = 500.0  if is_risky else 5.0

    return {
        "transaction_id": txn_id,
        "TransactionAmt": amount,
        "ProductCD": 4 if is_risky else 0,
        "hour": hour,
        "day_of_week": day_map[day],
        "log_amount": log_amt,
        "is_large_amount": 1 if amount > 1000 else 0,
        "card4": card_map[card_type],
        "card1": card1, "card2": card2, "card6": card6,
        "P_emaildomain": email_map[email_domain],
        "R_emaildomain": email_map[email_domain],
        "addr1": addr1, "addr2": addr2, "dist1": dist1,
        "D1": d1, "D10": d10, "D15": d15,
        "C1": c1, "C2": c2, "C6": c6, "C13": c13, "C14": c14,
        "M4": m4, "M5": m5, "M6": m6,
        "V12": v12, "V13": v13, "V29": v29, "V30": v30, "V33": v33, "V34": v34,
    }

# Sidebar
with st.sidebar:
    st.markdown("**AI FRAUD DETECTION**")
    st.markdown("---")
    is_running, health = check_api_status()
    if is_running:
        st.success("✅ API Connected")
        if health.get("model_loaded"): st.success("✅ ML Model Ready")
        else: st.warning("⚠️ Model not loaded")
        if health.get("llm_available"): st.success("✅ LLM Active")
        else: st.info("ℹ️ LLM not configured")
    else:
        st.error("❌ API Offline")
        st.code("uvicorn api.main:app --reload")
    st.markdown("---")
    page = st.radio("Navigation", ["📊 Overview","🚨 Alerts","🔍 Analyze Transaction","📡 Live Feed"])
    st.markdown("---")
    auto_refresh = st.toggle("Auto-refresh (5s)", value=False)
    if auto_refresh:
        time.sleep(5)
        st.rerun()

# Overview
if page == "📊 Overview":
    st.title("🔍 Fraud Detection Dashboard")
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    stats = get_stats()
    c1,c2,c3,c4 = st.columns(4)
    with c1: st.metric("Total Alerts (24h)", stats.get("total_alerts",0))
    with c2: st.metric("Critical Alerts", stats.get("by_risk_level",{}).get("CRITICAL",0))
    with c3: st.metric("Flagged Amount", f"${stats.get('total_flagged_amount',0):,.0f}")
    with c4: st.metric("Avg Fraud Probability", f"{stats.get('avg_fraud_probability',0):.1%}")
    st.markdown("---")
    cl, cr = st.columns(2)
    with cl:
        risk_data = stats.get("by_risk_level",{})
        if risk_data and any((v or 0) > 0 for v in risk_data.values()):
            df_r = pd.DataFrame({"Risk Level":list(risk_data.keys()),"Count":list(risk_data.values())})
            colors = {"CRITICAL":"#dc3545","HIGH":"#fd7e14","MEDIUM":"#ffc107","LOW":"#28a745"}
            fig = px.pie(df_r,names="Risk Level",values="Count",title="Alerts by Risk Level",color="Risk Level",color_discrete_map=colors,hole=0.4)
            fig.update_layout(height=300,margin=dict(t=40,b=20))
            st.plotly_chart(fig,use_container_width=True)
        else:
            st.info("No data yet. Start the simulator.")
    with cr:
        st.subheader("Recent Alerts")
        alerts = get_alerts(limit=10)
        if alerts:
            for a in alerts[:8]:
                ca,cb,cc = st.columns([1,1,2])
                with ca: st.write(f"{risk_color(a.get('risk_level',''))} {a.get('risk_level','')}")
                with cb: st.write(f"Score: **{a.get('risk_score',0)}**")
                with cc: st.write(action_badge(a.get('action','')))
        else:
            st.info("No alerts yet.")

# Alerts
elif page == "🚨 Alerts":
    st.title("🚨 Fraud Alerts")
    c1,c2 = st.columns([1,3])
    with c1: risk_filter = st.selectbox("Filter by Risk",["All","CRITICAL","HIGH","MEDIUM","LOW"])
    with c2: limit = st.slider("Show last N alerts",10,200,50)
    alerts = get_alerts(limit=limit,risk_level=risk_filter if risk_filter!="All" else None)
    if alerts:
        df = pd.DataFrame(alerts)
        df['risk_icon'] = df['risk_level'].map({"CRITICAL":"🔴","HIGH":"🟠","MEDIUM":"🟡","LOW":"🟢"}).fillna("⚪")
        disp = [c for c in ['risk_icon','risk_level','risk_score','action','top_factor','timestamp'] if c in df.columns]
        st.dataframe(df[disp].rename(columns={'risk_icon':'','risk_level':'Risk Level','risk_score':'Score','action':'Action','top_factor':'Top Risk Factor','timestamp':'Time'}),use_container_width=True,height=500)
    else:
        st.info("No alerts found.")

# Analyze Transaction
elif page == "🔍 Analyze Transaction":
    st.title("🔍 Analyze a Transaction")
    st.caption("Enter transaction details to get an instant fraud analysis with AI explanation.")

    with st.expander("📋 Demo Cheat Sheet — Click to see guaranteed results"):
        st.markdown("""
| Risk Level | Amount | Hour | Day | Days Since Last | Count |
|---|---|---|---|---|---|
| 🟢 **LOW** | 10 | 12 | Tuesday | 90 | 1 |
| 🟡 **MEDIUM** | 25 | 10 | Wednesday | 30 | 2 |
| 🟠 **HIGH** | 50 | 14 | Thursday | 14 | 3 |
| 🔴 **CRITICAL** | 500 | 20 | Friday | 2 | 15 |
        """)

    with st.form("transaction_form"):
        st.subheader("Transaction Details")
        c1,c2,c3 = st.columns(3)
        with c1:
            txn_id = st.text_input("Transaction ID", value="TXN-TEST-001")
            amount = st.number_input("Amount ($)", min_value=0.0, value=10.0, step=5.0)
            hour   = st.slider("Hour of Day", 0, 23, 12)
        with c2:
            day = st.selectbox("Day of Week",["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"],index=1)
            d1  = st.number_input("Days Since Last Transaction", min_value=0.0, value=90.0)
            c1_val = st.number_input("Transaction Count (C1)", min_value=0, value=1)
        with c3:
            card_type    = st.selectbox("Card Type",["Visa","Mastercard","Discover","Amex"])
            email_domain = st.selectbox("Email Domain",["gmail.com","yahoo.com","hotmail.com","Other/Unknown"])
            include_llm  = st.checkbox("Include AI Explanation", value=True)
        submitted = st.form_submit_button("🔍 Analyze Transaction", type="primary")

    if submitted:
        transaction = build_full_transaction(txn_id, amount, hour, day, d1, c1_val, card_type, email_domain)
        with st.spinner("🔄 Analyzing transaction..."):
            result = predict_transaction(transaction, include_llm=include_llm)

        if "error" not in result:
            level = result.get("risk_level","UNKNOWN")
            score = result.get("risk_score",0)
            prob  = result.get("fraud_probability",0)
            st.markdown("---")
            cr1,cr2,cr3 = st.columns(3)
            with cr1: st.metric("Fraud Probability", f"{prob:.1%}")
            with cr2: st.metric("Risk Score", f"{score}/100")
            with cr3:
                icons = {"CRITICAL":"🔴","HIGH":"🟠","MEDIUM":"🟡","LOW":"🟢"}
                st.metric("Risk Level", f"{icons.get(level,'')} {level}")

            action = result.get("action","")
            ac = {"BLOCK_IMMEDIATELY":"error","REVIEW_URGENT":"warning","MONITOR":"info","CLEAR":"success"}
            getattr(st, ac.get(action,"info"))(f"**Recommended Action:** {action_badge(action)}\n\n{result.get('action_reason','')}")

            explanation = result.get("explanation","")
            if explanation:
                st.subheader("🧠 AI Explanation")
                st.markdown(f'<div class="explanation-box" style="color:#1a1a2e;">{explanation}</div>', unsafe_allow_html=True)


            top_factors = result.get("top_factors",[])
            if top_factors:
                st.subheader("📊 Risk Factor Analysis")
                dff = pd.DataFrame(top_factors)
                dff['color'] = dff['shap_value'].apply(lambda x: '#dc3545' if x>0 else '#28a745')
                dff['abs_shap'] = dff['shap_value'].abs()
                dff = dff.sort_values('abs_shap')
                fig = go.Figure(go.Bar(x=dff['shap_value'],y=dff['feature'],orientation='h',
                    marker_color=dff['color'],text=[f"{v:+.3f}" for v in dff['shap_value']],textposition='outside'))
                fig.update_layout(title="SHAP Feature Contributions (Red = increases fraud risk)",
                    xaxis_title="SHAP Value",yaxis_title="Feature",height=350,
                    margin=dict(l=20,r=20,t=40,b=20),plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(zeroline=True,zerolinecolor='gray',zerolinewidth=1))
                st.plotly_chart(fig,use_container_width=True)
        else:
            st.error(f"Analysis failed: {result.get('error')}")

# Live Feed
elif page == "📡 Live Feed":
    st.title("📡 Live Transaction Feed")
    c1,c2 = st.columns(2)
    with c1: st.code("# Terminal 1\nuvicorn api.main:app --reload")
    with c2: st.code("# Terminal 2\npython src/simulator.py")
    st.markdown("---")
    alerts = get_alerts(limit=30)
    if alerts:
        for a in alerts:
            ca,cb,cc,cd = st.columns([1,1,1,3])
            with ca: st.write(f"{risk_color(a.get('risk_level','LOW'))} **{a.get('risk_level','')}**")
            with cb: st.write(f"Score: {a.get('risk_score',0)}/100")
            with cc: st.write(action_badge(a.get('action','')))
            with cd: st.write(f"_{a.get('top_factor','')}_")
            st.divider()
    else:
        st.info("No transactions yet. Start the simulator.")