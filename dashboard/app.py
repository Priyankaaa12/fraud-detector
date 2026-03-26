"""
app.py — Streamlit Dashboard
=============================
WHAT THIS FILE DOES:
  A real-time web dashboard that shows live fraud alerts, transaction statistics,
  and detailed AI explanations for each flagged transaction.

HOW TO RUN:
  streamlit run dashboard/app.py

WHAT YOU'LL SEE:
  - Page 1 (Overview): Stats, recent alerts, risk distribution chart
  - Page 2 (Alerts): Filterable table of all flagged transactions
  - Page 3 (Analyze): Submit a transaction manually and see the full AI analysis
  - Page 4 (Live Feed): Watch the transaction simulator in real time

The dashboard auto-refreshes every 5 seconds when the simulator is running.
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
import time

# ─── Page Configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── API Configuration ───────────────────────────────────────────────────────
API_BASE = "http://localhost:8000"

# ─── Custom Styling ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 16px 20px;
        border: 1px solid #e9ecef;
        margin-bottom: 10px;
    }
    .risk-critical { color: #dc3545; font-weight: bold; }
    .risk-high { color: #fd7e14; font-weight: bold; }
    .risk-medium { color: #ffc107; font-weight: bold; }
    .risk-low { color: #28a745; font-weight: bold; }
    .explanation-box {
        background: #f0f4ff;
        border-left: 4px solid #4361ee;
        padding: 16px;
        border-radius: 0 8px 8px 0;
        margin: 10px 0;
        line-height: 1.6;
    }
    .stAlert > div { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ─── Helper Functions ─────────────────────────────────────────────────────────

def check_api_status():
    """Check if the FastAPI backend is running."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=3)
        return r.status_code == 200, r.json()
    except:
        return False, {}


def get_alerts(limit=50, risk_level=None):
    """Fetch recent alerts from the API."""
    try:
        params = {"limit": limit}
        if risk_level and risk_level != "All":
            params["risk_level"] = risk_level
        r = requests.get(f"{API_BASE}/alerts", params=params, timeout=5)
        return r.json().get("alerts", [])
    except:
        return []


def get_stats():
    """Fetch dashboard statistics."""
    try:
        r = requests.get(f"{API_BASE}/stats", timeout=5)
        return r.json()
    except:
        return {}


def predict_transaction(transaction: dict, include_llm: bool = True):
    """Send a transaction for fraud analysis."""
    try:
        r = requests.post(
            f"{API_BASE}/predict",
            json=transaction,
            params={"include_llm": include_llm},
            timeout=30
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def risk_color(level: str) -> str:
    colors = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
    return colors.get(level, "⚪")


def action_badge(action: str) -> str:
    badges = {
        "BLOCK_IMMEDIATELY": "🚫 BLOCK",
        "REVIEW_URGENT": "⚠️ REVIEW",
        "MONITOR": "👁️ MONITOR",
        "CLEAR": "✅ CLEAR"
    }
    return badges.get(action, action)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.shields.io/badge/AI-Fraud%20Detection-4361ee?style=for-the-badge")
    st.markdown("---")
    
    # API Status
    is_running, health = check_api_status()
    if is_running:
        st.success("✅ API Connected")
        if health.get("model_loaded"):
            st.success("✅ ML Model Ready")
        else:
            st.warning("⚠️ Model not loaded")
        if health.get("llm_available"):
            st.success("✅ LLM Active")
        else:
            st.info("ℹ️ LLM not configured")
    else:
        st.error("❌ API Offline")
        st.code("uvicorn api.main:app --reload", language="bash")
    
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📊 Overview", "🚨 Alerts", "🔍 Analyze Transaction", "📡 Live Feed"]
    )
    
    st.markdown("---")
    auto_refresh = st.toggle("Auto-refresh (5s)", value=False)
    if auto_refresh:
        st.info("🔄 Refreshing every 5 seconds")
        time.sleep(5)
        st.rerun()


# ─── Page 1: Overview ─────────────────────────────────────────────────────────

if page == "📊 Overview":
    st.title("🔍 Fraud Detection Dashboard")
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")
    
    stats = get_stats()
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Alerts (24h)",
            stats.get("total_alerts", 0),
            help="Total flagged transactions in the last 24 hours"
        )
    
    with col2:
        critical = stats.get("by_risk_level", {}).get("CRITICAL", 0)
        st.metric("Critical Alerts", critical, delta=None)
    
    with col3:
        amount = stats.get("total_flagged_amount", 0)
        st.metric("Flagged Amount", f"${amount:,.0f}")
    
    with col4:
        avg_prob = stats.get("avg_fraud_probability", 0)
        st.metric("Avg Fraud Probability", f"{avg_prob:.1%}")
    
    st.markdown("---")
    
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # Risk level distribution chart
        risk_data = stats.get("by_risk_level", {})
        if risk_data:
            df_risk = pd.DataFrame({
                "Risk Level": list(risk_data.keys()),
                "Count": list(risk_data.values())
            })
            colors = {
                "CRITICAL": "#dc3545",
                "HIGH": "#fd7e14",
                "MEDIUM": "#ffc107",
                "LOW": "#28a745"
            }
            fig = px.pie(
                df_risk,
                names="Risk Level",
                values="Count",
                title="Alerts by Risk Level",
                color="Risk Level",
                color_discrete_map=colors,
                hole=0.4
            )
            fig.update_layout(height=300, margin=dict(t=40, b=20))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No alert data yet. Start the simulator to generate transactions.")
    
    with col_right:
        # Recent alerts list
        st.subheader("Recent Alerts")
        alerts = get_alerts(limit=10)
        
        if alerts:
            for alert in alerts[:8]:
                with st.container():
                    col_a, col_b, col_c = st.columns([1, 1, 2])
                    with col_a:
                        st.write(f"{risk_color(alert.get('risk_level', ''))} {alert.get('risk_level', '')}")
                    with col_b:
                        st.write(f"Score: **{alert.get('risk_score', 0)}**")
                    with col_c:
                        st.write(f"{action_badge(alert.get('action', ''))}")
        else:
            st.info("No alerts yet. Ensure the API is running and the simulator is active.")


# ─── Page 2: Alerts ───────────────────────────────────────────────────────────

elif page == "🚨 Alerts":
    st.title("🚨 Fraud Alerts")
    
    # Filters
    col1, col2 = st.columns([1, 3])
    with col1:
        risk_filter = st.selectbox("Filter by Risk", ["All", "CRITICAL", "HIGH", "MEDIUM", "LOW"])
    with col2:
        limit = st.slider("Show last N alerts", 10, 200, 50)
    
    alerts = get_alerts(limit=limit, risk_level=risk_filter if risk_filter != "All" else None)
    
    if alerts:
        # Convert to DataFrame for display
        df = pd.DataFrame(alerts)
        
        # Add emoji column
        df['risk_icon'] = df['risk_level'].map(
            {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"}
        ).fillna("⚪")
        
        # Display
        display_cols = ['risk_icon', 'risk_level', 'risk_score', 'action', 
                       'top_factor', 'timestamp']
        available_cols = [c for c in display_cols if c in df.columns]
        
        st.dataframe(
            df[available_cols].rename(columns={
                'risk_icon': '', 
                'risk_level': 'Risk Level',
                'risk_score': 'Score',
                'action': 'Action',
                'top_factor': 'Top Risk Factor',
                'timestamp': 'Time'
            }),
            use_container_width=True,
            height=500
        )
        
        # Expandable detail for first alert
        if st.checkbox("Show explanation for most recent alert"):
            latest = alerts[0]
            st.markdown(f"""
            <div class="explanation-box">
            {latest.get('explanation', 'No explanation available.')}
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.info("No alerts found. Try running the simulator: `python src/simulator.py`")


# ─── Page 3: Analyze Transaction ─────────────────────────────────────────────

elif page == "🔍 Analyze Transaction":
    st.title("🔍 Analyze a Transaction")
    st.caption("Enter transaction details to get an instant fraud analysis with AI explanation.")
    
    with st.form("transaction_form"):
        st.subheader("Transaction Details")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            txn_id = st.text_input("Transaction ID", value="TXN-TEST-001")
            amount = st.number_input("Amount ($)", min_value=0.0, value=4200.0, step=50.0)
            hour = st.slider("Hour of Day", 0, 23, 3)
        
        with col2:
            day = st.selectbox("Day of Week", 
                              ["Monday", "Tuesday", "Wednesday", "Thursday", 
                               "Friday", "Saturday", "Sunday"],
                              index=5)
            d1 = st.number_input("Days Since Last Transaction", min_value=0.0, value=0.0)
            c1 = st.number_input("Transaction Count (C1)", min_value=0, value=50)
        
        with col3:
            card_type = st.selectbox("Card Type", ["Visa", "Mastercard", "Discover", "Amex"])
            email_domain = st.selectbox("Email Domain", 
                                       ["gmail.com", "yahoo.com", "hotmail.com", 
                                        "Other/Unknown"])
            include_llm = st.checkbox("Include AI Explanation", value=True)
        
        submitted = st.form_submit_button("🔍 Analyze Transaction", type="primary")
    
    if submitted:
        # Map inputs to encoded values
        card_map = {"Visa": 2, "Mastercard": 1, "Discover": 3, "Amex": 4}
        email_map = {"gmail.com": 0, "yahoo.com": 1, "hotmail.com": 2, "Other/Unknown": 8}
        day_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                  "Friday": 4, "Saturday": 5, "Sunday": 6}
        
        transaction = {
            "transaction_id": txn_id,
            "TransactionAmt": amount,
            "hour": hour,
            "day_of_week": day_map[day],
            "card4": card_map[card_type],
            "P_emaildomain": email_map[email_domain],
            "D1": d1,
            "C1": c1,
            "is_large_amount": 1 if amount > 1000 else 0,
        }
        
        with st.spinner("🔄 Analyzing transaction..."):
            result = predict_transaction(transaction, include_llm=include_llm)
        
        if "error" not in result:
            # Risk level banner
            level = result.get("risk_level", "UNKNOWN")
            score = result.get("risk_score", 0)
            prob = result.get("fraud_probability", 0)
            
            color_map = {
                "CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢"
            }
            
            # Main result
            st.markdown("---")
            col_r1, col_r2, col_r3 = st.columns(3)
            with col_r1:
                st.metric("Fraud Probability", f"{prob:.1%}")
            with col_r2:
                st.metric("Risk Score", f"{score}/100")
            with col_r3:
                st.metric("Risk Level", f"{color_map.get(level, '')} {level}")
            
            # Recommended action
            action = result.get("action", "")
            action_colors = {
                "BLOCK_IMMEDIATELY": "error",
                "REVIEW_URGENT": "warning",
                "MONITOR": "info",
                "CLEAR": "success"
            }
            alert_type = action_colors.get(action, "info")
            getattr(st, alert_type)(f"**Recommended Action:** {action_badge(action)}\n\n{result.get('action_reason', '')}")
            
            # AI Explanation
            explanation = result.get("explanation", "")
            if explanation:
                st.subheader("🧠 AI Explanation")
                st.markdown(f"""
                <div class="explanation-box">
                {explanation}
                </div>
                """, unsafe_allow_html=True)
            
            # SHAP factors chart
            top_factors = result.get("top_factors", [])
            if top_factors:
                st.subheader("📊 Risk Factor Analysis")
                
                df_factors = pd.DataFrame(top_factors)
                df_factors['color'] = df_factors['shap_value'].apply(
                    lambda x: '#dc3545' if x > 0 else '#28a745'
                )
                df_factors['abs_shap'] = df_factors['shap_value'].abs()
                df_factors = df_factors.sort_values('abs_shap')
                
                fig = go.Figure(go.Bar(
                    x=df_factors['shap_value'],
                    y=df_factors['feature'],
                    orientation='h',
                    marker_color=df_factors['color'],
                    text=[f"{v:+.3f}" for v in df_factors['shap_value']],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title="SHAP Feature Contributions (Red = increases fraud risk)",
                    xaxis_title="SHAP Value (Impact on Fraud Probability)",
                    yaxis_title="Feature",
                    height=350,
                    margin=dict(l=20, r=20, t=40, b=20),
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(zeroline=True, zerolinecolor='gray', zerolinewidth=1)
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
            st.info("Make sure the API server is running: `uvicorn api.main:app --reload`")


# ─── Page 4: Live Feed ────────────────────────────────────────────────────────

elif page == "📡 Live Feed":
    st.title("📡 Live Transaction Feed")
    st.caption("Real-time stream of transactions. Run `python src/simulator.py` to populate this feed.")
    
    col_info1, col_info2 = st.columns(2)
    with col_info1:
        st.code("# Terminal 1: Start API\nuvicorn api.main:app --reload", language="bash")
    with col_info2:
        st.code("# Terminal 2: Start Simulator\npython src/simulator.py", language="bash")
    
    st.markdown("---")
    
    alerts = get_alerts(limit=30)
    
    if alerts:
        for alert in alerts:
            risk = alert.get("risk_level", "LOW")
            score = alert.get("risk_score", 0)
            action = alert.get("action", "")
            top_factor = alert.get("top_factor", "")
            ts = alert.get("timestamp", "")
            
            icon = risk_color(risk)
            
            with st.container():
                c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
                with c1:
                    st.write(f"{icon} **{risk}**")
                with c2:
                    st.write(f"Score: {score}/100")
                with c3:
                    st.write(action_badge(action))
                with c4:
                    st.write(f"_{top_factor}_")
            
            st.divider()
    else:
        st.info("No transactions yet. Start the simulator in a separate terminal window.")
        st.markdown("""
        **Quick Start Guide:**
        1. Open Terminal 1 → `uvicorn api.main:app --reload`  
        2. Open Terminal 2 → `python src/simulator.py`
        3. Come back here and enable Auto-refresh in the sidebar
        """)
