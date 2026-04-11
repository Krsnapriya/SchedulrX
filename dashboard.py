import streamlit as st
import httpx
import pandas as pd
from datetime import datetime
import json
import time

# Premium Page Config
st.set_page_config(
    page_title="SchedulrX Dashboard",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Glassmorphism & Dark HUD
st.markdown("""
<style>
    .stApp { background: #0f172a; color: #f8fafc; }
    [data-testid="stSidebar"] { background: rgba(30, 41, 59, 0.5); backdrop-filter: blur(10px); border-right: 1px solid rgba(56, 189, 248, 0.2); }
    .stButton>button { background: rgba(56, 189, 248, 0.1); border: 1px solid #38bdf8; color: #38bdf8; border-radius: 8px; font-weight: 600; width: 100%; }
    .stButton>button:hover { background: #38bdf8; color: #0f172a; box-shadow: 0 0 15px rgba(56, 189, 248, 0.4); }
    .stSelectbox div[data-baseweb="select"] { background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(56, 189, 248, 0.2); }
    .stMetric { background: rgba(30, 41, 59, 0.5); padding: 15px; border-radius: 12px; border: 1px solid rgba(56, 189, 248, 0.1); }
    .console-header { font-family: 'JetBrains Mono', monospace; font-size: 0.8rem; color: #94a3b8; margin-bottom: 1px; }
    .console-box { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; padding: 1rem; background: #020617; border-radius: 12px; border: 1px solid rgba(56, 189, 248, 0.2); height: 250px; overflow-y: auto; color: #38bdf8; }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = "http://localhost:8000"

def log_msg(msg, type="info"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    color = "#38bdf8" if type=="info" else "#10b981" if type=="success" else "#ef4444"
    if 'logs' not in st.session_state: st.session_state.logs = []
    st.session_state.logs.insert(0, f'<span style="color:#94a3b8;">[{timestamp}]</span> <span style="color:{color};">{msg}</span>')

# App State
if 'session_id' not in st.session_state: st.session_state.session_id = None
if 'state' not in st.session_state: st.session_state.state = {}
if 'logs' not in st.session_state: st.session_state.logs = []

# Sidebar: Controls
with st.sidebar:
    st.title("🚀 SchedulrX")
    st.caption("POMDP Scheduling Engine v2.1.0")
    st.divider()
    
    task_name = st.selectbox("Complexity Level", ["easy", "medium", "hard"])
    
    if st.button("Initialize Episode"):
        try:
            resp = httpx.post(f"{API_URL}/reset", params={"task_name": task_name})
            data = resp.json()
            st.session_state.session_id = data['session_id']
            st.session_state.state = data['observation']
            log_msg(f"Session Initialized: {st.session_state.session_id[:8]}...", "success")
        except Exception as e:
            log_msg(f"Initialization Error: {e}", "error")

    st.divider()
    if st.button("Refresh State"):
        if st.session_state.session_id:
            try:
                resp = httpx.get(f"{API_URL}/state", params={"session_id": st.session_state.session_id})
                st.session_state.state = resp.json()
                log_msg("State refreshed", "info")
            except: log_msg("Failed to fetch state", "error")

# Main Dashboard
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Environment Visualization")
    
    if st.session_state.session_id:
        st.write(f"Task: **{task_name.upper()}** | Session: `{st.session_state.session_id[:12]}`")
        
        # Calendar Grid (Mock representation for high-fidelity feel)
        hours = [8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
        days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
        
        cal_df = pd.DataFrame("", index=[f"{h:02d}:00" for h in hours], columns=days)
        
        # Overlay scheduled meetings
        scheduled = st.session_state.state.get('scheduled_meetings', [])
        for m in scheduled:
            dt = datetime.fromisoformat(m['time'].replace("Z", "+00:00"))
            day_name = dt.strftime("%a")
            hour_str = f"{dt.hour:02d}:00"
            if day_name in days and hour_str in cal_df.index:
                cal_df.at[hour_str, day_name] = f"✅ {m['meeting_id']}"
        
        st.table(cal_df)
    else:
        st.info("Please initialize a session to see the environment state.")

with col2:
    st.subheader("HUD Metrics")
    m1, m2 = st.columns(2)
    current_state = st.session_state.state
    
    m1.metric("Score", f"{current_state.get('total_reward', 0.0):.2f}")
    m2.metric("Steps", f"{current_state.get('step_count', 0)}/80")
    
    st.divider()
    st.subheader("Reactive Console")
    st.markdown('<div class="console-box">' + ("<br>".join(st.session_state.logs)) + '</div>', unsafe_allow_html=True)
    if st.button("Clear Log"):
        st.session_state.logs = []
        st.rerun()

# Technical Specs
with st.expander("Technical Observations"):
    st.json(st.session_state.state)
