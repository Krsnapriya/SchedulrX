import streamlit as st
import httpx
import pandas as pd
from datetime import datetime, timedelta
import json
import time
from openai import OpenAI


# Premium Page Config
st.set_page_config(
    page_title="SchedulrX Engine HUD",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Glassmorphism & Diagnostic Aesthetic
st.markdown("""
<style>
    .stApp { background: #0f172a; color: #f8fafc; }
    [data-testid="stSidebar"] { background: rgba(30, 41, 59, 0.5); backdrop-filter: blur(10px); border-right: 1px solid rgba(56, 189, 248, 0.2); }
    .stButton>button { background: rgba(56, 189, 248, 0.1); border: 1px solid #38bdf8; color: #38bdf8; border-radius: 8px; font-weight: 600; width: 100%; }
    .stButton>button:hover { background: #38bdf8; color: #0f172a; box-shadow: 0 0 15px rgba(56, 189, 248, 0.4); }
    .stSelectbox div[data-baseweb="select"] { background: rgba(255, 255, 255, 0.05); border: 1px solid rgba(56, 189, 248, 0.2); }
    .stMetric { background: rgba(30, 41, 59, 0.5); padding: 15px; border-radius: 12px; border: 1px solid rgba(56, 189, 248, 0.1); }
    .console-box { font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; padding: 1rem; background: #020617; border-radius: 12px; border: 1px solid rgba(56, 189, 248, 0.2); height: 180px; overflow-y: auto; color: #38bdf8; }
    .profile-card { background: rgba(15, 23, 42, 0.8); border: 1px solid rgba(56, 189, 248, 0.2); padding: 1rem; border-radius: 12px; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_URL = "http://localhost:8000"

def log_msg(msg, type="info"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    color = "#38bdf8" if type=="info" else "#10b981" if type=="success" else "#ef4444"
    if 'logs' not in st.session_state: st.session_state.logs = []
    st.session_state.logs.insert(0, f'<span style="color:#94a3b8;">[{timestamp}]</span> <span style="color:{color};">{msg}</span>')

# App State Initialization
if 'session_id' not in st.session_state: st.session_state.session_id = None
if 'state' not in st.session_state: st.session_state.state = {}
if 'logs' not in st.session_state: st.session_state.logs = []

# Sidebar: Environment Lifecycle
with st.sidebar:
    st.title("🚀 SchedulrX HUD")
    st.caption("POMDP Diagnostics v2.1.0")
    st.divider()
    
    task_name = st.selectbox("Complexity Level", ["easy", "medium", "hard"])
    
    if st.button("Initialize Episode"):
        try:
            resp = httpx.post(f"{API_URL}/reset", params={"task_name": task_name})
            data = resp.json()
            st.session_state.session_id = data['session_id']
            # Re-fetch state immediately for full richness
            state_resp = httpx.get(f"{API_URL}/state", params={"session_id": st.session_state.session_id})
            st.session_state.state = state_resp.json()
            log_msg(f"Session Initialized: {st.session_state.session_id[:8]}...", "success")
        except Exception as e:
            log_msg(f"Initialization Error: {e}", "error")

    if st.session_state.session_id:
        st.success(f"Mode: {st.session_state.state.get('task','?').upper()}")
        st.code(f"ID: {st.session_state.session_id[:16]}")

    st.divider()
    if st.button("Manual Refresh"):
        if st.session_state.session_id:
            try:
                resp = httpx.get(f"{API_URL}/state", params={"session_id": st.session_state.session_id})
                st.session_state.state = resp.json()
                log_msg("Environment state synced", "info")
            except: log_msg("Failed to sync state", "error")

    st.divider()
    st.subheader("Auto-Pilot 🤖")
    api_key = st.text_input("OpenAI API Key (sk-...)", type="password")
    if st.button("Run Auto-Pilot"):
        if not api_key:
            st.error("API Key required.")
        elif not st.session_state.session_id:
            st.error("Initialize episode first.")
        else:
            st.session_state.autopilot_running = True
            st.session_state.api_key = api_key
            st.rerun()
    if st.button("Stop Auto-Pilot"):
        st.session_state.autopilot_running = False
        log_msg("Auto-Pilot stopped.", "info")
        st.rerun()

# Main Dashboard Layout
if not st.session_state.session_id:
    st.title("Welcome to SchedulrX")
    st.info("👈 Select a task complexity and initialize an episode to start diagnostics.")
    st.stop()

col_viz, col_play = st.columns([3, 2])

# Column 1: Visualization Hub
with col_viz:
    st.subheader("Environment Viewport")
    
    # 1. Calendar Grid
    hours = [9, 10, 11, 12, 13, 14, 15, 16, 17]
    days = ["Mon", "Tue", "Wed", "Thu", "Fri"]
    cal_df = pd.DataFrame("", index=[f"{h:02d}:00" for h in hours], columns=days)
    
    scheduled = st.session_state.state.get('scheduled_meetings', [])
    for m in scheduled:
        # SchedulrX uses local timestamps in availability now
        dt = datetime.fromisoformat(m['time'])
        day_name = dt.strftime("%a")
        hour_str = f"{dt.hour:02d}:00"
        if day_name in days and hour_str in cal_df.index:
            cal_df.at[hour_str, day_name] = f"✅ {m['meeting_id']}"
    
    st.table(cal_df)
    
    # 2. Metrics HUD
    m1, m2, m3 = st.columns(3)
    current_state = st.session_state.state
    m1.metric("Reward", f"{current_state.get('total_reward', 0.0):.4f}")
    m2.metric("Step", f"{current_state.get('step_count', 0)} / {current_state.get('max_steps', 80)}")
    m3.metric("Budget", f"{current_state.get('read_budget_remaining', 0)} Read(s)")

# Column 2: Action Playground
with col_play:
    st.subheader("Action Console")
    
    # Form for taking manual steps
    with st.form("manual_step"):
        action_type = st.selectbox("Action Type", ["read_profile", "schedule_meeting"])
        
        # Dynamic inputs based on action
        if action_type == "read_profile":
            p_ids = [p['id'] for p in current_state.get('participants', [])]
            target_pid = st.selectbox("Participant ID", p_ids if p_ids else ["p1", "p2", "p3", "p4", "p5"])
            target_mid = None
            target_time = None
        else:
            pending_ids = current_state.get('pending_meetings', [])
            target_mid = st.selectbox("Meeting ID", pending_ids if pending_ids else ["r1", "r2", "r3"])
            target_pid = None
            # Default to a mock time for ease of testing
            target_time = st.text_input("Proposed ISO Time", "2026-04-06T09:00:00+05:30")

        if st.form_submit_button("EXECUTE ACTION"):
            action_payload = {"action_type": action_type}
            if target_pid: action_payload["participant_id"] = target_pid
            if target_mid: action_payload["meeting_id"] = target_mid
            if target_time: action_payload["proposed_time"] = target_time
            
            try:
                resp = httpx.post(f"{API_URL}/step", json={"session_id": st.session_state.session_id, "action": action_payload})
                data = resp.json()
                st.session_state.state = data['observation'] # Observation is returned in body
                # Re-sync metrics from /state for full detail
                state_resp = httpx.get(f"{API_URL}/state", params={"session_id": st.session_state.session_id})
                st.session_state.state = state_resp.json()
                
                rew = data.get('reward', 0.0)
                log_msg(f"Step Executed | Reward: {rew:+.4f}", "success" if rew >= 0 else "error")
                st.rerun()
            except Exception as e:
                log_msg(f"Action Failed: {e}", "error")

    st.subheader("Reactive Status")
    st.markdown('<div class="console-box">' + ("<br>".join(st.session_state.logs)) + '</div>', unsafe_allow_html=True)
    if st.button("Clear Log"):
        st.session_state.logs = []
        st.rerun()

# Profile Discovery Section
st.divider()
st.subheader("Discovered Profiles")
profiles = current_state.get('profiles_read', {})
if not profiles:
    st.caption("No profiles discovered yet. Use 'read_profile' to uncover hidden constraints.")
else:
    cols = st.columns(len(profiles))
    for i, (pid, data) in enumerate(profiles.items()):
        with cols[i]:
            st.markdown(f"""
            <div class="profile-card">
                <b>{pid.upper()} Profile</b><br>
                <small>Max: {data.get('max_meetings_per_day')} sessions/day</small><br>
                <small>Avoid: {", ".join(data.get('avoid_days', []))}</small><br>
                <small>Prefers: {", ".join(data.get('preferred_times', []))}</small>
            </div>
            """, unsafe_allow_html=True)


# Auto-Pilot Execution Loop
if st.session_state.get('autopilot_running', False):
    current_state = st.session_state.state
    if current_state.get("done", False) or current_state.get("step_count", 0) >= current_state.get("max_steps", 80):
        st.session_state.autopilot_running = False
        log_msg("Auto-Pilot finished.", "info")
        st.rerun()
        
    client = OpenAI(api_key=st.session_state.api_key)
    prompt = f"""You are a God-Tier RL Agent solving a POMDP meeting scheduling environment.
Your capabilities:
1. OVERCOME PARTIAL OBSERVABILITY: Participants' availabilities are hidden. You MUST use 'read_profile' to discover them.
2. MANAGE TRUST BUDGET: You only have a limited amount of honest reads. Do not query if 'trust_scores' for a participant is 0.
3. ADVERSARIAL ROBUSTNESS: In hard mode, critical stakeholders (like P5) will reject schedules if you haven't read their profile first.
4. DEPENDENCY CHAINING: Meetings may have a 'depends_on' field. You MUST schedule the prerequisite meeting first.
5. COUNTER-PROPOSALS: If a chosen time violates a soft constraint, a participant may offer an alternative. Check 'counter_proposals' array and use 'accept_proposal' with the 'proposal_id'.
6. STOCHASTIC CANCELLATIONS: Meetings may unexpectedly be cancelled. Check 'cancelled_meetings' and immediately 'reschedule_meeting'.
7. ACTION FORMAT: Respond ONLY with valid JSON. Valid action_type: "read_profile", "schedule_meeting", "accept_proposal", "reschedule_meeting".

Current State:
{json.dumps(current_state, default=str)}

Return ONLY JSON: {{"action_type": string, "participant_id": optional_string, "meeting_id": optional_string, "proposed_time": optional_string, "proposal_id": optional_string}}"""
    
    try:
        llm_resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": "You are a deterministic AI agent outputting raw JSON."},
                      {"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=300
        )
        action_text = llm_resp.choices[0].message.content.strip()
        if action_text.startswith("```json"): action_text = action_text[7:-3]
        action_dict = json.loads(action_text)
        
        # Take step
        try:
            resp = httpx.post(f"{API_URL}/step", json={"session_id": st.session_state.session_id, "action": action_dict})
            data = resp.json()
            st.session_state.state = data['observation']
            state_resp = httpx.get(f"{API_URL}/state", params={"session_id": st.session_state.session_id})
            st.session_state.state = state_resp.json()
            
            rew = data.get('reward', 0.0)
            log_msg(f"Auto-Pilot Executed: {action_dict.get('action_type')} | Reward: {rew:+.4f}", "success" if rew >= 0 else "error")
        except Exception as step_e:
            log_msg(f"Auto-Pilot Step API Error: {step_e}", "error")
            st.session_state.autopilot_running = False
    except Exception as llm_e:
        log_msg(f"Auto-Pilot LLM Error: {llm_e}", "error")
        st.session_state.autopilot_running = False
        
    time.sleep(1)
    st.rerun()
