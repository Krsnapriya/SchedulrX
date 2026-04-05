import streamlit as st
import requests
import json
import os
import pandas as pd
from datetime import datetime

# Streamlit config
st.set_page_config(
    page_title="SchedulrX Engine",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Avoid crashing if Altair isn't ready
try:
    import altair as alt
    HAS_ALTAIR = True
except ImportError:
    HAS_ALTAIR = False

# --- Premium UI CSS ---
st.markdown("""
    <style>
        /* Base typography and colors */
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Outfit:wght@400;700&display=swap');
        
        html, body, [class*="css"]  {
            font-family: 'Inter', sans-serif;
            color: #e2e8f0;
        }
        
        h1, h2, h3 { 
            font-family: 'Outfit', sans-serif; 
            color: #f8fafc; 
            font-weight: 700 !important; 
            letter-spacing: -0.04em; 
        }
        
        .stApp {
            background: radial-gradient(circle at top left, #1e293b, #0f172a, #020617);
        }
        
        /* Glassmorphism Panels */
        .stColumn > div {
            background: rgba(30, 41, 59, 0.4);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255, 255, 255, 0.05);
            border-radius: 16px;
            padding: 1.5rem;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
        }
        
        /* Badges */
        .badge-live {
            background: linear-gradient(90deg, rgba(56, 189, 248, 0.2), rgba(14, 165, 233, 0.2));
            color: #38bdf8;
            padding: 0.3rem 0.8rem;
            border-radius: 9999px;
            font-size: 0.7rem;
            font-weight: 700;
            text-transform: uppercase;
            border: 1px solid rgba(56, 189, 248, 0.3);
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }
        
        .badge-live::before {
            content: "";
            width: 8px;
            height: 8px;
            background-color: #38bdf8;
            border-radius: 50%;
            display: inline-block;
            box-shadow: 0 0 8px #38bdf8;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(0.95); opacity: 1; }
            50% { transform: scale(1.1); opacity: 0.6; }
            100% { transform: scale(0.95); opacity: 1; }
        }
        
        .reasoning-panel {
            background: rgba(15, 23, 42, 0.6);
            border-left: 4px solid #38bdf8;
            border-radius: 8px;
            padding: 1.2rem;
            margin-top: 1rem;
            color: #94a3b8;
            font-size: 0.9rem;
            line-height: 1.6;
        }
        
        /* Button Styling */
        .stButton > button {
            border-radius: 8px !important;
            font-weight: 600 !important;
            transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            background: rgba(30, 41, 59, 0.5) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
            border-color: rgba(56, 189, 248, 0.4) !important;
        }
        
        /* Interactive Input Styling */
        .stTextInput input, .stSelectbox select, .stTextArea textarea {
            background-color: rgba(15, 23, 42, 0.8) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            color: #f1f5f9 !important;
            border-radius: 8px !important;
        }
        
        /* Custom Scrollbar */
        ::-webkit-scrollbar { width: 8px; }
        ::-webkit-scrollbar-track { background: #0f172a; }
        ::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
        ::-webkit-scrollbar-thumb:hover { background: #475569; }
    </style>
""", unsafe_allow_html=True)

BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8001")

# --- Initialize Session State ---
state_keys = [
    "session_id", "curr_obs", "prev_obs", "last_reward", 
    "is_done", "grade", "reward_history", "reasoning_history"
]
for k in state_keys:
    if k not in st.session_state:
        st.session_state[k] = [] if "history" in k else (0.0 if "reward" in k else None)
        if k == "is_done": st.session_state[k] = False

# --- REASONING EXTRACTOR ENGINE ---
def extract_reasoning(prev, action, curr, reward):
    """Deterministically extract agent intent and environment repercussions."""
    insights = []
    if not action: return insights
    
    act_type = action.get("action_type")
    
    # 1. Action intent
    if act_type == "read_profile":
        pid = action.get("participant_id")
        insights.append(f"🔍 Intent: Read hidden constraints for `{pid}`")
        if curr and prev:
            new_profs = set(curr.get("profiles_read", {}).keys()) - set(prev.get("profiles_read", {}).keys())
            if new_profs: 
                insights.append(f"🧠 Discovery: Learned penalty triggers for {list(new_profs)}")
            
    elif act_type == "schedule_meeting":
        mid = action.get("meeting_id")
        time = action.get("proposed_time")
        insights.append(f"📅 Intent: Schedule `{mid}` at {time}")
        
        # Conflict / Constraint verification
        if prev and curr:
            prev_sched = {m["meeting_id"] for m in prev.get("scheduled_meetings", [])}
            curr_sched = {m["meeting_id"] for m in curr.get("scheduled_meetings", [])}
            
            if mid in (curr_sched - prev_sched):
                insights.append("✅ Outcome: Successfully navigated timezone overlaps.")
            elif mid not in curr_sched:
                insights.append("❌ Outcome: Blocked by hard constraint or invalid slot.")
                
    # Reward mapping
    if reward is not None:
        if reward >= 0.4: insights.append("📈 Trade-off: Optimized schedule (High Reward)")
        elif reward > 0:  insights.append("👍 Trade-off: Acceptable placement (Positive Reward)")
        elif reward == 0: insights.append("➖ Trade-off: Neutral impact")
        elif reward < 0:  insights.append(f"⚠️ Trade-off: Suboptimal. Constraint violation incurred (Reward: {reward})")
        
    return insights

# --- Header Section ---
st.markdown('<div class="badge-live">● LIVE / v3 AGENT OBSERVATORY</div>', unsafe_allow_html=True)
st.title("SchedulrX")
st.markdown("<p style='color: #8b949e; font-size: 1.1rem; margin-top: -10px; margin-bottom: 2rem;'>Interactive demonstration of agent intelligence, learning, and reasoning under uncertainty.</p>", unsafe_allow_html=True)

# --- Layout ---
col_left, col_main, col_right = st.columns([1, 2.8, 1.2], gap="large")

# --- PANEL 1: CONTROL CENTER ---
with col_left:
    st.markdown("### 🎛️ Control Center")
    task = st.selectbox("Complexity Level", ["easy", "medium", "hard"])
    
    if st.button("Initialize Engine", use_container_width=True):
        with st.spinner("Initializing constraints..."):
            try:
                res = requests.post(f"{BASE_URL}/reset", params={"task_name": task})
                if res.status_code == 200:
                    data = res.json()
                    st.session_state.session_id = data.get("session_id")
                    st.session_state.curr_obs = data.get("observation")
                    st.session_state.prev_obs = None
                    st.session_state.last_reward = 0.0
                    st.session_state.reward_history = []
                    st.session_state.reasoning_history = []
                    st.session_state.is_done = False
                    st.session_state.grade = None
                else: st.error("Reset failed.")
            except Exception as e: st.error(f"API Error: {e}")
                
    st.markdown("<br>", unsafe_allow_html=True)
    if st.session_state.session_id:
        st.success(f"Session Active: `{st.session_state.session_id[:8]}...`")
    else:
        st.warning("Offline (Reset to start)")

    st.markdown("---")
    st.markdown("### ⚡ Action Builder")
    
    tab_build, tab_json = st.tabs(["Smart UI", "Raw JSON"])
    action_json_str = "{}"
    
    with tab_build:
        action_type = st.selectbox("Action", ["read_profile", "schedule_meeting", "cancel_meeting"])
        action_payload = {"action_type": action_type}
        
        if action_type == "read_profile":
            action_payload["participant_id"] = st.text_input("Participant ID (e.g. p1)", "p1")
        elif action_type == "schedule_meeting":
            action_payload["meeting_id"] = st.text_input("Meeting ID", "r1")
            action_payload["proposed_time"] = st.text_input("Time (ISO format)", "2026-04-06T10:00:00+00:00")
        
        action_json_str = json.dumps(action_payload, indent=2)
            
    with tab_json:
        action_json_str = st.text_area("JSON Payload", value=action_json_str, height=150)
        
    if st.button("▶️ Dispatch Action", type="primary", use_container_width=True, disabled=not st.session_state.session_id or st.session_state.is_done):
        try:
            parsed_action = json.loads(action_json_str)
            payload = {"session_id": st.session_state.session_id, "action": parsed_action}
            
            st.session_state.prev_obs = st.session_state.curr_obs
            res = requests.post(f"{BASE_URL}/step", json=payload)
            
            if res.status_code == 200:
                data = res.json()
                st.session_state.curr_obs = data.get("observation")
                reward = data.get("reward", 0.0)
                st.session_state.last_reward = reward
                st.session_state.reward_history.append(reward)
                st.session_state.is_done = data.get("done", False)
                st.session_state.grade = None 
                
                reasoning = extract_reasoning(st.session_state.prev_obs, parsed_action, st.session_state.curr_obs, reward)
                if reasoning:
                    st.session_state.reasoning_history.append(reasoning)
                st.rerun()
            else: st.error(res.text)
        except Exception as e: st.error(f"Execution Error: {e}")

# --- PANEL 2: MAIN VISUALIZATION ---
with col_main:
    st.markdown("### 📅 Temporal Matrix")
    
    obs = st.session_state.curr_obs
    if obs and HAS_ALTAIR:
        events = []
        
        # 1. Overlay Participant Availability Windows
        for p in obs.get("participants", []):
            pid = p["id"]
            for a in p.get("availability", []):
                events.append({
                    "Participant": pid, 
                    "Start": a["start"], 
                    "End": a["end"], 
                    "State": "Available", 
                    "Details": f"Base timezone block"
                })
                
        # 2. Overlay Scheduled Meetings (Highlight Conflicts if logic permits)
        reqs_map = {r["id"]: r["duration_minutes"] for r in obs.get("requests", [])}
        for m in obs.get("scheduled_meetings", []):
            dur = reqs_map.get(m["meeting_id"], 60)
            start_dt = pd.to_datetime(m["time"])
            end_dt = start_dt + pd.Timedelta(minutes=dur)
            for p_id in m.get("participants", []):
                events.append({
                    "Participant": p_id, 
                    "Start": start_dt.isoformat(), 
                    "End": end_dt.isoformat(), 
                    "State": "Scheduled", 
                    "Details": f"Meeting: {m['meeting_id']}"
                })
                
        if events:
            df = pd.DataFrame(events)
            df['Start'] = pd.to_datetime(df['Start'], utc=True)
            df['End'] = pd.to_datetime(df['End'], utc=True)
            
            # Color schema matches prompt expectations (Green/Available, Blue/Scheduled)
            color_scale = alt.Scale(
                domain=['Available', 'Scheduled', 'Conflict'], 
                range=['rgba(46, 160, 67, 0.15)', '#58a6ff', '#f85149']
            )
            
            chart = alt.Chart(df).mark_bar(opacity=0.9, cornerRadius=2).encode(
                x=alt.X('Start:T', title='Timeline (UTC)'),
                x2='End:T',
                y=alt.Y('Participant:N', title='', sort=None),
                color=alt.Color('State:N', scale=color_scale),
                tooltip=['Participant', 'State', 'Start', 'End', 'Details']
            ).properties(height=350).interactive()
            
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No timeline data generated yet.")
    else:
        st.info("Calendar requires active session.")

    st.markdown("### 📦 Observation Payload")
    if obs:
        t1, t2, t3, t4 = st.tabs(["Pending Requests", "Active Meetings", "Agent Knowledge", "Raw Global State"])
        with t1: st.json([r for r in obs.get("requests", []) if r["id"] not in [m["meeting_id"] for m in obs.get("scheduled_meetings", [])]])
        with t2: st.json(obs.get("scheduled_meetings", []))
        with t3: st.json(obs.get("profiles_read", {}))
        with t4: st.json(obs, expanded=False)

# --- PANEL 3: REASONING & LEARNING ---
with col_right:
    st.markdown("### 🧠 Agent Insights")
    
    if st.session_state.reasoning_history:
        latest = st.session_state.reasoning_history[-1]
        st.markdown('<div class="reasoning-panel">', unsafe_allow_html=True)
        st.markdown("**Trajectory Delta:**")
        for log in latest:
            st.markdown(f"- {log}")
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.info("Awaiting agent execution flow...")
        
    st.markdown("---")
    st.markdown("### 📈 Reward Trajectory")
    
    col_R1, col_R2 = st.columns(2)
    with col_R1: st.metric("Step Reward", f"{st.session_state.last_reward:.2f}")
    with col_R2: st.metric("Cumulative", f"{sum(st.session_state.reward_history):.2f}")
    
    if len(st.session_state.reward_history) > 0:
        st.line_chart(st.session_state.reward_history, height=180)
        
    if st.session_state.is_done:
        st.markdown('<div class="badge-live" style="background-color:rgba(210,153,34,0.15); color:#d29922; border-color:rgba(210,153,34,0.4); margin-top: 10px;">🏁 Episode Terminal State</div>', unsafe_allow_html=True)
        
    if st.button("🏆 Evaluate Performance", use_container_width=True, disabled=not st.session_state.session_id):
        try:
            res = requests.get(f"{BASE_URL}/grader", params={"session_id": st.session_state.session_id})
            if res.status_code == 200:
                st.session_state.grade = res.json().get("score", 0.0)
        except Exception as e: st.error(f"Grader Error: {e}")
            
    if st.session_state.grade is not None:
        s = st.session_state.grade
        st.success(f"Final Score: {s * 100:.1f}%")
        st.progress(min(1.0, max(0.0, s)))
        
    st.markdown("---")
    st.markdown("### 🤖 Baselines")
    
    bl_task = st.selectbox("Baseline Task", ["easy", "medium", "hard"], index=2, key="bl_task")
    
    if st.button("⚡ RL Agent (HeuristicRL-v1)", use_container_width=True):
        with st.spinner("Running RL agent..."):
            try:
                res = requests.post(f"{BASE_URL}/rl-baseline", params={"task_name": bl_task})
                if res.status_code == 200:
                    data = res.json()
                    st.metric("RL Score", f"{data.get('mean_score', 0) * 100:.1f}%")
                    st.metric("RL Reward", f"{data.get('mean_reward', 0):.3f}")
                    st.metric("Steps Used", int(data.get('mean_steps', 0)))
                    
                    trajectory = data.get('trajectory', [])
                    if trajectory:
                        st.markdown("**Trajectory:**")
                        traj_rewards = [t['cumulative_reward'] for t in trajectory]
                        st.line_chart(traj_rewards, height=120)
                        
                        with st.expander("Full Trajectory Log"):
                            for t in trajectory:
                                act = t.get('action', {})
                                st.caption(f"Step {t['step']}: `{act.get('action_type')}` → reward {t['reward']}")
                else:
                    st.error(res.text)
            except Exception as e:
                st.error(f"RL Error: {e}")
    
    if st.button("🧠 LLM Agent (GPT-4o-mini)", use_container_width=True):
        with st.spinner("Running LLM baseline..."):
            try:
                res = requests.post(f"{BASE_URL}/baseline", params={"task_name": bl_task})
                if res.status_code == 200:
                    st.json(res.json())
                else:
                    st.error(res.text)
            except Exception as e:
                st.error(f"LLM Error: {e}")
