import streamlit as st
import requests
import json
import os
import pandas as pd

st.set_page_config(
    page_title="SchedulrX",
    page_icon="📅",
    layout="wide",
    initial_sidebar_state="collapsed",
)

try:
    import altair as alt
    HAS_ALTAIR = True
except ImportError:
    HAS_ALTAIR = False

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; color: #c9d1d9; }
.stApp { background-color: #0d1117; }
h1, h2, h3 { color: #e6edf3; font-weight: 600 !important; }
header { visibility: hidden; }
footer { visibility: hidden; }
.mono { font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem; }
.tag {
    display: inline-block; padding: 2px 8px; border-radius: 4px;
    font-size: 0.7rem; font-weight: 600; font-family: 'IBM Plex Mono', monospace;
    background: rgba(88, 166, 255, 0.1); color: #58a6ff;
    border: 1px solid rgba(88, 166, 255, 0.3); margin: 2px;
}
.stTextInput input, .stSelectbox select, .stTextArea textarea {
    background-color: #161b22 !important;
    border: 1px solid #30363d !important;
    color: #c9d1d9 !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
</style>
""", unsafe_allow_html=True)

BASE_URL = os.getenv("API_URL", "http://127.0.0.1:8001")

for k, default in [
    ("session_id", None), ("curr_obs", None), ("prev_obs", None),
    ("last_reward", 0.0), ("is_done", False), ("grade", None),
    ("reward_history", []), ("step_log", []),
]:
    if k not in st.session_state:
        st.session_state[k] = default


def _step_log(action_dict, reward, info, done):
    entry = {
        "step":   len(st.session_state.step_log) + 1,
        "action": action_dict.get("action_type", "?"),
        "detail": action_dict.get("participant_id") or action_dict.get("meeting_id") or "",
        "reward": reward,
        "result": info.get("scheduled") or info.get("discovered") or info.get("error") or ("done" if done else "ok"),
    }
    st.session_state.step_log.append(entry)


# ── Header ─────────────────────────────────────────────────────────────────
st.title("SchedulrX")
st.markdown(
    "<p style='color:#8b949e; font-size:1rem; margin-top:-10px;'>"
    "POMDP meeting-scheduling environment for RL research · "
    "<span class='tag'>OpenEnv</span>"
    "<span class='tag'>POMDP</span>"
    "<span class='tag'>Multi-timezone</span>"
    "<span class='tag'>Hidden constraints</span>"
    "</p>",
    unsafe_allow_html=True,
)
st.divider()

col_left, col_main, col_right = st.columns([1, 2.8, 1.2], gap="large")

# ── Left: controls ──────────────────────────────────────────────────────────
with col_left:
    st.markdown("### Episode Setup")
    task = st.selectbox("Difficulty", ["easy", "medium", "hard"],
                        help="easy=1 meeting, medium=3+trap, hard=3+multiple traps")

    if st.button("Reset", use_container_width=True, type="primary"):
        try:
            res = requests.post(f"{BASE_URL}/reset", params={"task_name": task})
            if res.status_code == 200:
                data = res.json()
                st.session_state.session_id    = data["session_id"]
                st.session_state.curr_obs      = data["observation"]
                st.session_state.prev_obs      = None
                st.session_state.last_reward   = 0.0
                st.session_state.reward_history = []
                st.session_state.step_log      = []
                st.session_state.is_done       = False
                st.session_state.grade         = None
                st.rerun()
            else:
                st.error(f"Reset failed: {res.status_code}")
        except Exception as e:
            st.error(str(e))

    if st.session_state.session_id:
        st.caption(f"Session: `{st.session_state.session_id[:12]}…`")
        st.caption(f"Steps taken: {len(st.session_state.step_log)}")
    else:
        st.warning("No active session")

    st.divider()
    st.markdown("### Action")

    tab_ui, tab_raw = st.tabs(["Builder", "JSON"])
    action_str = "{}"

    with tab_ui:
        atype = st.selectbox("Type", ["read_profile", "schedule_meeting"])
        payload = {"action_type": atype}
        if atype == "read_profile":
            payload["participant_id"] = st.selectbox("Participant", ["p1", "p2", "p3", "p4", "p5"])
        else:
            payload["meeting_id"]    = st.text_input("Meeting ID", "r1")
            payload["proposed_time"] = st.text_input("Proposed time (ISO 8601)", "")
        action_str = json.dumps(payload, indent=2)

    with tab_raw:
        action_str = st.text_area("JSON", value=action_str, height=120)

    disabled = not st.session_state.session_id or st.session_state.is_done
    if st.button("▶  Execute", type="primary", use_container_width=True, disabled=disabled):
        try:
            parsed = json.loads(action_str)
            res = requests.post(
                f"{BASE_URL}/step",
                json={"session_id": st.session_state.session_id, "action": parsed},
            )
            if res.status_code == 200:
                data = res.json()
                st.session_state.prev_obs      = st.session_state.curr_obs
                st.session_state.curr_obs      = data["observation"]
                reward                          = data.get("reward", 0.0)
                st.session_state.last_reward   = reward
                st.session_state.is_done       = data.get("done", False)
                st.session_state.reward_history.append(reward)
                _step_log(parsed, reward, data.get("info", {}), data.get("done", False))
                st.rerun()
            else:
                st.error(res.text)
        except Exception as e:
            st.error(str(e))

# ── Main: timeline + observations ──────────────────────────────────────────
with col_main:
    st.markdown("### Availability & Schedule Timeline")
    obs = st.session_state.curr_obs
    if obs and HAS_ALTAIR:
        events = []
        for p in obs.get("participants", []):
            for a in p.get("availability", []):
                events.append({
                    "Participant": p["id"], "Name": p["name"],
                    "Start": a["start"], "End": a["end"],
                    "State": "Available", "Detail": p["timezone"],
                })
        req_dur = {r["id"]: r["duration_minutes"] for r in obs.get("requests", [])}
        for m in obs.get("scheduled_meetings", []):
            dur      = req_dur.get(m["meeting_id"], 60)
            start_dt = pd.to_datetime(m["time"])
            end_dt   = start_dt + pd.Timedelta(minutes=dur)
            for pid in m.get("participants", []):
                events.append({
                    "Participant": pid, "Name": pid,
                    "Start": start_dt.isoformat(), "End": end_dt.isoformat(),
                    "State": "Scheduled", "Detail": m["meeting_id"],
                })
        if events:
            df = pd.DataFrame(events)
            df["Start"] = pd.to_datetime(df["Start"], utc=True)
            df["End"]   = pd.to_datetime(df["End"],   utc=True)
            color_scale = alt.Scale(
                domain=["Available", "Scheduled"],
                range=["rgba(46,160,67,0.12)", "#388bfd"],
            )
            chart = (
                alt.Chart(df)
                .mark_bar(opacity=0.85, cornerRadius=2)
                .encode(
                    x=alt.X("Start:T", title="Time (UTC)"),
                    x2="End:T",
                    y=alt.Y("Participant:N", sort=None, title=""),
                    color=alt.Color("State:N", scale=color_scale),
                    tooltip=["Name", "State", "Start", "End", "Detail"],
                )
                .properties(height=300)
                .interactive()
            )
            st.altair_chart(chart, use_container_width=True)
        else:
            st.info("Initialize an episode to see the timeline.")
    elif not HAS_ALTAIR:
        st.info("Install `altair` for the timeline view.")
    else:
        st.info("Initialize an episode to see the timeline.")

    st.divider()
    st.markdown("### Observation")
    if obs:
        t1, t2, t3, t4 = st.tabs(["Pending", "Scheduled", "Profiles Read", "Raw"])
        with t1:
            sched_ids = {m["meeting_id"] for m in obs.get("scheduled_meetings", [])}
            st.json([r for r in obs.get("requests", []) if r["id"] not in sched_ids])
        with t2:
            st.json(obs.get("scheduled_meetings", []))
        with t3:
            pr = obs.get("profiles_read", {})
            if pr:
                st.json(pr)
            else:
                st.caption("No profiles discovered yet.")
        with t4:
            st.json(obs, expanded=False)

# ── Right: metrics + baselines ─────────────────────────────────────────────
with col_right:
    st.markdown("### Metrics")

    c1, c2 = st.columns(2)
    with c1:
        st.metric("Last reward", f"{st.session_state.last_reward:.2f}")
    with c2:
        cumulative = sum(st.session_state.reward_history)
        st.metric("Cumulative", f"{cumulative:.2f}")

    if st.session_state.reward_history:
        st.line_chart(st.session_state.reward_history, height=140)

    if st.session_state.step_log:
        st.markdown("**Step log**")
        for entry in st.session_state.step_log[-6:]:
            color = "#3fb950" if entry["reward"] > 0 else "#f85149" if entry["reward"] < 0 else "#8b949e"
            st.markdown(
                f"<span class='mono' style='color:{color}'>"
                f"#{entry['step']} {entry['action']}({entry['detail']}) → {entry['reward']:.2f} · {entry['result']}"
                f"</span>",
                unsafe_allow_html=True,
            )

    if st.session_state.is_done:
        st.success("Episode complete")

    if st.button("Grade episode", use_container_width=True,
                 disabled=not st.session_state.session_id):
        try:
            res = requests.get(
                f"{BASE_URL}/grader",
                params={"session_id": st.session_state.session_id},
            )
            if res.status_code == 200:
                st.session_state.grade = res.json()
        except Exception as e:
            st.error(str(e))

    if st.session_state.grade:
        g = st.session_state.grade
        s = g.get("score", 0.0)
        st.metric("Score", f"{s:.3f}", help="Strictly in (0, 1)")
        st.progress(float(s))
        with st.expander("Score breakdown"):
            st.json(g.get("components", {}))
            st.caption(f"Meetings: {g.get('completed')}/{g.get('total')}")
            st.caption(f"Profiles: {g.get('profiles_discovered')}/{g.get('profiles_needed')}")

    st.divider()
    st.markdown("### Baselines")

    bl_task = st.selectbox("Task", ["easy", "medium", "hard"], index=2, key="bl")
    if st.button("Run heuristic baseline", use_container_width=True):
        with st.spinner("Running…"):
            try:
                res = requests.post(f"{BASE_URL}/baseline", params={"task_name": bl_task})
                if res.status_code == 200:
                    data = res.json()
                    st.metric("Score", f"{data.get('final_score', 0):.3f}")
                    st.metric("Steps", data.get("steps_used", 0))
                    st.json(data.get("components", {}))
                else:
                    st.error(res.text)
            except Exception as e:
                st.error(str(e))
