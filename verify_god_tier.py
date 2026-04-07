import os
import sys
import pytz
from datetime import datetime, timedelta
from env import SchedulrXEnv
from models.schemas import Action

def test_god_tier_logic():
    env = SchedulrXEnv()
    
    print("\n--- TEST: Hidden Availability (POMDP) ---")
    obs = env.reset(task_name="medium", seed=42)
    p1 = next(p for p in obs.participants if p.id == "p1")
    if p1.availability is None:
        print("✅ SUCCESS: Participant availability is hidden before profile read.")
    else:
        print("❌ FAILURE: Availability leaked prematurely!")

    print("\n--- TEST: Profile Discovery ---")
    action = Action(action_type="read_profile", participant_id="p1")
    obs, r, _, _ = env.step(action)
    p1_new = next(p for p in obs.participants if p.id == "p1")
    if p1_new.availability is not None:
        print(f"✅ SUCCESS: Availability revealed for p1. Rewards: {r}")
    else:
        print("❌ FAILURE: Availability still hidden after profile read!")

    print("\n--- TEST: Sequence Dependency (Planning) ---")
    # 'medium' has r2 depends on r1
    r2 = next(r for r in obs.requests if r.id == "r2")
    # Try scheduling r2 BEFORE r1
    proposed_time = (env.episode_start_time + timedelta(days=2)).isoformat()
    action = Action(action_type="schedule_meeting", meeting_id="r2", proposed_time=proposed_time)
    obs, r, _, info = env.step(action)
    if r < 0 and "r1" not in [m["meeting_id"] for m in obs.scheduled_meetings]:
        print(f"✅ SUCCESS: Prevented scheduling r2 before r1. Reward: {r}")
    else:
        print("❌ FAILURE: Allowed r1/r2 dependency violation!")

    print("\n--- TEST: Negotiation Engine (CounterProposals) ---")
    # Try scheduling r3 at a time Carol (p3) is busy (Carol is NOT in r3 in medium task)
    # Actually, let's just trigger a conflict for anyone.
    # We'll schedule a valid meeting r1 first.
    p1_avail = p1_new.availability[0]["start"]
    action = Action(action_type="schedule_meeting", meeting_id="r1", proposed_time=p1_avail)
    obs, r, _, _ = env.step(action)
    
    # Now try to schedule r3 at the SAME TIME as r1 (conflict)
    action = Action(action_type="schedule_meeting", meeting_id="r3", proposed_time=p1_avail)
    obs, r, _, info = env.step(action)
    if obs.counter_proposals:
        print(f"✅ SUCCESS: Conflict triggered negotiation. Proposal: {obs.counter_proposals[0]['proposed_time']}")
    else:
        print("❌ FAILURE: No counter-proposal triggered on conflict.")

if __name__ == "__main__":
    test_god_tier_logic()
