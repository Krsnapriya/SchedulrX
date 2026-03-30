from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from datetime import datetime
import uuid

class Participant(BaseModel):
    id: str
    name: str
    timezone: str
    availability: List[Dict[str, str]]  # public slots

class MeetingRequest(BaseModel):
    id: str
    title: str
    duration_minutes: int
    priority: int  # 1-10
    participants: List[str]

class HiddenProfile(BaseModel):
    preferred_times: List[str]  # e.g. ["morning", "afternoon"]
    avoid_days: List[str]
    max_meetings_per_day: int
    fatigue_penalty: float  # 0.0-1.0
    soft_constraints: Dict[str, float]  # e.g. {"back_to_back": -0.4}

class Observation(BaseModel):
    current_time: datetime
    participants: List[Participant]
    requests: List[MeetingRequest]
    scheduled_meetings: List[Dict]
    profiles_read: Dict[str, HiddenProfile]
    step_count: int
    info: Optional[Dict] = Field(default_factory=dict)

class Action(BaseModel):
    action_type: str = Field(..., pattern="^(schedule_meeting|reschedule_meeting|cancel_meeting|read_profile|propose_alternative)$")
    meeting_id: Optional[str] = None
    proposed_time: Optional[str] = None  # ISO format
    participant_id: Optional[str] = None
    alternative_slot: Optional[str] = None

class Reward(BaseModel):
    value: float
    reason: str