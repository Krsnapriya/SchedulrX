"""
SchedulrX Pydantic Schemas
===========================
Typed models for the OpenEnv API.
All new fields are Optional with defaults so existing code doesn't break.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class HiddenProfile(BaseModel):
    profile: str = Field(
        default="",
        description="Natural language description of the participant's routine and preferences."
    )
    history: List[str] = Field(
        default_factory=list,
        description="Historical interaction notes or past constraints expressed by the participant."
    )
    preferred_times: List[str] = Field(
        default_factory=list,
        description="Time-of-day preferences: morning | afternoon | evening"
    )
    avoid_days: List[str] = Field(
        default_factory=list,
        description="Day names to avoid: Monday, Friday, etc."
    )
    max_meetings_per_day: int = Field(
        default=3,
        description="Hard cap on meetings per calendar day"
    )
    fatigue_penalty: float = Field(
        default=0.2,
        description="Score penalty applied when max_meetings_per_day is exceeded"
    )
    soft_constraints: Dict[str, float] = Field(
        default_factory=dict,
        description="Named soft constraint penalties, e.g. back_to_back: -0.4"
    )


class Participant(BaseModel):
    id: str
    name: str
    timezone: str
    availability: Optional[List[Dict[str, str]]] = Field(
        default=None,
        description="List of {start, end} ISO windows. None = hidden (read profile first)."
    )


class MeetingRequest(BaseModel):
    id: str
    title: str
    duration_minutes: int
    priority: int = Field(ge=1, le=10)
    participants: List[str]
    depends_on: Optional[str] = Field(
        default=None,
        description="Meeting ID that must be scheduled before this one."
    )
    deadline_hours: Optional[int] = Field(
        default=None,
        description="Meeting must be scheduled within this many hours of episode start."
    )


class CounterProposal(BaseModel):
    proposal_id: str
    meeting_id: str
    proposed_time: str
    reason: str


class Observation(BaseModel):
    current_time: datetime
    participants: List[Participant]
    requests: List[MeetingRequest]
    scheduled_meetings: List[Dict[str, Any]] = Field(default_factory=list)
    cancelled_meetings: List[str] = Field(
        default_factory=list,
        description="IDs of meetings cancelled by the environment."
    )
    profiles_read: Dict[str, Any] = Field(
        default_factory=dict,
        description="Discovered hidden profiles keyed by participant ID."
    )
    counter_proposals: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Active counter-proposals from participants. Accept with accept_proposal."
    )
    step_count: int = 0
    trust_scores: Dict[str, int] = Field(
        default_factory=dict,
        description="Remaining honest reads per participant."
    )
    last_event: Optional[str] = Field(
        default=None,
        description="Most recent dynamic event or participant update message."
    )


class Action(BaseModel):
    action_type: str = Field(
        description=(
            "read_profile | schedule_meeting | accept_proposal | reschedule_meeting"
        )
    )
    # schedule_meeting / reschedule_meeting
    meeting_id: Optional[str] = None
    proposed_time: Optional[str] = Field(
        default=None,
        description="ISO 8601 datetime for the proposed meeting start"
    )
    # read_profile
    participant_id: Optional[str] = None
    # accept_proposal
    proposal_id: Optional[str] = Field(
        default=None,
        description="ID of a counter-proposal from the counter_proposals list"
    )


class Reward(BaseModel):
    value: float = Field(..., description="The scalar reward value for the step.")
    components: Dict[str, float] = Field(
        default_factory=dict,
        description="Optional breakdown of reward components (completion, constraints, etc.)"
    )
