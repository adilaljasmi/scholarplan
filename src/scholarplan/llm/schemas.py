"""Output schemas for every LLM call.

The Unit 6 design required the Planner and Processor to return *schema-valid
JSON*. Pydantic models make that contract executable: a response that does
not validate is rejected and re-requested (see ``base.call_json``), rather
than being parsed optimistically. Constraining the planner's output space in
this way is one of the main mitigations for unreliable LLM planning
(Kambhampati et al., 2024; Huang et al., 2024).
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator


class PlannedSubtask(BaseModel):
    query: str = Field(min_length=3, max_length=200)
    rationale: str = Field(default="", max_length=400)


class PlanOutput(BaseModel):
    thought: str = Field(max_length=1500)
    subtasks: list[PlannedSubtask] = Field(min_length=1)


ReActActionName = Literal["issue_subtask", "refine_subtask", "alternate_source",
                          "abandon_subtask", "synthesise"]


class ReActOutput(BaseModel):
    """One ReAct step: a Thought followed by exactly one Action (Yao et al., 2023)."""
    thought: str = Field(max_length=1500)
    action: ReActActionName
    subtask_id: Optional[int] = None
    query: Optional[str] = Field(default=None, max_length=200)
    source: Optional[Literal["arxiv", "openalex", "crossref"]] = None


class SourceSummary(BaseModel):
    source_id: int
    summary: str = Field(min_length=10)
    claims: list[str] = Field(default_factory=list)

    @field_validator("summary")
    @classmethod
    def _truncate(cls, v: str) -> str:
        # Truncate rather than reject: an over-long summary is a cosmetic
        # problem, and rejecting it cost a whole re-plan in a live run (R15).
        return v if len(v) <= 1200 else v[:1197] + "..."

    @field_validator("claims")
    @classmethod
    def _non_empty_claims(cls, v: list[str]) -> list[str]:
        return [c.strip() for c in v if c and c.strip()]


class ProcessorOutput(BaseModel):
    summaries: list[SourceSummary]


class ClaimVerdict(BaseModel):
    claim_id: int
    label: Literal["supported", "unsupported"]
    reason: str = Field(default="", max_length=600)


class CriticOutput(BaseModel):
    verdicts: list[ClaimVerdict]


class SynthesisOutput(BaseModel):
    text: str = Field(min_length=20, max_length=6000)


class SynthesisVerdict(BaseModel):
    label: Literal["supported", "unsupported"]
    reason: str = Field(default="", max_length=800)
