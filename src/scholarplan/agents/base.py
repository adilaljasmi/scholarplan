"""Common agent contract.

Each agent is a Gaia *role* (Wooldridge, Jennings and Kinny, 2000) with:
  * responsibilities: what ``act`` does;
  * permissions: enforced by the ScopedBlackboard it receives;
  * a trigger: ``can_act`` inspects the blackboard and says whether the
    role's input is present.

The split between ``can_act`` (cheap, side-effect free) and ``act`` is what
lets the controller schedule agents opportunistically, as a blackboard
control component does (Hayes-Roth, 1985): it asks every knowledge source
whether it is applicable and runs one, rather than following a hard-coded
pipeline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from scholarplan.blackboard import Blackboard, ScopedBlackboard
from scholarplan.config import Settings


class Agent(ABC):
    role: str = "agent"
    uses_llm: bool = False

    def __init__(self, settings: Settings):
        self.settings = settings

    def scoped(self, bb: Blackboard) -> ScopedBlackboard:
        return bb.view(self.role)

    @abstractmethod
    def can_act(self, bb: Blackboard) -> bool:
        """True when this role's triggering input is on the blackboard."""

    @abstractmethod
    def act(self, bb: Blackboard) -> str:
        """Perform one unit of work; return a one-line description for the trace."""
