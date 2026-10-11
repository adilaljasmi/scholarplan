"""The five ScholarPlan roles (Unit 6, Figure 1)."""

from scholarplan.agents.archivist import ArchivistAgent
from scholarplan.agents.base import Agent
from scholarplan.agents.critic import CriticAgent
from scholarplan.agents.planner import PlannerAgent
from scholarplan.agents.processor import ProcessorAgent
from scholarplan.agents.retriever import RetrieverAgent

__all__ = ["Agent", "PlannerAgent", "RetrieverAgent", "ProcessorAgent", "CriticAgent", "ArchivistAgent"]
