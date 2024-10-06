from typing import TypeVar, Generic
from abc import abstractmethod, ABC
from dataclasses import dataclass


@dataclass
class State(ABC):
    """
    State in an adversarial MDP.
    It must somehow know whose agent's turn it is.
    """

    value: float
    current_agent: int


A = TypeVar("A")
S = TypeVar("S", bound=State)


class CompetitiveEnv(ABC, Generic[A, S]):
    """Competitive environment"""

    @abstractmethod
    def reset(self) -> S:
        """Reset the MDP to its initial state and returns it."""

    @abstractmethod
    def available_actions(self, state: S) -> list[A]:
        """Returns the list of available actions for the current agent from the given state."""

    @abstractmethod
    def step(self, state: S, action: A) -> S:
        """Returns the next state and the reward."""

    @abstractmethod
    def is_final(self, state: S) -> bool:
        """Returns whether the given state is final."""
