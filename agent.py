from mesa import Agent
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from model import ContagionModel

class ContagionAgent(Agent):
    """Agent representing an individual in the contagion model."""
    def __init__(self, model: 'ContagionModel', awareness: float, threshold: float, awareness_incr: float, threshold_decr: float):
        """Create a new agent."""
        super().__init__(model)
        self.awareness: float = awareness
        self.threshold: float = threshold
        self.state: str = "Susceptible"
        self.awareness_incr: float = awareness_incr
        self.threshold_decr: float = threshold_decr

    def spread_awareness(self) -> None:
        """Spread awareness to neighbors if adopted."""
        if self.state == "Adopted":
            neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
            for neighbor in neighbors:
                if neighbor.state == "Susceptible":
                    neighbor.awareness += self.awareness_incr

    def get_aware(self) -> None:
        """Check if agent becomes aware."""
        if self.state == "Susceptible" and self.awareness >= 1:
            self.state = "Aware"

    def aware_sinking_threshold(self) -> None:
        """Decrease threshold for adoption if aware."""
        if self.state == "Aware":
            self.threshold = max(0, self.threshold - self.threshold_decr)

    def adopt(self) -> None:
        """Check if agent adopts behavior."""
        if self.state == "Aware":
            neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
            adopted_neighbors = sum(
                1 for neighbor in neighbors if neighbor.state == "Adopted"
            )
            if len(neighbors) > 0 and (adopted_neighbors / len(neighbors)) >= self.threshold:
                self.state = "Adopted"

    def step(self) -> None:
        """Agent step."""
        self.spread_awareness()
        self.get_aware()
        self.aware_sinking_threshold()
        self.adopt()