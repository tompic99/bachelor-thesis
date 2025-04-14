from mesa import Agent

class ContagionAgent(Agent):
    """Agent representing an individual in the contagion model."""

    def __init__(self, model, awareness, threshold, awareness_incr, threshold_decr):
        """Create a new agent."""
        super().__init__(model)
        self.awareness = awareness
        self.threshold = threshold
        self.state = "Susceptible"
        self.awareness_incr = awareness_incr
        self.threshold_decr = threshold_decr

        #print(self.threshold)

    def spread_awareness(self):
        """Spread awareness to neighbors."""
        if self.state == "Adopted":
            neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
            for neighbor in neighbors:
                if neighbor.state == "Susceptible":
                    neighbor.awareness += self.awareness_incr


    def get_aware(self):
        """Check if agent becomes aware."""
        if self.state == "Susceptible" and self.awareness >= 1:
            self.state = "Aware"

    def aware_sinking_threshold(self):
        """Decrease threshold for adoption."""
        if self.state == "Aware":
            self.threshold -= self.threshold_decr

    def adopt(self):
        """Check if agent adopts behavior."""
        if self.state == "Aware":
            neighbors = self.model.grid.get_neighbors(self.pos, include_center=False)
            adopted_neighbors = sum(
                1 for neighbor in neighbors if neighbor.state == "Adopted"
            )
            if len(neighbors) > 0 and (adopted_neighbors / len(neighbors)) >= self.threshold:
                self.state = "Adopted"

    def step(self):
        """Agent step."""
        self.spread_awareness()
        self.get_aware()
        self.aware_sinking_threshold()
        self.adopt()