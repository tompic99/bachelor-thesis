from mesa import Model
from mesa.datacollection import DataCollector
from agent import ContagionAgent
from mesa.space import NetworkGrid
import networkx as nx
import random
from typing import Optional

class ContagionModel(Model):
    """Model for simulating contagion dynamics."""
    def __init__(self,
                 num_agents: int = 10,
                 network_type: str = "small_world",
                 seed_fraction: float = 0.2,
                 seeding_strategy: str = "high_degree",
                 awareness_incr: float = 0.1,
                 threshold_decr: float = 0.02,
                 sw_k: float = 0.3,
                 sw_p: float = 0.2,
                 r_p: float = 0.3,
                 sf_m: int = 3,
                 th_init_min: float = 0,
                 th_init_max: float = 1,
                 seed: Optional[int] = None):
        super().__init__()

        # Set seed for reproducibility
        self.seed = seed
        self.random = random.Random(seed)
        random.seed(seed)

        # Create network
        if network_type == "small_world":
            k = max(2, int(num_agents * sw_k))
            self.network = nx.watts_strogatz_graph(num_agents, k=k, p=sw_p, seed=seed)
        elif network_type == "random":
            self.network = nx.erdos_renyi_graph(num_agents, p=r_p, seed=seed)
        elif network_type == "scale_free":
            self.network = nx.barabasi_albert_graph(num_agents, m=sf_m, seed=seed)
        else:
            raise ValueError("Invalid network type. Choose from 'small_world', 'random', or 'scale_free'.")
        self.grid = NetworkGrid(self.network)

        # Data collector
        self.datacollector = DataCollector(
            agent_reporters={"State": "state", "Awareness": "awareness", "Threshold": "threshold"},
            model_reporters={
                "Susceptible": lambda m: sum(1 for a in m.agents if a.state == "Susceptible"),
                "Aware": lambda m: sum(1 for a in m.agents if a.state == "Aware"),
                "Adopted": lambda m: sum(1 for a in m.agents if a.state == "Adopted"),
            }
        )

        # Create agents
        for i in range(num_agents):
            awareness = 0.0
            threshold = self.random.uniform(th_init_min, th_init_max)
            agent = ContagionAgent(self, awareness, threshold, awareness_incr, threshold_decr)
            self.grid.place_agent(agent, i)

        # Seed initial adopters
        seed_count = max(1, int(seed_fraction * num_agents))
        seed_nodes = []
        if seeding_strategy == "random":
            seed_nodes = self.random.sample(range(num_agents), seed_count)
        elif seeding_strategy == "high_degree":
            degree_sorted_nodes = sorted(self.network.degree, key=lambda x: x[1], reverse=True)
            seed_nodes = [node for node, degree in degree_sorted_nodes[:seed_count]]
        elif seeding_strategy == "low_degree":
            degree_sorted_nodes = sorted(self.network.degree, key=lambda x: x[1])
            seed_nodes = [node for node, degree in degree_sorted_nodes[:seed_count]]
        elif seeding_strategy == "hybrid":
            degree_sorted_nodes = sorted(self.network.degree, key=lambda x: x[1], reverse=True)
            half_high = (seed_count + 1) // 2
            half_random = seed_count // 2
            seed_nodes = [node for node, degree in degree_sorted_nodes[:half_high]]
            remaining = set(range(num_agents)) - set(seed_nodes)
            if remaining and half_random > 0:
                seed_nodes += self.random.sample(list(remaining), min(half_random, len(remaining)))
        elif seeding_strategy == "community":
            communities = list(nx.algorithms.community.greedy_modularity_communities(self.network))
            per_community = max(1, seed_count // len(communities)) if communities else seed_count
            for community in communities:
                if len(seed_nodes) < seed_count:
                    degree_sorted_nodes = sorted(self.network.degree(community), key=lambda x: x[1], reverse=True)
                    seed_nodes += [node for node, degree in degree_sorted_nodes[:per_community]]
            seed_nodes = seed_nodes[:seed_count]
        else:
            raise ValueError(f"Unknown seeding_strategy: {seeding_strategy}")

        for node in seed_nodes:
            agent = self.grid.get_cell_list_contents([node])[0]
            agent.state = "Adopted"

    def step(self) -> None:
        """Advance the model by one step."""
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)
        
        # Check if all agents have adopted, if so, stop the simulation
        if all(agent.state == "Adopted" for agent in self.agents):
            self.running = False
