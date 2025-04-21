from mesa import Model
from mesa.datacollection import DataCollector
from agent import ContagionAgent
from mesa.space import NetworkGrid
import networkx as nx
import random
import matplotlib.pyplot as plt

class ContagionModel(Model):
    """Model for simulating contagion dynamics."""
    def __init__(self,
                 num_agents=10,
                 network_type="small_world",
                 seed_fraction=0.2,
                 seeding_strategy="high_degree",
                 awareness_incr=0.1,
                 threshold_decr=0.02,
                 sw_k=0.3,
                 sw_p=0.2,
                 r_p=0.3,
                 sf_m=3,
                 th_init_min=0,
                 th_init_max=1,
                 seed=None):
        super().__init__()

        # Set seed for reproducibility
        random.seed(seed)

        # Create network
        if network_type == "small_world":
            self.network = nx.watts_strogatz_graph(num_agents, k=max(2, int(num_agents*sw_k)), p=sw_p)#k=6,p=0.1
            #self.network = nx.watts_strogatz_graph(num_agents, k=4, p=rewiring_prob)
        elif network_type == "random":
            self.network = nx.erdos_renyi_graph(num_agents, p=r_p) #p=r/(num_agents-1)
            #self.network = nx.erdos_renyi_graph(num_agents, p=rewiring_prob)
        elif network_type == "scale_free":
            self.network = nx.barabasi_albert_graph(num_agents, m=sf_m)
        else:
            raise ValueError("Invalid network type. Choose from 'small_world', 'random', or 'scale_free'.")
        self.grid = NetworkGrid(self.network)

        #network visualization
        #nx.draw(self.network)
        #plt.show()

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
            awareness = 0 #random.uniform(0.0, 0.5)
            threshold = random.uniform(th_init_min, th_init_max) #i/num_agents#random.uniform(0,1) #random.uniform(0.1, 0.5)
            agent = ContagionAgent(self, awareness, threshold, awareness_incr, threshold_decr)
            self.grid.place_agent(agent, i)

        # Seed initial adopters
        seed_count = int(seed_fraction * num_agents)
        seed_nodes = []
        #seed_nodes = random.sample(self.network.nodes, seed_count)
        if seeding_strategy == "random":
            seed_nodes = random.sample(range(num_agents), seed_count)
        elif seeding_strategy == "high_degree":
            degree_sorted_nodes = sorted(self.network.degree, key=lambda x: x[1], reverse=True)
            seed_nodes = [node for node, degree in degree_sorted_nodes[:seed_count]]
        elif seeding_strategy == "low_degree":
            degree_sorted_nodes = sorted(self.network.degree, key=lambda x: x[1], reverse=False)
            seed_nodes = [node for node, degree in degree_sorted_nodes[:seed_count]]
        elif seeding_strategy == "hybrid": #half high degree, half random
            degree_sorted_nodes = sorted(self.network.degree, key=lambda x: x[1], reverse=True)
            seed_nodes = [node for node, degree in degree_sorted_nodes[:(seed_count+1)//2]]
            seed_nodes += random.sample(range(num_agents), seed_count//2)
        elif seeding_strategy == "community": #detect communities, rank nodes within communities, select top nodes
            communities = nx.algorithms.community.greedy_modularity_communities(self.network)
            communities = list(communities)
            for community in communities:
                if len(seed_nodes) < seed_count:
                    degree_sorted_nodes = sorted(self.network.degree(community), key=lambda x: x[1], reverse=True)
                    seed_nodes += [node for node, degree in degree_sorted_nodes[:max(seed_count//len(communities), 1)]]
            print(seed_nodes)
            print(communities)
        for node in seed_nodes:
            agent = self.grid.get_cell_list_contents([node])[0]
            agent.state = "Adopted"

    def step(self):
        """Advance the model by one step."""
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)
        
        # Check if all agents have adopted, if so, stop the simulation
        if all(agent.state == "Adopted" for agent in self.agents):
            self.running = False
