import mesa
print(f"mesa.__version__: {mesa.__version__}")

from mesa.visualization import SolaraViz, make_plot_component, make_space_component
from model import ContagionModel
import solara
from mesa.visualization.utils import update_counter
from matplotlib.figure import Figure
import numpy as np

def agent_portrayal(agent):
    color = "tab:green"
    if agent.state == "Susceptible":
        color = "tab:red"
    elif agent.state == "Aware":
        color = "tab:blue"
    return {"size": 100, "color": color}

def post_process_lineplot(ax):
    ax.set_ylim(ymin=0)
    ax.set_ylabel("# people")
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

# Create visualization components
SpacePlot = make_space_component(agent_portrayal)
StatePlot = make_plot_component(
    {"Susceptible": "tab:red", "Aware": "tab:blue", "Adopted": "tab:green"},
    post_process=post_process_lineplot
)
@solara.component
def ThresholdPlot(model):
    update_counter.get()
    fig = Figure()
    ax = fig.subplots()
    
    # Collect thresholds by agent state
    susceptible = [agent.threshold for agent in model.agents if agent.state == "Susceptible"]
    aware = [agent.threshold for agent in model.agents if agent.state == "Aware"]
    adopted = [agent.threshold for agent in model.agents if agent.state == "Adopted"]
    
    # Create bins for the histogram
    bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
    
    # Plot histograms for each state with transparency so they can overlap
    if susceptible:
        ax.hist(susceptible, bins=bins, alpha=0.5, color="tab:red", label="Susceptible")
    if aware:
        ax.hist(aware, bins=bins, alpha=0.5, color="tab:blue", label="Aware")
    if adopted:
        ax.hist(adopted, bins=bins, alpha=0.5, color="tab:green", label="Adopted")
    
    # Set labels and legend
    ax.set_xlabel("Threshold Value")
    ax.set_ylabel("Number of Agents")
    ax.set_title("Distribution of Adoption Thresholds by Agent State")
    ax.legend()
    
    # Ensure consistent y-axis scale
    ax.set_ylim(0, len(model.agents))
    
    solara.FigureMatplotlib(fig)

# Define model parameters
model_params = {
    "num_agents": {
        "type": "SliderInt",
        "value": 10,
        "min": 10,
        "max": 100,
        "step": 10
    },
    "network_type": {
        "type": "Select",
        "value": "small_world",
        "values": ["small_world", "random", "scale_free"]
    },
    "seeding_strategy": {
        "type": "Select",
        "value": "random",
        "values": ["random", "high_degree", "low_degree", "hybrid", "community"]
    },
    "seed_fraction": {
        "type": "SliderFloat",
        "value": 0.1,
        "min": 0,
        "max": 1,
        "step": 0.1
    },
    # Small world parameters
    "sw_k": {
        "type": "SliderFloat",
        "label": "Small World - Average Node Degree (k)",
        "value": 0.3,
        "min": 0.1,
        "max": 0.5,
        "step": 0.1
    },
    "sw_p": {
        "type": "SliderFloat",
        "label": "Small World - Rewiring Probability (p)",
        "value": 0.2,
        "min": 0,
        "max": 1,
        "step": 0.1
    },
    # Random network parameters
    "r_p": {
        "type": "SliderFloat",
        "label": "Random - Connection Probability (p)",
        "value": 0.3,
        "min": 0,
        "max": 1,
        "step": 0.1
    },
    # Scale-free network parameters
    "sf_m": {
        "type": "SliderInt",
        "label": "Scale Free - Edges Per New Node (m)",
        "value": 3,
        "min": 1,
        "max": 10,
        "step": 1
    }
}

# Create the model instance
model1 = ContagionModel()

# Create the visualization
page = SolaraViz(
    model1,
    components=[SpacePlot, StatePlot, ThresholdPlot],
    model_params=model_params,
    name="Contagion Model"
)