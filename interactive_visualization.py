import mesa
print(f"mesa.__version__: {mesa.__version__}")

from mesa.visualization import Slider, SolaraViz, make_plot_component, make_space_component
from model import ContagionModel
import solara  # Import solara for reactivity

def agent_portrayal(agent):
    color = "tab:green"
    if agent.state == "Susceptible":
        color = "tab:red"
    elif agent.state == "Aware":
        color = "tab:blue"
    return {"size": 100, "color": color}

"""
# Define network type as a stateful variable
@solara.component
def Page():
    network_type, set_network_type = solara.use_state("small_world")  # Dynamic state

    def get_model_params():
        params = {
            "num_agents": {
                "type": "SliderInt",
                "label": "Number of Agents",
                "value": 10,
                "min": 10,
                "max": 100,
                "step": 10
            },
            "network_type": {
                "type": "Select",
                "label": "Network Type",
                "value": network_type,  # Use state variable
                "values": ["small_world", "random", "scale_free"],
                "on_change": set_network_type  # Reactively update state
            },
            "seeding_strategy": {
                "type": "Select",
                "label": "Seeding Strategy",
                "value": "random",
                "values": ["random", "high_degree", "low_degree", "hybrid", "community"]
            },
            "seed_fraction": {
                "type": "SliderFloat",
                "label": "Seed Fraction",
                "value": 0.1,
                "min": 0,
                "max": 1,
                "step": 0.1
            },
        }

        # Conditionally add parameters based on the selected network type
        if network_type == "small_world":
            params["sw_k"] = {
                "type": "SliderFloat",
                "label": "Average Node Degree",
                "value": 4,
                "min": 1,
                "max": 10,
                "step": 1
            }
            params["sw_p"] = {
                "type": "SliderFloat",
                "label": "Rewiring Probability",
                "value": 0.1,
                "min": 0,
                "max": 1,
                "step": 0.1
            }
        elif network_type == "random":
            params["r_p"] = {
                "type": "SliderFloat",
                "label": "Connection Probability",
                "value": 0.1,
                "min": 0,
                "max": 1,
                "step": 0.1
            }

        return params
"""

model_params = {
    "num_agents": {
        "type": "SliderInt",
        "label": "Number of Agents",
        "value": 10,
        "min": 10,
        "max": 100,
        "step": 10
    },
    "network_type": {
        "type": "Select",
        "label": "Network Type",
        "value": "small_world",
        "values": ["small_world", "random", "scale_free"]
    },
    "seeding_strategy": {
        "type": "Select",
        "label": "Seeding Strategy",
        "value": "random",
        "values": ["random", "high_degree", "low_degree", "hybrid", "community"]
    },
    "seed_fraction": {
        "type": "SliderFloat",
        "label": "Seed Fraction",
        "value": 0.1,
        "min": 0,
        "max": 1,
        "step": 0.1
    },
}

def post_process_lineplot(ax):
    ax.set_ylim(ymin=0)
    ax.set_ylabel("# people")
    ax.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")

SpacePlot = make_space_component(agent_portrayal)
StatePlot = make_plot_component(
    {"Susceptible": "tab:red", "Aware": "tab:blue", "Adopted": "tab:green"},
    post_process=post_process_lineplot
)

model1 = ContagionModel()

page = SolaraViz(
    model1,
    components=[SpacePlot, StatePlot],
    model_params=model_params,  # Ensure function is used
    name="Contagion Model"
)

# Run the component
page