from params1 import *
from model import ContagionModel

from mesa.batchrunner import batch_run
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

results = batch_run(
    ContagionModel,
    parameters=params,
    iterations=iterations,
    max_steps=max_steps,
    number_processes=number_processes,
    data_collection_period=data_collection_period,
    display_progress=display_progress,
)

results_df = pd.DataFrame(results)

# Plot 1: Cumulative Adoption Over Time

mean_results = (
    results_df.groupby(["Step", "seeding_strategy", "seed_fraction"])[["Adopted"]]
    .mean()
    .reset_index()
)

seed_fractions = mean_results["seed_fraction"].unique()
fig, axes = plt.subplots(1, len(seed_fractions), figsize=(18, 6), sharey=True)

for i, seed_fraction in enumerate(seed_fractions):
    filtered_results = mean_results[mean_results["seed_fraction"] == seed_fraction]

    sns.lineplot(
        data=filtered_results,
        x="Step",
        y="Adopted",
        hue="seeding_strategy",
        markers=True,
        ax=axes[i],
    )
    axes[i].set_title(f"Seed Fraction = {seed_fraction:.1f}")
    axes[i].set_xlabel("Step")
    if i == 0:
        axes[i].set_ylabel("Number of Adopted Agents")
    else:
        axes[i].set_ylabel("")

# Adjust layout
plt.tight_layout()

# Save the figure as a PNG file
plt.savefig("seeding_strategy_comparison.png", dpi=300)

# Close the figure to free memory
plt.close()

# Plot 2: Adoption Fraction Over Time

results_df["Total"] = results_df["Susceptible"] + results_df["Aware"] + results_df["Adopted"]

results_df["Adopted_Fraction"] = results_df["Adopted"] / results_df["Total"]

mean_fractions = (
    results_df.groupby(["Step", "seeding_strategy", "seed_fraction"])[["Adopted_Fraction"]]
    .mean()
    .reset_index()
)

# Extract unique seed fractions
seed_fractions = mean_fractions["seed_fraction"].unique()

# Create subplots
fig, axes = plt.subplots(1, len(seed_fractions), figsize=(18, 6), sharey=True)

for i, seed_fraction in enumerate(seed_fractions):
    # Filter data for the current seed fraction
    filtered_results = mean_fractions[mean_fractions["seed_fraction"] == seed_fraction]

    # Create lineplot on the current axis
    sns.lineplot(
        data=filtered_results,
        x="Step",
        y="Adopted_Fraction",
        hue="seeding_strategy",
        markers=True,
        ax=axes[i],
    )

    # Set title and labels
    axes[i].set_title(f"Seed Fraction = {seed_fraction:.1f}")
    axes[i].set_xlabel("Step")
    if i == 0:
        axes[i].set_ylabel("Fraction of Agents Adopted")
    else:
        axes[i].set_ylabel("")

# Adjust layout and save the figure
plt.tight_layout()
plt.savefig("adoption_fraction_subplots.png", dpi=300)
plt.show()


# Plot 3: Time to Peak Adoption

# Calculate the time step where peak adoption occurs for each seeding strategy
peak_time = (
    results_df.groupby("seeding_strategy")
    .apply(lambda x: x.loc[x["Adopted"].idxmax(), "Step"])
    .reset_index(name="Peak Time")
)

# Plot the peak time for each strategy
plt.figure(figsize=(10, 6))
sns.barplot(data=peak_time, x="seeding_strategy", y="Peak Time", palette="viridis")
plt.title("Time to Peak Adoption by Seeding Strategy")
plt.xlabel("Seeding Strategy")
plt.ylabel("Time to Peak Adoption (Steps)")  # Added unit to the y-axis label
plt.tight_layout()

# Save the figure as a PNG file
plt.savefig("time_to_peak_adoption.png", dpi=300)

# Close the figure to free memory
plt.close()


# Plot 4: Final Adoption Level Comparison

# Calculate final adoption levels for each seeding strategy
final_adoption = (
    results_df[results_df["Step"] == results_df["Step"].max()]
    .groupby("seeding_strategy")["Adopted"]
    .mean()
    .reset_index()
)

# Plot the final adoption levels
plt.figure(figsize=(10, 6))
sns.barplot(data=final_adoption, x="seeding_strategy", y="Adopted", palette="coolwarm")
plt.title("Final Adoption Levels by Seeding Strategy")
plt.xlabel("Seeding Strategy")
plt.ylabel("Final Number of Adopted Agents")
plt.tight_layout()

# Save the figure as a PNG file
plt.savefig("final_adoption_levels.png", dpi=300)

# Close the figure to free memory
plt.close()

# Plot 5: Adoption Speed vs. Seeding Efficiency

# Calculate Time to 50% Adoption
adoption_speed = (
    results_df[results_df["Adopted_Fraction"] >= 0.5]
    .groupby("seeding_strategy")["Step"]
    .min()
    .reset_index(name="Time to 50% Adoption")
)

# Calculate Final Adoption Levels
adoption_effectiveness = (
    results_df[results_df["Step"] == results_df["Step"].max()]
    .groupby("seeding_strategy")["Adopted"]
    .mean()
    .reset_index(name="Final Adoption")
)

# Merge Speed and Effectiveness Metrics
adoption_metrics = pd.merge(adoption_speed, adoption_effectiveness, on="seeding_strategy")

# Plot Adoption Speed vs. Seeding Efficiency
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=adoption_metrics,
    x="Time to 50% Adoption",
    y="Final Adoption",
    hue="seeding_strategy",
    style="seeding_strategy",
    s=150,  # Size of markers
)
plt.title("Adoption Speed vs. Seeding Efficiency")
plt.xlabel("Time to 50% Adoption (Steps)")
plt.ylabel("Final Number of Adopted Agents")
plt.legend(title="Seeding Strategy")
plt.tight_layout()

# Save the figure as a PNG file
plt.savefig("adoption_speed_vs_efficiency.png", dpi=300)

# Close the figure to free memory
plt.close()

"""
plt.figure(figsize=(14, 8))
sns.lineplot(
    data=mean_results,  # Replace with your aggregated DataFrame
    x="Step",
    y="Adopted",
    hue="seeding_strategy",
    style="seed_fraction",
    markers=True,
)
plt.title("Cumulative Adoption Over Time by Seeding Strategy")
plt.xlabel("Step")
plt.ylabel("Number of Adopted Agents")
plt.legend(title="Seeding Strategy")
plt.tight_layout()
plt.show()


# Parameters
num_agents = 5
network_type = "small_world"
rewiring_prob = 0.2
seed_fraction = 0.2
num_steps = 10

# Initialize model
model = ContagionModel(
    num_agents=num_agents,
    network_type=network_type,
    rewiring_prob=rewiring_prob,
    seed_fraction=seed_fraction,
)

# Run the simulation
for step in range(num_steps):
    model.step()

print(model.datacollector.get_agent_vars_dataframe())
"""
"""results_df = pd.DataFrame(model)
print(results_df.keys())"""
"""
params = {"num_agents": [50, 100, 200], "network_type": ["small_world", "random"], "rewiring_prob": [0.1, 0.2, 0.3], "seed_fraction": [0.1, 0.2, 0.3]}

if True:
    results = batch_run(
        ContagionModel,
        parameters=params,
        iterations=2,
        max_steps=100,
        number_processes=1,
        data_collection_period=1,
        display_progress=True,
    )

results_df = pd.DataFrame(results)
print(results_df.keys())
results_df.to_csv("results.csv")

f1 = plt.figure(1)
df_grouped = (
    results_df.groupby(["Step", "network_type", "State"])
    .size()
    .reset_index(name="Count")
)

df_filtered = df_grouped[df_grouped["Step"] < 30]

sns.lineplot(data=df_filtered,
             x="Step",
             y="Count",
             hue="State",
             style="network_type",
             markers=True,
             dashes=False
             )

f2= plt.figure(2)

df_totals = (
    results_df.groupby(["Step", "network_type"])
    .size()
    .reset_index(name="Total")
)

df_grouped = pd.merge(df_grouped, df_totals, on=["Step", "network_type"])

df_grouped["Fraction"] = df_grouped["Count"] / df_grouped["Total"]

df_filtered = df_grouped[df_grouped["Step"] < 30]

sns.lineplot(
    data=df_filtered,
    x="Step",
    y="Fraction",
    hue="State",
    style="network_type",
    markers=True,
    dashes=False,
)
plt.show()



params = {"num_agents": [100, 500, 1000], "network_type": ["small_world", "random", "scale_free"], "seed_fraction": [0.1, 0.2, 0.3]}
results = batch_run(
    ContagionModel,
    parameters=params,
    iterations=2,
    max_steps=100,
    number_processes=1,
    data_collection_period=1,
    display_progress=True,
)
results =results.gr

# Parametes seeding_strategy
# Define parameters
params = {
    "num_agents": [1000],
    "network_type": ["random"],
    "seed_fraction": [0.1, 0.2, 0.3],
    "seeding_strategy": ["random", "high_degree", "low_degree", "hybrid", "community"],
}

# Run batch
results = batch_run(
    ContagionModel,
    parameters=params,
    iterations=1,
    max_steps=50,
    number_processes=1,
    data_collection_period=1,
    display_progress=True,
)

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Calculate average results
mean_results = (
    results_df.groupby(["Step", "seeding_strategy", "seed_fraction"])[["Susceptible", "Aware", "Adopted"]]
    .mean()
    .reset_index()
)

# Save results to CSV
mean_results.to_csv("seeding_comparison_results.csv", index=False)


# Plot: Number of Agents in Each State over Time by Seeding Strategy
plt.figure(figsize=(14, 8))
sns.lineplot(
    data=mean_results,
    x="Step",
    y="Adopted",
    hue="seeding_strategy",
    style="seed_fraction",
    markers=True,
)
plt.title("Comparison of Seeding Strategies: Adoption Over Time")
plt.xlabel("Step")
plt.ylabel("Number of Agents Adopted")
plt.legend(title="Seeding Strategy")
plt.tight_layout()
plt.show()

# Plot: Fraction of States over Time by Seeding Strategy
results_df["Total"] = results_df["Susceptible"] + results_df["Aware"] + results_df["Adopted"]
results_df["Adopted_Fraction"] = results_df["Adopted"] / results_df["Total"]

mean_fractions = (
    results_df.groupby(["Step", "seeding_strategy"])[["Adopted_Fraction"]]
    .mean()
    .reset_index()
)

plt.figure(figsize=(14, 8))
sns.lineplot(
    data=mean_fractions,
    x="Step",
    y="Adopted_Fraction",
    hue="seeding_strategy",
    markers=True,
)
plt.title("Comparison of Seeding Strategies: Fraction Adopted Over Time")
plt.xlabel("Step")
plt.ylabel("Fraction of Agents Adopted")
plt.legend(title="Seeding Strategy")
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))
sns.lineplot(
    data=mean_results,
    x="Step",
    y="Adopted",
    hue="seeding_strategy",
    style="seed_fraction",
    markers=True,
)
plt.title("Cumulative Adoption Over Time by Seeding Strategy")
plt.xlabel("Step")
plt.ylabel("Number of Adopted Agents")
plt.legend(title="Seeding Strategy")
plt.tight_layout()
plt.show()

results_df["Adopted_Fraction"] = results_df["Adopted"] / results_df["Total"]
plt.figure(figsize=(14, 8))
sns.lineplot(
    data=mean_fractions,
    x="Step",
    y="Adopted_Fraction",
    hue="seeding_strategy",
    style="seed_fraction",
)
plt.title("Adoption Fraction Over Time by Seeding Strategy")
plt.xlabel("Step")
plt.ylabel("Fraction Adopted")
plt.legend(title="Seeding Strategy")
plt.tight_layout()
plt.show()

results_df["Total"] = results_df["Susceptible"] + results_df["Aware"] + results_df["Adopted"]
results_df["Adopted_Fraction"] = results_df["Adopted"] / results_df["Total"]




mean_fractions = (
    results_df.groupby(["Step", "seeding_strategy"])[["Adopted_Fraction"]]
    .mean()
    .reset_index()
)







#Plot cumulative adoption over time grouped by seeding strategy
fig1 =plt.figure(1)
plt.figure(figsize=(14, 8))
sns.lineplot(
    data=mean_results,
    x="Step",
    y="Adopted",
    hue="seeding_strategy",
    style="seed_fraction",
    markers=True,
)
plt.title("Cumulative Adoption Over Time by Seeding Strategy")
plt.xlabel("Step")
plt.ylabel("Number of Adopted Agents")
plt.legend(title="Seeding Strategy")
plt.tight_layout()
plt.show()
fig1.savefig("cumulative_adoption_over_time.png")
plt.close()

#Plot adoption fraction over time grouped by seeding strategy
fig2 = plt.figure(2)
plt.figure(figsize=(14, 8))
sns.lineplot(
    data=mean_fractions,
    x="Step",
    y="Adopted_Fraction",
    hue="seeding_strategy",
    style="seed_fraction",
)
plt.title("Adoption Fraction Over Time by Seeding Strategy")
plt.xlabel("Step")
plt.ylabel("Fraction Adopted")
plt.legend(title="Seeding Strategy")
plt.tight_layout()
plt.show()
fig2.savefig("adoption_fraction_over_time.png")
plt.close()

#Bar plot comparing the time to taken to reach peak adoption for each seeding strategy
peak_results = results_df.groupby(["seeding_strategy", "seed_fraction"])["Adopted"].max().reset_index()
peak_results["Time_to_Peak"] = results_df.groupby(["seeding_strategy", "seed_fraction"])["Step"].max().reset_index()["Step"]
peak_results["Fraction_Adopted"] = peak_results["Adopted"] / 1000

fig3 = plt.figure(3)
plt.figure(figsize=(14, 8))
sns.barplot(
    data=peak_results,
    x="seeding_strategy",
    y="Time_to_Peak",
    hue="seed_fraction",
)
plt.title("Time to Peak Adoption by Seeding Strategy")
plt.xlabel("Seeding Strategy")
plt.ylabel("Time to Peak Adoption")
plt.legend(title="Seed Fraction")
plt.tight_layout()
plt.show()
fig3.savefig("time_to_peak_adoption.png")
plt.close()




# Parameters random
params = {"num_agents": [1000], "network_type": ["random"], "seed_fraction": [0.3], "seeding_strategy": ["high_degree"]} #, "random", "scale_free"
results = batch_run(
    ContagionModel,
    parameters=params,
    iterations=30,
    max_steps=50,
    number_processes=1,
    data_collection_period=1,
    display_progress=True,
)
results_df = pd.DataFrame(results)
mean_results = (results_df.groupby(["Step"])[["Susceptible", "Aware", "Adopted"]]
                .mean()
                .reset_index())

print(results_df.columns)
print(mean_results.columns)
mean_results.to_csv("results.csv")

plt.figure(figsize=(10, 6))
sns.lineplot(data=mean_results, x="Step", y="Susceptible", label="Susceptible")
sns.lineplot(data=mean_results, x="Step", y="Aware", label="Aware")
sns.lineplot(data=mean_results, x="Step", y="Adopted", label="Adopted")
plt.title(f"State Development (Number of Agents: 1000, Network: random, Seed Fraction: 0.3)")
plt.xlabel("Step")
plt.ylabel("Number of Agents")
plt.legend()
plt.show()
# End Parameters random




# Parameters small_world
params = {"num_agents": [1000], "network_type": ["small_world"], "seed_fraction": [0.3]} #, "random", "scale_free"
results = batch_run(
    ContagionModel,
    parameters=params,
    iterations=30,
    max_steps=50,
    number_processes=1,
    data_collection_period=1,
    display_progress=True,
)
results_df = pd.DataFrame(results)
mean_results = (results_df.groupby(["Step"])[["Susceptible", "Aware", "Adopted"]]
                .mean()
                .reset_index())

print(results_df.columns)
print(mean_results.columns)
mean_results.to_csv("results.csv")

plt.figure(figsize=(10, 6))
sns.lineplot(data=mean_results, x="Step", y="Susceptible", label="Susceptible")
sns.lineplot(data=mean_results, x="Step", y="Aware", label="Aware")
sns.lineplot(data=mean_results, x="Step", y="Adopted", label="Adopted")
plt.title(f"State Development (Number of Agents: 1000, Network: small_world, Seed Fraction: 0.3)")
plt.xlabel("Step")
plt.ylabel("Number of Agents")
plt.legend()
plt.show()
# End Parameters small_world

average_results = results_df.groupby(["num_agents", "network_type", "seed_fraction", "Step"]).mean(numeric_only=True).reset_index()
print(len(average_results))
print(average_results)
print(len(results))
average_results.to_csv("results.csv")


def network_metrics(G):
    metrics = {
        "Num Nodes": G.number_of_nodes(),
        "Num Edges": G.number_of_edges(),
        "Avg Degree": sum(dict(G.degree()).values()) / G.number_of_nodes(),
        "Avg Clustering Coefficient": nx.average_clustering(G),
        "Avg Shortest Path Length": nx.average_shortest_path_length(G),
        "Diameter": nx.diameter(G),
    }
    return metrics





# Plot results
#lineplot with steps on x axis and number of agents per state on y axis
g = sns.lineplot(data=results_df, x="Step", y="Susceptible")
g = sns.lineplot(data=results_df, x="Step", y="Aware")
g = sns.lineplot(data=results_df, x="Step", y="Adopted")
g.set(title="Model Results", xlabel="Step", ylabel="Number of Agents")
plt.show()



if False:
    results_filtered = results_df[(results_df.Step == 100)]
    g = sns.histplot(results_df.State, discrete=True)
    g.set(title="Final Adopted Agents Distribution", xlabel="State", ylabel="Number of Agents")
    plt.show()

multiple_num_agents = [50, 100, 200]
adoption_multiple_num_agents = results_df[results_df.num_agents.isin(multiple_num_agents)]
g = sns.lineplot(data=adoption_multiple_num_agents, x="Step")
g.set(title="Adopted Agents Over Time", xlabel="Step", ylabel="Number of Agents")
plt.show()
"""

