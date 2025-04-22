from params import params, iterations, max_steps, number_processes, data_collection_period, display_progress
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
plt.show()

# Close the figure to free memory
plt.close()