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

# Save results as CSV
results_df.to_csv("results.csv", index=False)

# Plot 1: Cumulative Adoption Over Time (with error bars)
mean_results = (
    results_df.groupby(["Step", "seeding_strategy", "seed_fraction", "network_type"])[["Adopted"]]
    .agg(['mean', 'std'])
    .reset_index()
)
mean_results.columns = ["Step", "seeding_strategy", "seed_fraction", "network_type", "Adopted_mean", "Adopted_std"]

seed_fractions = mean_results["seed_fraction"].unique()
network_types = mean_results["network_type"].unique()
fig, axes = plt.subplots(len(network_types), len(seed_fractions), figsize=(6*len(seed_fractions), 4*len(network_types)), sharey=True)
if len(network_types) == 1:
    axes = [axes]
if len(seed_fractions) == 1:
    axes = [[ax] for ax in axes]

for i, network_type in enumerate(network_types):
    for j, seed_fraction in enumerate(seed_fractions):
        ax = axes[i][j] if len(network_types) > 1 else axes[j]
        filtered = mean_results[(mean_results["seed_fraction"] == seed_fraction) & (mean_results["network_type"] == network_type)]
        sns.lineplot(
            data=filtered,
            x="Step",
            y="Adopted_mean",
            hue="seeding_strategy",
            marker="o",
            ax=ax,
            errorbar=None
        )
        ax.fill_between(
            filtered["Step"],
            filtered["Adopted_mean"] - filtered["Adopted_std"],
            filtered["Adopted_mean"] + filtered["Adopted_std"],
            alpha=0.2
        )
        ax.set_title(f"{network_type}, Seed Fraction={seed_fraction:.2f}")
        ax.set_xlabel("Step")
        if j == 0:
            ax.set_ylabel("# Adopted Agents")
        else:
            ax.set_ylabel("")
        ax.grid(True)
plt.tight_layout()
plt.savefig("seeding_strategy_comparison.png", dpi=300)
plt.close()

# Plot 2: Final Adoption Fraction by Seeding Strategy and Network Type
results_df["Total"] = results_df["Susceptible"] + results_df["Aware"] + results_df["Adopted"]
results_df["Adopted_Fraction"] = results_df["Adopted"] / results_df["Total"]
final_fractions = (
    results_df[results_df["Step"] == results_df["Step"].max()]
    .groupby(["seeding_strategy", "network_type", "seed_fraction"])["Adopted_Fraction"]
    .agg(['mean', 'std'])
    .reset_index()
)
final_fractions.columns = ["seeding_strategy", "network_type", "seed_fraction", "Adopted_Fraction_mean", "Adopted_Fraction_std"]
plt.figure(figsize=(10, 6))
sns.barplot(
    data=final_fractions,
    x="seeding_strategy",
    y="Adopted_Fraction_mean",
    hue="network_type",
    errorbar=None,
    capsize=0.1
)
plt.ylabel("Final Adoption Fraction")
plt.xlabel("Seeding Strategy")
plt.title("Final Adoption Fraction by Seeding Strategy and Network Type")
plt.legend(title="Network Type")
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("final_adoption_fraction.png", dpi=300)
plt.close()

# Plot 3: Time to Peak Adoption (fix groupby warning)
peak_time = (
    results_df.groupby(["seeding_strategy", "network_type", "seed_fraction"], group_keys=False)
    .apply(lambda x: x.loc[x["Adopted"].idxmax(), ["Step"]])
    .reset_index()
)
plt.figure(figsize=(10, 6))
sns.barplot(
    data=peak_time,
    x="seeding_strategy",
    y="Step",
    hue="network_type",
    capsize=0.1
)
plt.title("Time to Peak Adoption by Seeding Strategy and Network Type")
plt.xlabel("Seeding Strategy")
plt.ylabel("Time to Peak Adoption (Steps)")
plt.legend(title="Network Type")
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig("time_to_peak_adoption.png", dpi=300)
plt.close()

# Plot 4: Adoption Speed vs. Seeding Efficiency (scatter)
adoption_speed = (
    results_df[results_df["Adopted_Fraction"] >= 0.5]
    .groupby(["seeding_strategy", "network_type", "seed_fraction"])["Step"]
    .min()
    .reset_index(name="Time_to_50pct_Adoption")
)
adoption_effectiveness = final_fractions[["seeding_strategy", "network_type", "seed_fraction", "Adopted_Fraction_mean"]]
adoption_metrics = pd.merge(adoption_speed, adoption_effectiveness, on=["seeding_strategy", "network_type", "seed_fraction"])
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=adoption_metrics,
    x="Time_to_50pct_Adoption",
    y="Adopted_Fraction_mean",
    hue="seeding_strategy",
    style="network_type",
    s=120
)
plt.title("Adoption Speed vs. Final Adoption Fraction")
plt.xlabel("Time to 50% Adoption (Steps)")
plt.ylabel("Final Adoption Fraction")
plt.legend(title="Seeding Strategy / Network Type")
plt.grid(True)
plt.tight_layout()
plt.savefig("adoption_speed_vs_efficiency.png", dpi=300)
plt.close()

# Plot 5: Adoption Curve by Network Type (new plot)
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=mean_results,
    x="Step",
    y="Adopted_mean",
    hue="network_type",
    style="seeding_strategy",
    errorbar=None
)
plt.title("Adoption Curve by Network Type and Seeding Strategy")
plt.xlabel("Step")
plt.ylabel("# Adopted Agents (mean)")
plt.legend(title="Network Type / Seeding Strategy")
plt.grid(True)
plt.tight_layout()
plt.savefig("adoption_curve_by_network_type.png", dpi=300)
plt.close()

# Additional Plot: Adoption Threshold Distribution
thresholds = results_df.groupby(["Step", "seeding_strategy", "network_type"])["Threshold"].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.lineplot(
    data=thresholds,
    x="Step",
    y="Threshold",
    hue="seeding_strategy",
    style="network_type",
    marker="o"
)
plt.title("Average Adoption Threshold Over Time")
plt.xlabel("Step")
plt.ylabel("Average Threshold")
plt.legend(title="Seeding Strategy / Network Type")
plt.grid(True)
plt.tight_layout()
plt.savefig("threshold_over_time.png", dpi=300)
plt.close()
