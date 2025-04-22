import pandas as pd

# Load CSV
df = pd.read_csv("/Users/tompichler/Projects/bachelor-thesis/results.csv")

# Ensure correct data types
df['Adopted'] = pd.to_numeric(df['Adopted'], errors='coerce')
df['Aware'] = pd.to_numeric(df['Aware'], errors='coerce')

# Group data for summarizing
summary = df.groupby([
    'RunId', 'network_type', 'seed_fraction', 'seeding_strategy',
    'awareness_incr', 'threshold_decr', 'th_init_max', 'iteration'
])

# Calculate metrics per simulation run
metrics = summary.apply(lambda x: pd.Series({
    'final_adoption_rate': x['Adopted'].iloc[-1] / x['num_agents'].iloc[0],
    'final_awareness_rate': x['Aware'].iloc[-1] / x['num_agents'].iloc[0],
    'peak_adoption_level': x['Adopted'].max() / x['num_agents'].iloc[0],
    'peak_adoption_time': x.loc[x['Adopted'].idxmax(), 'Step'],
    'time_to_50pct_adoption': x[x['Adopted'] >= 0.5 * x['num_agents'].iloc[0]]['Step'].min()
})).reset_index()

# Save summarized metrics to CSV
metrics.to_csv('data_summary.csv', index=False)