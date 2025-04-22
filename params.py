# Unified parameter file for both interactive and batch runs
params = {
    "num_agents": [50],
    "network_type": ["small_world", "scale_free", "random"],
    "seed_fraction": [0.01, 0.05, 0.1],
    "seeding_strategy": ["random", "high_degree", "community"],
    "seed": [42],
    "awareness_incr": [0.05, 0.1, 0.2],
    "threshold_decr": [0, 0.02, 0.05],
    "sw_k": [0.06],  # k = 6 for 100 agents
    "sw_p": [0.2],
    "r_p": [0.06],    # p = 6/99 for avg degree ~6
    "sf_m": [3],     # m = 3 for avg degree ~6
    "th_init_min": [0],
    "th_init_max": [0.3, 0.5, 0.8]
}

# Adjust sw_k, r_p, and sf_m to depend on num_agents and a variable degree
degree = 6  # Example average degree
params["sw_k"] = [degree / params["num_agents"][0]]
params["r_p"] = [degree / (params["num_agents"][0] - 1)]
params["sf_m"] = [degree // 2]

# Batch run settings
iterations = 10
max_steps = 50
number_processes = 1
data_collection_period = 1
display_progress = True
