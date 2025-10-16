# Jupyter-compatible version of run.py
from model import OpenCoverModel
from mesa.batchrunner import batch_run
import pickle
import pandas as pd
from tqdm import tqdm

# === Simulation Parameters ===
behavior = "part"  # options: "pure_levy", "truncated_levy", "composite_levy", "biased_composite", "resource_aware", "comparative_rewards", "all", "nearest_flower"
landing_prob = 1.0
n_agents = 60
iterations = 3            # Number of repetitions per setting (each with a different seed)
steps = 3 * 60 * 60       # Total number of simulation steps
num_processes = 3         # Use None for all CPUs
replenish_rate = 0
interval = 10 * 60        # Saving at every 10 minutes


# === Determine which behaviors to run ===
if behavior == "all":
    behaviors_to_run = ["pure_levy", "truncated_levy", "composite_levy", "biased_composite", "resource_aware", "comparative_rewards", "nearest_flower"]
elif behavior == "part":
    behaviors_to_run = [ "resource_aware", "comparative_rewards"]
else:
    behaviors_to_run = [behavior]

# === Run simulations ===
for b in behaviors_to_run:
    print(f"\n=== Running behavior: {b} ===\n")

    # Batch parameters
    param_dict = {
        "n_agents": n_agents,
        "bee_behavior": b,
        "landing_prob": landing_prob,
        "seed": range(iterations),
        "replenish_rate": replenish_rate,
    }

    results = batch_run(
        model_cls=OpenCoverModel,
        parameters=param_dict,
        iterations=1,
        max_steps=steps,
        number_processes=num_processes,
        data_collection_period= interval,
        display_progress=True
    )

    # Save output
    df = pd.DataFrame(results)
    output_filename = f"batch_results_{b}.pkl"
    with open(output_filename, "wb") as f:
        pickle.dump(results, f)

    print(f"Simulation complete. Results saved to {output_filename}")