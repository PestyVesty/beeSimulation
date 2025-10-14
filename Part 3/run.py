from model import OpenCoverModel
import pickle
import tqdm as tqdm
import argparse
from mesa.batchrunner import batch_run
import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run batch simulations of the bee model.")
    parser.add_argument(
        "-b", "--behavior", type=str,
        choices=[
            "pure_levy",
            "truncated_levy",
            "composite_levy",
            "biased_composite",
            "resource_aware",
            "comparative_rewards", 
            "all"
        ],
        default="composite_levy"
    )
    parser.add_argument(
        "-lp", "--landing_prob", type=float,
        default=1.0,
        help="Probability of landing when a valid flower is under the bee."
    )
    parser.add_argument(
        "-rs", "--reward_size", type=int,
        default=5,
        help="Length of the reward memory window (used by behaviors that rely on reward memory)."
    )
    parser.add_argument(
        "-na", "--n_agents", type=int,
        default=60,
        help="Number of bee agents."
    )
    parser.add_argument(
        "-i", "--iterations", type=int,
        default=5,
        help="Number of different random seeds (repetitions) for each parameter set."
    )
    parser.add_argument(
        "-s", "--steps", type=int,
        default=5 * 60 * 60,
        help="Number of simulation steps."
    )
    parser.add_argument(
        "-np", "--num_processes", type=int,
        default=10,
        help="Number of processes to run in parallel. Use None to use all CPUs."
    )
    args = parser.parse_args()

    # Determine which behaviors to run
    if args.behavior == "all":
        behaviors_to_run = [
            "pure_levy",
            "truncated_levy",
            "composite_levy",
            "biased_composite",
            "resource_aware",
            "comparative_rewards",
            "nearest_flower"
        ]
    else:
        behaviors_to_run = [args.behavior]

    for behavior in behaviors_to_run:
        print(f"\n=== Running behavior: {behavior} ===\n")

        # Parameter dictionary for batch_run -> must match OpenCoverModel.__init__
        param_dict = {
            "n_agents": args.n_agents,
            "bee_behavior": behavior,
            "landing_prob": args.landing_prob,
            "reward_size": args.reward_size,  # safe to pass if your model forwards it to the behavior/bee
            "seed": range(args.iterations)
        }

        results = batch_run(
            model_cls=OpenCoverModel,
            parameters=param_dict,
            iterations=1,
            max_steps=args.steps,
            number_processes=args.num_processes,
            data_collection_period=10,
            display_progress=True
        )

        # Save results (pkl of the raw list, plus a CSV convenience export)
        df = pd.DataFrame(results)
        pkl_name = f"batch_results_{behavior}.pkl"
        csv_name = f"batch_results_{behavior}.csv"

        with open(pkl_name, "wb") as f:
            pickle.dump(results, f)
        df.to_csv(csv_name, index=False)

        print(f"Simulation complete. Results saved to {pkl_name} and {csv_name}")
