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
        choices=["pure_levy", "truncated_levy", "composite_levy", "biased_composite", "resource_aware", "all"],
        default="composite_levy"
    )
    parser.add_argument(
        "-lp", "--landing_prob", type=float,
        default=1.0
    )
    # parser.add_argument(
    #     "-rs", "--reward_size", type=int,
    #     default=5
    # )
    parser.add_argument(
        "-na", "--n_agents", type=int,
        default=60
    )
    parser.add_argument(
        "-i", "--iterations", type=int,
        default=5,
        help="Number of times each parameter set is repeated"
    )
    parser.add_argument(
        "-s", "--steps", type=int,
        default=5 * 60 * 60,
        help="Number of simulation steps"
    )
    parser.add_argument(
        "-np", "--num_processes", type=int,
        default=10,
        help="Number of processes to run in parallel. Use None to use all CPUs"
    )
    args = parser.parse_args()

    # Determine which behaviors to run
    if args.behavior == "all":
        behaviors_to_run = ["pure_levy", "truncated_levy", "composite_levy", "biased_composite", "resource_aware"]
    else:
        behaviors_to_run = [args.behavior]

    for behavior in behaviors_to_run:
        print(f"\n=== Running behavior: {behavior} ===\n")

        # Define parameter dictionary for batch_run
        param_dict = {
            "n_agents": args.n_agents,
            "bee_behavior": behavior,
            "landing_prob": args.landing_prob,
            # "reward_size": args.reward_size,
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

        # Save results
        df = pd.DataFrame(results)
        output_filename = f"batch_results_{behavior}.pkl"
        with open(output_filename, "wb") as f:
            pickle.dump(results, f)
        print(f"Simulation complete. Results saved to {output_filename}")

    

    # # Initialize the model with parsed arguments
    # sim_model = OpenCoverModel(
    #     seed=0,
    #     bee_behavior=args.behavior,
    #     landing_prob=args.landing_prob,
    #     reward_size=args.reward_size,
    #     n_agents=args.n_agents
    # )

    # model = OpenCoverModel(seed=0)  # Initialize the model with a seed
    # for i in tqdm.tqdm(range(7 * 60 * 60)):  # Run the model for 500 steps
    #     model.step()
        
    # df = model.datacollector.get_agent_vars_dataframe()
    
    # with open(f"simulation_data_{argparse.behavior}.pkl", "wb") as f:
    #     pickle.dump(df, f)
    # print("Model finished running.")