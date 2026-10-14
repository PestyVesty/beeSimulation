"""Command-line entry point for batch bee simulations."""

import argparse
import pickle
from pathlib import Path

import pandas as pd
from mesa.batchrunner import batch_run

from .model import BEHAVIOR_CLASSES, OpenCoverModel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run batches of the polytunnel bee simulation."
    )
    parser.add_argument(
        "-b",
        "--behavior",
        choices=[*BEHAVIOR_CLASSES, "all"],
        default="resource_aware",
        help="Bee foraging strategy to simulate.",
    )
    parser.add_argument(
        "-lp",
        "--landing-probability",
        type=float,
        default=1.0,
        help="Probability of landing when a valid flower is beneath a bee.",
    )
    parser.add_argument(
        "-rr",
        "--replenish-rate",
        type=float,
        default=0.0,
        help="Nectar replenishment per simulation step.",
    )
    parser.add_argument(
        "-rs",
        "--reward-size",
        type=int,
        default=5,
        help="Number of observations in a bee's reward-memory window.",
    )
    parser.add_argument(
        "-n",
        "--bees-per-hive",
        type=int,
        default=60,
        help="Number of bees created at each of the two hives.",
    )
    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=5,
        help="Number of random seeds per parameter set.",
    )
    parser.add_argument(
        "-s",
        "--steps",
        type=int,
        default=5 * 60 * 60,
        help="Number of simulation steps per run.",
    )
    parser.add_argument(
        "-p",
        "--processes",
        type=int,
        default=1,
        help="Number of worker processes.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for CSV and pickle results.",
    )
    return parser.parse_args()


def run_batch(args):
    behaviors = list(BEHAVIOR_CLASSES) if args.behavior == "all" else [args.behavior]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for index, behavior in enumerate(behaviors, start=1):
        print(f"[{index}/{len(behaviors)}] Simulating bee behavior: {behavior}")
        results = batch_run(
            model_cls=OpenCoverModel,
            parameters={
                "n_agents": args.bees_per_hive,
                "bee_behavior": behavior,
                "landing_prob": args.landing_probability,
                "replenish_rate": args.replenish_rate,
                "reward_size": args.reward_size,
            },
            rng=range(args.iterations),
            max_steps=args.steps,
            number_processes=args.processes,
            data_collection_period=10,
            display_progress=True,
        )

        stem = args.output_dir / f"batch_results_{behavior}"
        print(f"Saving {behavior} results to {args.output_dir}")
        with stem.with_suffix(".pkl").open("wb") as file:
            pickle.dump(results, file)
        pd.DataFrame(results).to_csv(stem.with_suffix(".csv"), index=False)

    print(f"Completed {len(behaviors)} behavior batch(es).")


def main():
    run_batch(parse_args())


if __name__ == "__main__":
    main()
