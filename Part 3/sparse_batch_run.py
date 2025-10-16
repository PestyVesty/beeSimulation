# sparse_batch_run.py (Mesa-3 style, matching batch_run.py structure)

"""Sparse batchrunner that respects models collecting at arbitrary intervals.

Unlike mesa.batchrunner.batch_run (which assumes per-step model_vars lists and
indexes them with a step number), this runner:
- Lets the model decide when to call datacollector.collect(self) (e.g., every 600 steps).
- Reads whatever rows were actually collected.
- Pairs model rows with agent rows via the real step number saved in model_reporters
  (REQUIRES a 'Step' reporter: e.g., 'Step': lambda m: m.steps).

Multiprocessing notes are the same as Mesa’s batch_run: prefer running from a
normal Python script (guarded with if __name__ == "__main__":) rather than a notebook.
"""

import itertools
import multiprocessing
from collections.abc import Iterable, Mapping
from functools import partial
from multiprocessing import Pool
from typing import Any

from tqdm.auto import tqdm
from mesa.model import Model

multiprocessing.set_start_method("spawn", force=True)


def sparse_batch_run(
    model_cls: type[Model],
    parameters: Mapping[str, Any | Iterable[Any]],
    number_processes: int | None = 1,
    iterations: int = 1,
    # data_collection_period is ignored here on purpose (model controls cadence)
    data_collection_period: int = -1,
    max_steps: int = 1000,
    display_progress: bool = True,
) -> list[dict[str, Any]]:
    """Batch run a mesa model where the model collects data sparsely.

    Args:
        model_cls: Model class to run.
        parameters: Dict of model kwargs (values may be singletons or iterables).
        number_processes: Processes to use (None = all CPUs). Defaults to 1.
        iterations: Repetitions per parameter combo. Defaults to 1.
        data_collection_period: Ignored. Kept only for signature compatibility.
        max_steps: Max steps to run. Defaults to 1000.
        display_progress: Show tqdm progress bar.

    Returns:
        List[Dict[str, Any]] where each dict is one collected row
        (if agent_reporters exist, one row per agent per collected step).
    """
    runs_list = []
    run_id = 0
    for iteration in range(iterations):
        for kwargs in _make_model_kwargs(parameters):
            runs_list.append((run_id, iteration, kwargs))
            run_id += 1

    process_func = partial(
        _model_run_func_sparse,
        model_cls,
        max_steps=max_steps,
        # keep the arg in the signature for parity; not used inside
        data_collection_period=data_collection_period,
    )

    results: list[dict[str, Any]] = []

    with tqdm(total=len(runs_list), disable=not display_progress) as pbar:
        if number_processes == 1:
            for run in runs_list:
                data = process_func(run)
                results.extend(data)
                pbar.update()
        else:
            with Pool(number_processes) as p:
                for data in p.imap_unordered(process_func, runs_list):
                    results.extend(data)
                    pbar.update()

    return results


def _make_model_kwargs(
    parameters: Mapping[str, Any | Iterable[Any]],
) -> list[dict[str, Any]]:
    """Create model kwargs from parameters dictionary (same behavior as Mesa)."""
    parameter_list = []
    for param, values in parameters.items():
        if isinstance(values, str):
            all_values = [(param, values)]
        elif isinstance(values, (list, tuple, set)) and len(values) == 0:
            raise ValueError(
                f"Parameter '{param}' contains an empty iterable, which is not allowed."
            )
        else:
            try:
                all_values = [(param, value) for value in values]
            except TypeError:
                all_values = [(param, values)]
        parameter_list.append(all_values)
    all_kwargs = itertools.product(*parameter_list)
    kwargs_list = [dict(kwargs) for kwargs in all_kwargs]
    return kwargs_list


def _model_run_func_sparse(
    model_cls: type[Model],
    run: tuple[int, int, dict[str, Any]],
    max_steps: int,
    data_collection_period: int,  # unused; kept for parity
) -> list[dict[str, Any]]:
    """Run a single model and collect *sparse* data based on what the model saved."""
    run_id, iteration, kwargs = run
    model = model_cls(**kwargs)

    # Run until max_steps (Mesa 3 increments model.steps itself)
    while model.running and model.steps <= max_steps:
        model.step()

    # Collect whatever the model actually saved
    return _collect_sparse(model, run_id, iteration, kwargs)


def _collect_sparse(
    model: Model,
    run_id: int,
    iteration: int,
    kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    """Collect model and agent data using the model's sparse DataCollector rows.

    REQUIREMENT: model.datacollector.model_reporters must include a "Step" reporter
    (e.g., 'Step': lambda m: m.steps) so we can map each saved row to the true step.
    """
    if not hasattr(model, "datacollector"):
        raise AttributeError(
            "The model does not have a datacollector attribute. Please add a DataCollector to your model."
        )
    dc = model.datacollector

    # Validate that "Step" exists in model_reporters so we can map rows to real steps
    if "Step" not in dc.model_vars:
        raise KeyError(
            "sparse_batch_run requires a 'Step' reporter in model_reporters "
            "(e.g., 'Step': lambda m: m.steps) to match model rows with agent rows."
        )

    # Model vars are lists aligned by *collection index*, not absolute step.
    # We iterate over collection index i, get the true step from model_vars["Step"][i],
    # grab that step's agent records, and emit rows.
    model_records: list[dict[str, Any]] = []
    # Length of any list in dc.model_vars (they should all be equal)
    num_rows = len(dc.model_vars["Step"])

    for i in range(num_rows):
        step_val = dc.model_vars["Step"][i]  # true step saved at this collection
        # Extract the i-th value for each model reporter
        model_data = {name: values[i] for name, values in dc.model_vars.items()}

        # Agent data saved under dc._agent_records keyed by true step
        all_agents_data = []
        raw_agent_data = dc._agent_records.get(step_val, [])
        for data in raw_agent_data:
            agent_dict = {"AgentID": data[1]}
            agent_dict.update(zip(dc.agent_reporters, data[2:]))
            all_agents_data.append(agent_dict)

        # Build rows (one per agent if agent_reporters exist; otherwise one row)
        if all_agents_data:
            rows = [
                {
                    "RunId": run_id,
                    "iteration": iteration,
                    "Step": step_val,
                    **kwargs,
                    **model_data,
                    **agent_data,
                }
                for agent_data in all_agents_data
            ]
        else:
            rows = [
                {
                    "RunId": run_id,
                    "iteration": iteration,
                    "Step": step_val,
                    **kwargs,
                    **model_data,
                }
            ]
        model_records.extend(rows)

    return model_records
