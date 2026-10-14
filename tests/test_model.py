import pytest
from mesa.batchrunner import batch_run

from bee_simulation.agents import MyBee, MyPlant
from bee_simulation.model import OpenCoverModel


def test_model_initializes_and_steps():
    model = OpenCoverModel(n_agents=1, seed=7, reward_size=3)

    bees = [agent for agent in model.agents if isinstance(agent, MyBee)]
    plants = [agent for agent in model.agents if isinstance(agent, MyPlant)]

    assert len(bees) == 2
    assert len(plants) == 303
    assert all(bee.reward_memory.maxlen == 3 for bee in bees)
    assert all(plant.sugar_concentration is not None for plant in plants)

    model.step()


def test_unknown_behavior_is_rejected():
    with pytest.raises(ValueError, match="Supported behaviors"):
        OpenCoverModel(n_agents=1, bee_behavior="unknown")


def test_batch_run_returns_collected_agent_rows():
    results = batch_run(
        OpenCoverModel,
        parameters={"n_agents": 0},
        rng=[7],
        max_steps=1,
        number_processes=1,
        data_collection_period=1,
        display_progress=False,
    )

    assert results
    assert {row["Step"] for row in results} == {0, 1}
    assert all("Agent Type" in row for row in results)
