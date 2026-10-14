"""Mesa model for bee foraging in an open polytunnel."""

import mesa
import numpy as np
from mesa.datacollection import DataCollector
from shapely.geometry import LineString

from . import probability
from .agents import BeeHive, ComparativeRewards, MyBee, MyBeeBehavior, MyPlant, NearestFlower


BEHAVIOR_CLASSES = {
    "resource_aware": MyBeeBehavior,
    "comparative_rewards": ComparativeRewards,
    "nearest_flower": NearestFlower,
}


class OpenCoverModel(mesa.Model):
    """Simulate two bee colonies foraging in an open polytunnel."""

    def __init__(
        self,
        n_agents=60,
        seed=None,
        bee_behavior="resource_aware",
        landing_prob=1.0,
        replenish_rate=0.0,
        reward_size=5,
    ):
        super().__init__(rng=seed)

        if bee_behavior not in BEHAVIOR_CLASSES:
            supported = ", ".join(BEHAVIOR_CLASSES)
            raise ValueError(
                f"Unknown bee behavior {bee_behavior!r}. Supported behaviors: {supported}"
            )

        self.bee_behavior = bee_behavior
        self.landing_prob = landing_prob
        self.n_agents = n_agents
        self.replenish_rate = replenish_rate
        self.reward_size = reward_size

        self.space_width = 2.85 * 4
        self.space_length = 10 + 100 + 10
        self.space = mesa.space.ContinuousSpace(
            self.space_length, self.space_width, torus=False
        )

        self.walls = [
            LineString([(0, 0), (0, self.space_width)]),
            LineString([(0, self.space_width), (self.space_length, self.space_width)]),
            LineString(
                [
                    (self.space_length, self.space_width),
                    (self.space_length, 0),
                ]
            ),
            LineString([(self.space_length, 0), (0, 0)]),
        ]

        self.create_beehive_bees_agents(n_agents, (5, 2.85 * 2), my_id=0)
        self.create_beehive_bees_agents(
            n_agents, (15 + 100, 2.85 * 2), my_id=1
        )
        self.create_plant_agents()

        self.datacollector = DataCollector(
            # Mesa's batch runner uses model reports to index collected steps.
            model_reporters={"bee_behavior": "bee_behavior"},
            agent_reporters={
                "Position": "pos",
                "Agent Type": "agent_type",
                "unique_id": "unique_id",
                "Forage Timer": lambda agent: getattr(agent, "forage_timer", None),
                "Plot Code": lambda agent: agent.color_code(),
                "Plant visit timestamp": lambda agent: list(
                    getattr(agent, "visit_timestamp", [])
                ),
                "BeeHive ID": lambda agent: getattr(agent, "beehive_id", None),
                "Plant visited this step": lambda agent: getattr(
                    agent, "visited_this_step", False
                ),
                "Plant nectar amount": lambda agent: getattr(
                    agent, "nectar", None
                ),
                "Plant scent timer": lambda agent: getattr(
                    agent, "scent_mark_timer", None
                ),
                "Bee latest visited plant": self._latest_visited_positions,
                "Bee latest visited plant step": lambda agent: list(
                    getattr(agent, "latest_visited_flowers_steps", [])
                ),
            },
        )
        self.datacollector.collect(self)

    @staticmethod
    def _latest_visited_positions(agent):
        flowers = getattr(agent, "latest_visited_flowers", []) or []
        return [
            (float(plant.pos[0]), float(plant.pos[1]))
            for plant in flowers
            if getattr(plant, "pos", None) is not None
        ]

    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)

    def create_beehive_bees_agents(self, n_agents, coord, my_id):
        beehive = BeeHive(self, my_id)
        self.space.place_agent(beehive, coord)

        behavior_class = BEHAVIOR_CLASSES[self.bee_behavior]
        biased_distribution = probability.get_pdf_sampler("biased")

        for _ in range(n_agents):
            behavior = behavior_class(
                landing_prob=self.landing_prob,
                biased_distribution=biased_distribution,
            )
            bee = MyBee(
                self,
                behavior,
                beehive=beehive,
                reward_size=self.reward_size,
            )
            bee.theta = 0.0
            self.space.place_agent(bee, coord)

    def create_plant_agents(self):
        for x_coord in np.arange(10, 111, 1):
            for row in (1, 2, 3):
                plant = MyPlant(self, replenish_rate=self.replenish_rate)
                self.space.place_agent(plant, (x_coord, 2.85 * row))
                plant.on_placement()
