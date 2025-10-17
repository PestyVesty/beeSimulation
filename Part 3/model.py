import mesa
# from agents import Bee, Plant, PureLevyBehavior, TruncatedLevyBehavior, BeeHive, CompositeLevyBehavior, BiasedCompositeLevyBehavior, MyBeeBehavior, MyBee, MyPlant
from agents import Bee, Plant, BeeHive, MyBeeBehavior, MyBee, MyPlant, ComparativeRewards, NearestFlower, LevyFlight
from mesa.datacollection import DataCollector
from shapely.geometry import LineString
import numpy as np
import pdf_library

class OpenCoverModel(mesa.Model):
    def __init__(self, n_agents=60, seed=None, bee_behavior="resource_aware", landing_prob=1.0, replenish_rate=0.0, reward_size=5, interval=1):
        super().__init__(seed=seed)

        self.bee_behavior = bee_behavior
        self.landing_prob = landing_prob
        self.n_agents = n_agents
        self.replenish_rate = replenish_rate
        self.reward_size = reward_size

        self.interval = interval
        
        
        # Initialize space
        self.space_width = 2.85 * 4
        self.space_length = 10 + 100 + 10
        self.space = mesa.space.ContinuousSpace(self.space_length, self.space_width, False)

        # Define walls
        self.walls = [
            LineString([(0, 0), (0, self.space_width)]),
            LineString([(0, self.space_width), (self.space_length, self.space_width)]),
            LineString([(self.space_length, self.space_width), (self.space_length, 0)]),
            LineString([(self.space_length, 0), (0, 0)])
        ]

        # Create agents
        self.create_beehive_bees_agents(n_agents, (5, 2.85 * 2), my_id=0)
        self.create_beehive_bees_agents(n_agents, (15 + 100, 2.85 * 2), my_id=1)
        self.create_plant_agents()

        # Set up data collector
        self.datacollector = DataCollector(
            model_reporters={
                "Step": lambda m: m.steps,
                "Interval": lambda m: m.interval,
                "Behavior": lambda m: m.bee_behavior,
                "Replenish": lambda m: m.replenish_rate,
                "RewardSize": lambda m: m.reward_size,
                },
            agent_reporters={
                "Position": "pos",
                "Agent Type": "agent_type",
                "unique_id": "unique_id",
                "Forage Timer": "forage_timer",
                "Plot Code": lambda a: a.color_code(),
                "Plant visit timestamp": lambda a: list(getattr(a, "visit_timestamp", [])),
                "BeeHive ID": "beehive_id",
                "Plant visited this step": lambda a: getattr(a, "visited_this_step", False),
                "Plant nectar amount": lambda a: getattr(a, "nectar", None),
                "Plant scent timer": lambda a: getattr(a, "scent_mark_timer", None),
                "Bee latest visited plant": lambda a: [(float(p.pos[0]), float(p.pos[1])) for p in (getattr(a, "latest_visited_flowers", []) or []) if getattr(p, "pos", None) is not None],
                "Bee latest visited plant step": "latest_visited_flowers_steps",
                # "Reward window mean": lambda a: (np.mean(a.reward_memory)
                #                                  if hasattr(a, "reward_memory") and len(a.reward_memory) == a.reward_size
                #                                  else None),
                # "Long jump": lambda a: getattr(a, "is_long_jump", False),     # set where you choose the step
                # "Jump length": lambda a: getattr(a, "last_jump_length", None) # optional, for QC
            }
        )

        self.datacollector.collect(self)

    def step(self):
        self.agents.shuffle_do("step")


        if self.steps % self.interval == 0:
            self.datacollector.collect(self)

    def create_beehive_bees_agents(self, n_agents, coord, my_id):
        beehive = BeeHive(self, my_id)
        self.space.place_agent(beehive, coord)

        biased_distribution = pdf_library.get_pdf_sampler("biased")

        for _ in range(n_agents):
            # NEW: choose behavior based on self.bee_behavior
            if self.bee_behavior == "resource_aware":
                bee_behavior = MyBeeBehavior(
                    landing_prob=self.landing_prob,
                    biased_distribution=biased_distribution
                )
            elif self.bee_behavior == "comparative_rewards":
                bee_behavior = ComparativeRewards(
                    landing_prob=self.landing_prob,
                    biased_distribution=biased_distribution
                )
            elif self.bee_behavior == "nearest_flower":
                bee_behavior = NearestFlower(
                    landing_prob=self.landing_prob,
                    biased_distribution=biased_distribution
                )
            elif self.bee_behavior == "levy_flight":
                bee_behavior = LevyFlight(
                    landing_prob=self.landing_prob,
                    biased_distribution=biased_distribution
                )
            else:
                raise ValueError(f"Unknown or unsupported bee behavior: {self.bee_behavior}")

            bee = MyBee(self, bee_behavior, beehive=beehive, reward_size = self.reward_size)
            bee.theta = 0.0

            self.space.place_agent(bee, coord)

    def create_plant_agents(self):
        start_x = 10
        end_x = 110
        y = 2.85

        for x_coord in np.arange(start_x, end_x + 1, 1):  # NOTE: When changing the inter-section from 0.5 to 1, nectar amount deplenish at a faster rate, this is because having at 0.5 allows bees to row follow better, since the visual range is 0.7m.
            for row in [1, 2, 3]:
                plant = MyPlant(self, replenish_rate=self.replenish_rate)
                plant.on_placement()
                self.space.place_agent(plant, (x_coord, y * row))

                # Y-axis offset
                plant = MyPlant(self, replenish_rate=self.replenish_rate)
                plant.on_placement()
                self.space.place_agent(plant, (x_coord, y * row + 0.2))
                
                plant = MyPlant(self, replenish_rate=self.replenish_rate)
                plant.on_placement()
                self.space.place_agent(plant, (x_coord, y * row - 0.2))
                
                # X-axis offset
                plant = MyPlant(self, replenish_rate=self.replenish_rate)
                plant.on_placement()
                self.space.place_agent(plant, (x_coord + 0.2, y * row ))
                
                plant = MyPlant(self, replenish_rate=self.replenish_rate)
                plant.on_placement()
                self.space.place_agent(plant, (x_coord  - 0.2, y * row))



            
            
# ################################################################################ 
# ########################### Blank model ########################################
# ################################################################################

# class BlankModel(mesa.Model):
#     # def __init__(self, n_agents = 60, seed = None, bee_behavior="composite_levy", landing_prob=1.0, reward_size=5):
#     def __init__(self, n_agents = 60, seed = None, bee_behavior="composite_levy", landing_prob=1.0):
#         super().__init__(seed=seed)
        
#         self.bee_behavior = bee_behavior
#         self.landing_prob = landing_prob

#         self.n_agents = n_agents

#         # Initiate the environment space
#         self.space_width = 100
#         self.space_length = 100
        
#         self.space = mesa.space.ContinuousSpace(self.space_length, self.space_width, False)  # Allowing agents to wrap around the space, however this just allows bees to be on the edge of the environment
        
#         self.walls = [LineString([(0, 0), (0, self.space_width)]), 
#             LineString([(0, self.space_width), (self.space_length, self.space_width)]),
#             LineString([(self.space_length, self.space_width), (self.space_length, 0)]),
#             LineString([(self.space_length, 0), (0, 0)])]
        
#         # Create beehive agents within the environment and bees at the beehives
#         self.create_beehive_bees_agents(n_agents, (self.space_width / 2, self.space_length / 2), my_id = 0)
        
#         # Create plant agents within the enviornment
#         # self.create_plant_agents()

#         # Collecting the location of the agents
#         #! TODO: Decide what to collect
#         self.datacollector = DataCollector(
#             model_reporters={},  # Collecting the positions of the agents
#             agent_reporters={"Bee Positions" : "pos", # TODO: Change this from Bee Positions to Positions
#                              "Agent Type"    : "agent_type",
#                              "unique_id"     : "unique_id",
#                              "Forage Timer" : "forage_timer",
#                              "Plot Code"     : lambda a: a.color_code(),
#                              "Plant visit count" : "visit_count", # Cumalative count of visits to each plant
#                              "Plant visit timestamp" : "visit_timestamp",
#                              "BeeHive ID" : "beehive_id",
#                              "Plant visited:" : "visited", # Boolean
#                              "Plant visited this step": lambda a: getattr(a, "visited_this_step", False),
#                              "Plant nectar amount" : lambda a: getattr(a, "nectar", None), # Amount of nectar at the plant
#                              "Plant scent timer": lambda a: getattr(a, "scent_mark_timer", None), # Scent timer of the plant
#                             }   
#         )
        
#         self.datacollector.collect(self)
        
    
#     def step(self):
#         self.agents.shuffle_do("step")
#         self.datacollector.collect(self)
        
    
#     def create_beehive_bees_agents(self, n_agents, coord, my_id):
#         """
#         n_agents: number of bees to create at the beehive
#         coord: coordinates of the beehive in the environment
#         """
#         beehive = BeeHive(self, my_id)
#         self.space.place_agent(beehive, coord)
#         biased_distribution = None
#         # Prepare biased distribution for biased_composite (outside bee loop!)
#         if self.bee_behavior in ["biased_composite", "resource_aware"]:
#             x_discrete = np.array([-180, -135, -90, -45, 0, 45, 90, 135, 180])
#             y_discrete = np.array([0.06, 0.07, 0.13, 0.17, 0.20, 0.17, 0.13, 0.07, 0.06])
#             x_fine = np.linspace(-180, 180, 1000)
#             pdf_fine = np.interp(x_fine, x_discrete, y_discrete)
#             dx_pdf = x_fine[1] - x_fine[0]
#             pdf_fine /= np.sum(pdf_fine * dx_pdf)
#             cdf_pdf = np.cumsum(pdf_fine) * dx_pdf
#             cdf_pdf /= cdf_pdf[-1]

#             def biased_distribution(bee_random, n=1):
#                 return [np.interp(bee_random.random(), cdf_pdf, x_fine) for _ in range(n)]

#         # Create bees
#         for _ in range(n_agents):
#             if self.bee_behavior == "composite_levy":
#                 beeBehavior = CompositeLevyBehavior(
#                     landing_prob=self.landing_prob,
#                     # reward_size=self.reward_size
#                 )
#             elif self.bee_behavior == "biased_composite":
#                 beeBehavior = BiasedCompositeLevyBehavior(
#                     landing_prob=self.landing_prob,
#                     # reward_size=self.reward_size,
#                     biased_distribution=biased_distribution
#                 )
#             elif self.bee_behavior == "pure_levy":
#                 beeBehavior = PureLevyBehavior(landing_prob=self.landing_prob)
#             elif self.bee_behavior == "truncated_levy":
#                 beeBehavior = TruncatedLevyBehavior(landing_prob=self.landing_prob)
#             elif self.bee_behavior == "resource_aware":
#                 beeBehavior = MyBeeBehavior(
#                     landing_prob=self.landing_prob,
#                     # reward_size=self.reward_size,
#                     biased_distribution=biased_distribution
#                 )
#             else:
#                 raise ValueError(f"Unknown bee behavior: {self.bee_behavior}")

#             # Create the bee agent
#             if self.bee_behavior == "resource_aware":
#                 bee = MyBee(self, beeBehavior, beehive=beehive)
#             else:
#                 bee = Bee(self, beeBehavior, beehive=beehive)
#             bee.theta = 0.0  # Initialize bee facing direction

#             self.space.place_agent(bee, coord)