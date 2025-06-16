import mesa
from agents import Bee, Plant, PureLevyBehavior, TruncatedLevyBehavior, BeeHive, CompositeLevyBehavior, BiasedCompositeLevyBehavior, MyBeeBehavior, MyBee, MyPlant
from mesa.datacollection import DataCollector
from shapely.geometry import LineString
import numpy as np

class OpenCoverModel(mesa.Model):
    def __init__(self, n_agents = 60, seed = None, bee_behavior="composite_levy", landing_prob=1.0, reward_size=5):
        super().__init__(seed=seed)
        
        self.bee_behavior = bee_behavior
        self.landing_prob = landing_prob
        self.reward_size = reward_size
        self.n_agents = n_agents

        # Initiate the environment space
        self.space_width = 2.85 * 4
        self.space_length = 10 + 100 + 10
        
        self.space = mesa.space.ContinuousSpace(self.space_length, self.space_width, False)  # Allowing agents to wrap around the space, however this just allows bees to be on the edge of the environment
        
        self.walls = [LineString([(0, 0), (0, self.space_width)]), 
            LineString([(0, self.space_width), (self.space_length, self.space_width)]),
            LineString([(self.space_length, self.space_width), (self.space_length, 0)]),
            LineString([(self.space_length, 0), (0, 0)])]
        
        # Create beehive agents within the environment and bees at the beehives
        self.create_beehive_bees_agents(n_agents, (5, 2.85 * 2), my_id = 0)
        self.create_beehive_bees_agents(n_agents, (15 + 100, 2.85 * 2), my_id=1)
        
        # Create plant agents within the enviornment
        self.create_plant_agents()

        # Collecting the location of the agents
        #! TODO: Decide what to collect
        self.datacollector = DataCollector(
            model_reporters={},  # Collecting the positions of the agents
            agent_reporters={"Bee Positions" : "pos",
                             "Agent Type"    : "agent_type",
                             "unique_id"     : "unique_id",
                             "Plot Code"     : lambda a: a.color_code(),
                             "Plant visit count" : "visit_count", # Cumalative count of visits to each plant
                             "Plant visit timestamp" : "visit_timestamp",
                             "BeeHive ID" : "beehive_id",
                             "Plant visited:" : "visited", # Boolean
                             "Plant visited this step": lambda a: getattr(a, "visited_this_step", False),
                            }   
        )
        
        self.datacollector.collect(self)
        
    
    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)
        
    
    def create_beehive_bees_agents(self, n_agents, coord, my_id):
        """
        n_agents: number of bees to create at the beehive
        coord: coordinates of the beehive in the environment
        """
        beehive = BeeHive(self, my_id)
        self.space.place_agent(beehive, coord)
        biased_distribution = None
        # Prepare biased distribution for biased_composite (outside bee loop!)
        if self.bee_behavior in ["biased_composite", "my_behavior"]:
            x_discrete = np.array([-180, -135, -90, -45, 0, 45, 90, 135, 180])
            y_discrete = np.array([0.06, 0.07, 0.13, 0.17, 0.20, 0.17, 0.13, 0.07, 0.06])
            x_fine = np.linspace(-180, 180, 1000)
            pdf_fine = np.interp(x_fine, x_discrete, y_discrete)
            dx_pdf = x_fine[1] - x_fine[0]
            pdf_fine /= np.sum(pdf_fine * dx_pdf)
            cdf_pdf = np.cumsum(pdf_fine) * dx_pdf
            cdf_pdf /= cdf_pdf[-1]

            def biased_distribution(bee_random, n=1):
                return [np.interp(bee_random.random(), cdf_pdf, x_fine) for _ in range(n)]

        # Create bees
        for _ in range(n_agents):
            if self.bee_behavior == "composite_levy":
                beeBehavior = CompositeLevyBehavior(
                    landing_prob=self.landing_prob,
                    reward_size=self.reward_size
                )
            elif self.bee_behavior == "biased_composite":
                beeBehavior = BiasedCompositeLevyBehavior(
                    landing_prob=self.landing_prob,
                    reward_size=self.reward_size,
                    biased_distribution=biased_distribution
                )
            elif self.bee_behavior == "pure_levy":
                beeBehavior = PureLevyBehavior(landing_prob=self.landing_prob)
            elif self.bee_behavior == "truncated_levy":
                beeBehavior = TruncatedLevyBehavior(landing_prob=self.landing_prob)
            elif self.bee_behavior == "my_behavior":
                beeBehavior = MyBeeBehavior(
                    landing_prob=self.landing_prob,
                    reward_size=self.reward_size,
                    biased_distribution=biased_distribution
                )
            else:
                raise ValueError(f"Unknown bee behavior: {self.bee_behavior}")

            # Create the bee agent
            if self.bee_behavior == "my_behavior":
                bee = MyBee(self, beeBehavior, beehive=beehive)
            else:
                bee = Bee(self, beeBehavior, beehive=beehive)
            bee.theta = 0.0  # Initialize bee facing direction

            self.space.place_agent(bee, coord)

                        
    def create_plant_agents(self):
        """
        Plant locations:
            (10, 2.85 * 3), (11, 2.85 * 3), ..., (110, 2.85 * 3),
            (10, 2.85 * 2), (11, 2.85 * 2), ..., (110, 2.85 * 2),
            (10, 2.85 * 1), (11, 2.85 * 1), ..., (110, 2.85 * 3)
        """
        start_x = 10
        end_x = 110
        y = 2.85
        
        for x_coord in [x for x in range(start_x, end_x + 1)]:
            coord1 = (x_coord, y * 1)
            coord2 = (x_coord, y * 2)
            coord3 = (x_coord, y * 3)
            
            if self.bee_behavior == "my_behavior":
                plant1 = MyPlant(self)
                plant2 = MyPlant(self)
                plant3 = MyPlant(self)
            else:
                plant1 = Plant(self)
                plant2 = Plant(self)
                plant3 = Plant(self)
            
            self.space.place_agent(plant1, coord1)
            self.space.place_agent(plant2, coord2)
            self.space.place_agent(plant3, coord3)