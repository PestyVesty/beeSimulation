import mesa
from agents import Bee, Plant, BeeBehavior, BeeHive, CompositeLevyBehavior
from mesa.datacollection import DataCollector
from shapely.geometry import LineString

class OpenCoverModel(mesa.Model):
    def __init__(self, n_agents = 60, seed = None, bee_behavior="composite_levy", landing_prob=1.0, memory_size=5):
        super().__init__(seed=seed)
        
        self.bee_behavior = bee_behavior
        self.landing_prob = landing_prob
        self.memory_size = memory_size
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
                             "Plant visit count" : "visit_count",
                             "Plant visit timestamp" : "visit_timestamp",
                             "BeeHive ID" : "beehive_id",
                             "Plant visited:" : "visited", # Boolean
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
        
        for _ in range(n_agents):
            if self.bee_behavior == "composite_levy":
                beeBehavior = CompositeLevyBehavior(landing_prob=self.landing_prob, memory_size=self.memory_size)
            else:
                beeBehavior = BeeBehavior(self.bee_behavior, landing_prob=self.landing_prob)
            bee = Bee(self, beeBehavior, beehive = beehive)
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
            
            plant1 = Plant(self)
            plant2 = Plant(self)
            plant3 = Plant(self)
            
            self.space.place_agent(plant1, coord1)
            self.space.place_agent(plant2, coord2)
            self.space.place_agent(plant3, coord3)