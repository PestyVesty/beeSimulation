"""
This module replicates the environment of the greenhouse from the research paper:
Simulation-governed design and tuning of greenhouses for successful bee pollination.
"""

import mesa
from agents import Bee, Plant, BeeBehavior
from mesa.datacollection import DataCollector
from shapely.geometry import LineString

class GreenhouseModel(mesa.Model):
    def __init__(self, n_agents = 60, seed = None):
        super().__init__(seed=seed)
        
        # Initiate the environment space
        self.space_width = 18 * 0.7 # 18 cells * 0.7 m = 12.6 m
        self.space_length = 72 * 0.7 # 72 cells * 0.7 m = 50.4 m
        self.space = mesa.space.ContinuousSpace(self.space_length, self.space_width, True) # Allowing agents to wrap around the edges
        
        # Declaring walls        
        self.walls = [LineString([(0, 0), (0, self.space_width)]), 
            LineString([(0, self.space_width), (self.space_length, self.space_width)]),
            LineString([(self.space_length, self.space_width), (self.space_length, 0)]),
            LineString([(self.space_length, 0), (0, 0)])]
        
        # Create plant agents within the environment
        self.create_plant_agents()

        # # Create bee agents within the environment
        self.create_bee_agents(n_agents)
        
        # Collecting the locations of the agents
        #! Add additional data collections as needed here
        self.datacollector = DataCollector(
            model_reporters ={},  # Collecting the positions of the agents
            agent_reporters={"Bee Positions" : "pos",
                             "Agent Type"    : "agent_type",
                             "unique_id"     : "unique_id",
                             "Theta"         : "theta",
                             "Plot Code"     : lambda a: a.color_code()}   
        )
        

        # from pdb import set_trace as st
        # st()
        self.datacollector.collect(self)
    
    def step(self):
        self.agents.shuffle_do("step")
        self.datacollector.collect(self)
        
    def create_plant_agents(self, layout = "MFMFMFMF"):
        """
        Initialize the plant agents in the environment based on the input layout.
        
        By default, the flowers are arranged in alternating gender: MFMFMFMF.
        """
        assert len(layout) == 8, "Layout must be 8 characters long"
        x = 6 * 0.7 # 6 cells * 0.7 m = 4.2 m
        ys = [2, 3, 6, 7, 10, 11, 14, 15] # y-coordinates for the beds
        
        
        for i, row_type in enumerate(layout):
            y = ys[i] * 0.7
            gender = self.get_plant_row_gender(row_type)
            
            for j in range(60):
                coord = (x + j * 0.7, y)
                plant = Plant(self, gender=gender, coord=coord)                         # NOTE: this creates the plant and places it in the space
                self.space.place_agent(plant, coord)                                   # NOTE: this places the plant in the space
        
    def create_bee_agents(self, n_agents):
        """
        Initialize the bee agents in the model.
        """

        for _ in range(n_agents):
            x = self.random.uniform(0, self.space_length)
            y = self.random.uniform(0, self.space_width)
            coord = (x, y)
            
            theta = self.random.uniform(0, 360)
            
            beebehavior = BeeBehavior()                                                      
            bee = Bee(self, beebehavior, coord=coord, theta=theta)
            self.space.place_agent(bee, coord) 
            
    def get_plant_row_gender(self, char):
        return "male" if char == "M" else "female"