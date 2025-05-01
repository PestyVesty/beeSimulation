"""
This module simulate the environment of the greenhouse from the research paper:
Simulation-governed design and tuning of greenhouses for successful bee pollination.
"""

import mesa

class GreenhouseModel(mesa.Model):
    def __init__(self, n_agents = 60, seed = None):
        super().__init__(seed=seed)
        
        self.space_width = 100
        self.space_height = 200
        self.space = mesa.space.ContinuousSpace(self.space_width, self.space_height, True) # Allowing agents to wrap around the edges
        
        # Declare bee agents
        self.bees_init(self, n_agents)
        
        # Declare plant agents
        self.plant_layout_init(self)
        
    def step(self):
        self.agents.shuffle_do("step")
    
    def bees_init(self, model, n_agents):
        """
        Initialize the bee agents in the model.
        """
        for _ in range(n_agents):
            pass
        
        
    def plant_layout_init(self, model):
        """
        Initialize the plant agents in the model.
        """
        pass