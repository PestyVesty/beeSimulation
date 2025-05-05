import mesa
"""
Class for declaring bee and plant objects, and defining their behaviors

NOTE: use self.random to get a random seeded number
"""

class BeeBehavior():
    def __init__(self):
        self.bee = None

    def step(self):
        x, y = self.bee.pos
        new_pos = (x + 1, y + 1)
        self.bee.model.space.move_agent(self.bee, new_pos)
        
class Bee(mesa.Agent):
    def __init__(self, model, beebehavior : BeeBehavior, coord, theta):
        super().__init__(model)
        """
        Input:
        - model: the model instance of the environment
        - beebehavior: the behavior of the bee agent
        - coord: coordinates of the bee in the environment
        - theta: direction of the bee in degrees
        """
        self.agent_type = "bee"
        self.model = model
        self.behavior = beebehavior
        self.behavior.bee = self
        
        self.pos = coord
        self.theta = theta
        
        has_pollen = False
        pollen_timer = 0 
        pollen_timer_max = 25 # 25 steps of pollen carrying
        destination = None
    
    def step(self):
        """
        Bee behavior based on the declared bee behavior class
        """
        # Implement the behavior of the bee agent here
        self.behavior.step()
    
    def colorCode(self):
        """
        Color code for visualization of the bee
        """
        pass
"""
Plant behavior class
"""
class Plant(mesa.Agent):
    def __init__(self, model, gender, coord):
        """
        Input:
        - model: the model instance of the environment
        - gender: gender of the flower
        - coord: coordinates of the flower in the environment
        
        Additional attributes:
        - visited
        - gender
        - pollinated (female)
        - has_pollen (male)
        """
        super().__init__(model)
        self.agent_type = "plant"
        
        self.model = model
        self.gender = gender  # "male" or "female"
        self.pos = coord      # (x, y)
        self.visited = False  # Indication if flower has been visited before
        
        # Additional attributes based on the gender of the flower
        if self.gender == "female":
            self.pollinated = False
        elif self.gender == "male":
            self.has_pollen = True
        
    def step(self):
        """
        Plant behavior 
        """
        # Implement the behavior of the plant agent here
        pass
    
    
    def colorCode(self):
        """
        Color code for visualization of the flower
        
        Male:
        - Not visited (Has pollen): Blue
        - Visited (No pollen): Blue with white border line
        Female:
        - Not visited: Red
        - Visited (not pollinated): Red with white border lin
        - Visited (pollinated): Orage
        """
        if self.gender == "male":
            if not self.visited and self.has_pollen:
                return "blue"
            elif self.visited and not self.has_pollen:
                return "blue"  # TODO: add white border line
            else:
                raise "Male flower in state not considered or invalid"
        elif self.gender == "female":
            if not self.visited and not self.pollinated:
                return "red"
            elif self.visited and not self.pollinated:
                return "red" # TODO: add white border line
            elif self.visited and self.pollinated:
                return "orange"
            else:
                raise "Female flower in state not considered or invalid"
        
