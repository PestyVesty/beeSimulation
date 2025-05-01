import mesa
from agents import Beebehavior, PlantBehavior

"""
Bee behavior class

NOTE: use self.random to get a random seeded number
"""

class BeeBehavior():
    def __init__(self):
        pass

class Bee(mesa.Agent):
    def __init__(self, model, beebehavior : BeeBehavior):
        super().__init__(model)
        self.behavior = beebehavior
    


"""
Plant behavior class
"""
class PlantBehavior(PlantBehavior):
    def __init__(self, model):
        super().__init__(model)