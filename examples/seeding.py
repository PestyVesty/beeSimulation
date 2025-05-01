"""
Example of seeding a model and agents

Tip: after step 1, 2 and 3, whenever need a random number, just use self.random
"""

import mesa
class AwesomeModel(mesa.Model):
    def __init__(self, seed=None):                                              # 2. Add in attribute seed as optional in model 
        super().__init__(seed=seed)                                             # 3. Call super() with seed=seed to pass it to mesa.Model
        self.grid = mesa.space.MultiGrid(10, 10, True)
        a = AwesomeAgent(self)
        self.grid.place_agent(a, (1, 1))
        
        self.cool_method()
        
    def cool_method(self):
        interesting_number = self.random.random()                               # 4. Use self.random to get a random number seeded by the model's seed
        print(f"Model Number: {interesting_number}")

    def step(self):
        self.agents.shuffle_do("step")
        
class AwesomeAgent(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
    
    def step(self):
        print(f"Agent rand: {self.my_method()}")
    def my_method(self):
        random_number = self.random.randint(0, 100)                             # 5. Use self.random to get a random number seeded by the model's seed
        return random_number
    
if __name__ == "__main__":
    model = AwesomeModel(seed=0)                                                # 1. Pass in a seed to the model
    for i in range(3):
        model.step()
        