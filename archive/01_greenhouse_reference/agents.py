import mesa
import math
from shapely.geometry import LineString, Point
import numpy as np

"""
Class for declaring bee and plant objects, and defining their behaviors

NOTE: use self.random to get a random seeded number
"""


class BeeBehavior():
    def __init__(self):
        self.bee = None
        self.max_cauchy_length = 5 # Maximum length of the Cauchy distribution for bee movement

    def step(self):
        """
        Bee movement and Decision making replicating the behavior of the bee agent behavior of
        research paper "Simulation-governed design and tuning of greenhouses for successful bee pollination":
            1. If a bee sees an unvisited flower, it sets it as the target and moves towards it
                - spends 3 time steps on a flower
            2. Otherwise, fly to new destination based on:
                - random direction (uniformally drawn)  #! Testing: biased direction based the direction of the bee
                - random length (Cauchy-distributed)
        """
        # Busy foraging?
        # 1. No, is bee at destination? 
        if self.bee.forage_timer == 0:
            # 2. Have a set destination?
            # 2. No, bee is at destination NOTE: check if bee is at destination by looking whether route is empty or not
            if self.bee.travel_route == []:
                # 8. Any unvisited flower in the environment?
                self.get_new_directions()
                
            # 2. Yes, bee is not at destination
            else:
                # 3. Move one cell towards destination
                next_coord = self.bee.travel_route.pop(0) # Remove the first element of the route
                self.bee.model.space.move_agent(self.bee, next_coord)
                
                # 4. Do not need to check if bee is at destination. If destination is out of bounds, bee will stay on the wall
                
                # 5. If bee is at destination?
                # 5. Yes, check if there is any flower below it
                if self.bee.travel_route == []:
                    # 6. Is there a flower below it? 
                    below = self.bee.model.space.get_neighbors(self.bee.pos, 0, include_center=True)
                    plant_agents = [agent for agent in below if isinstance(agent, Plant)]
                    

                    available_plants = [plant for plant in plant_agents if not plant.visited and plant.bee_on is None]
                    
                    # 6. Yes, and the flower is unvisited
                    if available_plants:
                        # 7. Start foraging
                        self.bee.forage_timer = self.bee.forage_timer_max
                        plant = available_plants[0]
                        self.bee.on_plant = plant
                        plant.bee_on = self.bee
                        # self.bee.on_plant.visited = True     # Mark the flower as visited as soon as the bee lands on it, so that no other bee can visit it at the same time
                    
                    # 6. No, the flower is visited
                    else:
                        # 8. Any unvisited flower in the environment?
                        self.get_new_directions()
                
                
        # 1. Yes, bee is at destination, decrement
        else:

            self.bee.forage_timer -= 1
            if self.bee.forage_timer == 0:
                # Set pollen timer to max if the bee foraged from a male flower
                if self.bee.on_plant.gender == "male":
                    self.bee.pollen_timer = self.bee.pollen_timer_max
                    self.bee.has_pollen = True
                    self.bee.on_plant.has_pollen = False # Pollen is removed from the flower
                    
                # If the landed plant is female and the bee has pollen, pollinate the flower
                if self.bee.on_plant.gender == "female":
                    if self.bee.has_pollen:
                        self.bee.on_plant.pollinated = True # Pollen remains on the flower
                
                self.bee.on_plant.visited = True
                self.bee.on_plant.bee_on = None # Bee leaves the flower                
                self.bee.on_plant = None        # The flower has no bee on it anymore
                
        # Decrement pollen timer if greater than 0
        if self.bee.pollen_timer > 0:
            self.bee.pollen_timer -= 1
            if self.bee.pollen_timer == 0:
                self.bee.has_pollen = False
            
            
    def get_new_directions(self):
        """
        With respect to Figure 2 in the research paper, this function starts at the
        cell "Are any nearby flowers unvisited targets?"
        """
        # 8. Are any nearby flowers unvisited targets?
        neighbors = self.bee.model.space.get_neighbors(pos = self.bee.pos, radius=self.bee.vision_range, include_center=True)
        plant_agents = [agent for agent in neighbors if isinstance(agent, Plant)]
        unvisited_plants = [plant for plant in plant_agents if not plant.visited and plant.bee_on is None]
        
        # 8 -No-> 9, Pick new destination cell
        if unvisited_plants == []:
            # 9. Pick new destination
            self.bee.travel_route = self.cauchy_sample()
        
        # 8 -Yes-> 10, Pick one and set as new destination
        else:
            random_plant = self.bee.random.choice(unvisited_plants)
            self.bee.travel_route = [random_plant.pos]   
    
    def cauchy_sample(self):
        """
        With respect to Figure 2 in the research paper, this fuction corresponds
        to the cell "Pick new destination cell"
        
        Sampling the Cauchy distribution:
        1. Sample the direction
        2. Sample the length of the movement and determine new position coordinates
        3. Check if new position intersects with wall
            - If it does, set the new position to the wall position
        4. From the current to the new position, divide it up into 0.7 m steps
        5. Move towards the next position
        """
        
        while True:
            # 1. Sample the direction
            theta = self.bee.random.uniform(0, 360)
            
            # 2. Sample length of movement
            distance = self.cauchy_pdf(x=0)

            # Compute new coordinates
            destination_coord = self.compute_endpoint(self.bee.pos[0], self.bee.pos[1], distance, theta)
            start_point = Point(self.bee.pos)
            bee_path = LineString([self.bee.pos, destination_coord])
            
            # 3. Check if new position intersects with wall
            if self.bee.model.space.out_of_bounds(pos=destination_coord):
            
                max_distance = -1
                furthest_intersection = None
                for wall in self.bee.model.walls:
                    if bee_path.intersects(wall):
                        intersection = bee_path.intersection(wall)

                        # Could be a point or a line (e.g. overlapping)
                        if intersection.geom_type == "Point":
                            dist = start_point.distance(intersection)
                            if dist > max_distance:
                                max_distance = dist
                                furthest_intersection = intersection
                        elif intersection.geom_type == "MultiPoint":
                            for pt in intersection.geoms:
                                dist = start_point.distance(pt)
                                if dist > max_distance:
                                    max_distance = dist
                                    furthest_intersection = pt
                        # Handle line overlap (optional)
                        elif intersection.geom_type == "LineString":
                            # Use the farthest endpoint of the overlap
                            coords = list(intersection.coords)
                            for pt_coords in [coords[0], coords[-1]]:
                                pt = Point(pt_coords)
                                dist = start_point.distance(pt)
                                if dist > max_distance:
                                    max_distance = dist
                                    furthest_intersection = pt

                if furthest_intersection is not None:
                    bee_path = LineString([self.bee.pos, furthest_intersection])

            return self.sample_line(bee_path, 0.7)
    
    def sample_line(self, line: LineString, step: float) -> list:
        """
        Returns a list of coordinates spaced `step` meters apart along the LineString,
        including the end coordinate even if the last step is shorter.
        """
        if line.length == 0:
            return [line.coords[0]]  # Just a single point
        
        distances = np.arange(0, line.length, step).tolist()

        if distances[-1] < line.length:
            distances.append(line.length)  # Ensure final point is included

        points = [line.interpolate(distance) for distance in distances]
        return [tuple(point.coords[0]) for point in points]
        
    def cauchy_pdf(self, x, x0=0, gamma=1):
        numerator = 1
        denominator = math.pi * gamma * (1 + ((x - x0) / gamma) ** 2)
        return numerator / denominator
    
    def compute_endpoint(self, x0, y0, distance, theta):
        """
        Given current coordinates, distance and angle, compute the new coordinates
        """
        angle_rad = math.radians(theta)
        x1 = x0 + distance * math.cos(angle_rad)
        y1 = y0 + distance * math.sin(angle_rad)
        return x1, y1
    
class Bee(mesa.Agent):
    def __init__(self, model, beebehavior : BeeBehavior, theta):
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
        self.walls = self.model.walls
        self.space = self.model.space
        
        self.pos = None
        
        self.theta = theta
        self.vision_range = 0.7
        
        # Foraging and pollen carrying parameters (from paper)
        self.forage_timer = 0
        self.forage_timer_max = 3  # 3 steps for foraging
        self.has_pollen = False
        self.pollen_timer = 0 
        self.pollen_timer_max = 25 # 25 steps for pollen carrying
        # self.destination = None

        # Parameters for traveling
        self.travel_route = []
        self.on_plant = None   
        
    def step(self):
        """
        Bee behavior based on the declared bee behavior class
        """
        # Implement the behavior of the bee agent here
        self.behavior.step()
        
    
    
    def color_code(self):
        """
        Color code for visualization of the bee
        """
        if self.has_pollen:
            return {"color" : "white", "edgecolors" : "black"}
        else:
            return {"color" : "black", "edgecolors" : "none"}
    
"""
Plant behavior class
"""
class Plant(mesa.Agent):
    def __init__(self, model, gender):
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
        self.pos = None      # (x, y)
        self.bee_on = None    # Bee on the flower
        self.visited = False  # Indication if flower has been visited before (flower scent)
        
        # Additional attributes based on the gender of the flower
        if self.gender == "female":
            self.pollinated = False
        elif self.gender == "male":
            self.has_pollen = True
        
    def step(self):
        """
        Plant behavior
        
        However, the plants from the research paper does not exhibit any active
        behavior. They are passive agents and only react to the bee agents. 
        """
        pass
    
    
    def color_code(self):
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
                return {"color" : "blue", "edgecolors" : "none"}
            elif self.visited and not self.has_pollen:
                return {"color" : "blue", "edgecolors" : "white"}  # White border line
            else:
                raise "Male flower in state not considered or invalid" 
        elif self.gender == "female":
            if not self.visited and not self.pollinated:
                return {"color" : "red", "edgecolors" : "none"}
            elif self.visited and not self.pollinated:
                return {"color" : "red", "edgecolors" : "white"}
            elif self.visited and self.pollinated:
                return {"color" : "orange", "edgecolors" : "none"}
            else:
                raise "Female flower in state not considered or invalid"
        

class Wall(mesa.Agent):
    def __init__(self, model):
        """
        Input:
        - model: the model instance of the environment
        
        NOTE: Currently not in use, the wall is defined in the model class under
        the attribute "walls". The wall is a shapely LineString object.
        """
        super().__init__(model)
        self.agent_type = "wall"
        
        self.model = model
        
    def step(self):
        """
        Wall behavior
        
    
        """
        pass
    
    
    def color_code(self):
        """
        Color code for visualization of the wall
        
        The wall is represented as a black square.
        """
        return {"color" : "black", "edgecolors" : "none"}