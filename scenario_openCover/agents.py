import mesa
import math
from shapely.geometry import LineString, Point
import numpy as np

"""
Class for declaring bee and plant objects, and defining their behaviors

NOTE: use self.random to get a random seeded number
"""

class BeeHive(mesa.Agent):
    def __init__(self, model, my_id):
        super().__init__(model)
        self.agent_type = "beehive"
        self.model = model
        self.pos = None
        self.my_id = my_id
    
    def color_code(self):
        """
        Color code for visualization of the bee hive
        """
        return {"color" : "brown", "edgecolors" : "none"}


class BeeBehavior():
    def __init__(self, bee_behavior, landing_prob):
        self.bee = None
        # self.max_cauchy_length = 5 # Maximum length of the Cauchy distribution for bee movement
        self.behavior_type = bee_behavior
        self.landing_prob = landing_prob
        self.LONG_JUMP_THRESHOLD = 3.0

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
                # self.bee.model.space.move_agent(self.bee, next_coord)
                next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                self.bee.model.space.move_agent(self.bee, next_coord)
                # 4. Do not need to check if bee is at destination. If destination is out of bounds, bee will stay on the wall
                
                # 5. If bee is at destination?
                # 5. Yes, check if there is any flower below it
                if self.bee.travel_route == []:
                    if self.bee.is_long_jump:
                        self.get_new_directions()
                        self.bee.is_long_jump = False
                        return
                    # 6. Is there a flower below it? 
                    below = self.bee.model.space.get_neighbors(self.bee.pos, 0, include_center=True)
                    plant_agents = [agent for agent in below if isinstance(agent, Plant)]
                    
                    available_plants = [plant for plant in plant_agents if plant.bee_on is None and plant != self.bee.last_plant]
                    
                    # 6. Yes, and the flower is unvisited
                    if available_plants and self.land_on_flower(self.bee): #! sample whether to land on flower                        
                        
                        # 7. Start foraging
                        plant = available_plants[0] # Assuming there is only one flower below available_plants

                        self.bee.forage_timer = self.bee.forage_timer_max
                        self.bee.on_plant = plant
                        plant.bee_on = self.bee
                        
                        #! Update plant visit count and timestamp
                        plant.visit_count += 1
                        plant.visit_timestamp.append(self.bee.model.steps)
                        plant.visited = True
                    # 6. No, did not land on flower 
                    else:
                        # 8. Any unvisited flower in the environment?
                        self.get_new_directions()
                
                
        # 1. Yes, bee is at destination, decrement
        else:

            self.bee.forage_timer -= 1
            if self.bee.forage_timer == 0:
                self.bee.last_plant = self.bee.on_plant
                self.bee.on_plant.bee_on = None # Bee leaves the flower                
                self.bee.on_plant = None        # The flower has no bee on it anymore
                

    def clamp_position(self, pos, width, height):
        x = min(max(0, pos[0]), width - 0.001)
        y = min(max(0, pos[1]), height - 0.001)
        return (x, y)      
     
    def land_on_flower(self, bee):
        """
        Sample whether to land on flower or not
        """
        # bee_pos = bee.pos
        # beehive_pos = bee.beehive.pos
        # alpha = 663
        
        # distance = math.sqrt((bee_pos[0] - beehive_pos[0]) ** 2 + (bee_pos[1] - beehive_pos[1]) ** 2)
        # probability = math.exp(-distance / alpha)
        return bee.random.random() < self.landing_prob
    
    def get_new_directions(self):
        """
        With respect to Figure 2 in the research paper, this function starts at the
        cell "Are any nearby flowers avaliable targets?"
        """
        # 8. Are any nearby flowers avaliable targets?
        neighbors = self.bee.model.space.get_neighbors(pos = self.bee.pos, radius=self.bee.vision_range, include_center=True)
        plant_agents = [agent for agent in neighbors if isinstance(agent, Plant)]
        available_plants = [plant for plant in plant_agents if plant.bee_on is None and plant != self.bee.last_plant]
        
        # 8 -No-> 9, Pick new destination cell
        if available_plants == []:
            # 9. Pick new destination
            # self.bee.travel_route = self.cauchy_sample()
            if self.behavior_type == "pure_levy":
                
                self.bee.travel_route, self.bee.is_long_jump = self.levy_pure_sample()
            elif self.behavior_type == "truncated_levy":
                self.bee.travel_route, self.bee.is_long_jump = self.levy_truncated_sample()
            else:
                raise ValueError("Unknown bee behavior type. Use 'pure_levy' or 'truncated_levy'.")
        
        # 8 -Yes-> 10, Pick one and set as new destination
        else:
            random_plant = self.bee.random.choice(available_plants)
            self.bee.travel_route = [random_plant.pos]   
    
    # def cauchy_sample(self):
    #     """
    #     With respect to Figure 2 in the research paper, this fuction corresponds
    #     to the cell "Pick new destination cell"
        
    #     Sampling the Cauchy distribution:
    #     1. Sample the direction
    #     2. Sample the length of the movement and determine new position coordinates
    #     3. Check if new position intersects with wall
    #         - If it does, set the new position to the wall position
    #     4. From the current to the new position, divide it up into 0.7 m steps
    #     5. Move towards the next position
    #     """
        
    #     while True:
    #         # 1. Sample the direction
    #         theta = self.bee.random.uniform(0, 360)
            
    #         # 2. Sample length of movement
    #         distance = self.cauchy_pdf(x=0)

    #         # Compute new coordinates
    #         destination_coord = self.compute_endpoint(self.bee.pos[0], self.bee.pos[1], distance, theta)
    #         start_point = Point(self.bee.pos)
    #         bee_path = LineString([self.bee.pos, destination_coord])
            
    #         # 3. Check if new position intersects with wall
    #         if self.bee.model.space.out_of_bounds(pos=destination_coord):
            
    #             max_distance = -1
    #             furthest_intersection = None
    #             for wall in self.bee.model.walls:
    #                 if bee_path.intersects(wall):
    #                     intersection = bee_path.intersection(wall)

    #                     # Could be a point or a line (e.g. overlapping)
    #                     if intersection.geom_type == "Point":
    #                         dist = start_point.distance(intersection)
    #                         if dist > max_distance:
    #                             max_distance = dist
    #                             furthest_intersection = intersection
    #                     elif intersection.geom_type == "MultiPoint":
    #                         for pt in intersection.geoms:
    #                             dist = start_point.distance(pt)
    #                             if dist > max_distance:
    #                                 max_distance = dist
    #                                 furthest_intersection = pt
    #                     # Handle line overlap (optional)
    #                     elif intersection.geom_type == "LineString":
    #                         # Use the farthest endpoint of the overlap
    #                         coords = list(intersection.coords)
    #                         for pt_coords in [coords[0], coords[-1]]:
    #                             pt = Point(pt_coords)
    #                             dist = start_point.distance(pt)
    #                             if dist > max_distance:
    #                                 max_distance = dist
    #                                 furthest_intersection = pt

    #             if furthest_intersection is not None:
    #                 bee_path = LineString([self.bee.pos, furthest_intersection])

    #         return self.sample_line(bee_path, 0.7)
    def levy_pure_sample(self, mu=2.0, l_min=0.7):

        while True:
            # 1. Sample direction
            theta = self.bee.random.uniform(0, 360)

            # 2. Sample step length from pure Lévy
            U = self.bee.random.uniform(0, 1)
            distance = l_min * (1 - U) ** (-1 / mu)

            is_long_jump = distance > self.LONG_JUMP_THRESHOLD
            
            # 3. Compute destination point
            destination_coord = self.compute_endpoint(self.bee.pos[0], self.bee.pos[1], distance, theta)
            start_point = Point(self.bee.pos)
            bee_path = LineString([self.bee.pos, destination_coord])

            # 4. Handle wall collisions
            if self.bee.model.space.out_of_bounds(pos=destination_coord):
                max_distance = -1
                furthest_intersection = None
                for wall in self.bee.model.walls:
                    if bee_path.intersects(wall):
                        intersection = bee_path.intersection(wall)
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
                        elif intersection.geom_type == "LineString":
                            for pt_coords in [list(intersection.coords)[0], list(intersection.coords)[-1]]:
                                pt = Point(pt_coords)
                                dist = start_point.distance(pt)
                                if dist > max_distance:
                                    max_distance = dist
                                    furthest_intersection = pt
                if furthest_intersection:
                    bee_path = LineString([self.bee.pos, furthest_intersection])

            # 5. Sample along the line
            return self.sample_line(bee_path, 0.7), is_long_jump


    def levy_truncated_sample(self, mu=2.0, l_min=0.7, lambda_=0.01):

        while True:
            theta = self.bee.random.uniform(0, 360)

            # Sample length using rejection sampling
            while True:
                U1 = self.bee.random.uniform(0, 1)
                candidate = l_min * (1 - U1) ** (-1 / mu)
                U2 = self.bee.random.uniform(0, 1)
                if U2 < math.exp(-lambda_ * candidate):
                    distance = candidate
                    break
                
            is_long_jump = distance > self.LONG_JUMP_THRESHOLD
            
            destination_coord = self.compute_endpoint(self.bee.pos[0], self.bee.pos[1], distance, theta)
            start_point = Point(self.bee.pos)
            bee_path = LineString([self.bee.pos, destination_coord])

            if self.bee.model.space.out_of_bounds(pos=destination_coord):
                max_distance = -1
                furthest_intersection = None
                for wall in self.bee.model.walls:
                    if bee_path.intersects(wall):
                        intersection = bee_path.intersection(wall)
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
                        elif intersection.geom_type == "LineString":
                            for pt_coords in [list(intersection.coords)[0], list(intersection.coords)[-1]]:
                                pt = Point(pt_coords)
                                dist = start_point.distance(pt)
                                if dist > max_distance:
                                    max_distance = dist
                                    furthest_intersection = pt
                if furthest_intersection:
                    bee_path = LineString([self.bee.pos, furthest_intersection])

            return self.sample_line(bee_path, 0.7), is_long_jump
    
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
    


class CompositeLevyBehavior(BeeBehavior):
    def __init__(self, landing_prob, memory_size=5):
        super().__init__(bee_behavior="composite_levy", landing_prob=landing_prob)
        self.memory_size = memory_size
        self.memory_queue = []

    def step(self):
        if self.bee.forage_timer == 0:
            if self.bee.travel_route == []:
                self.get_new_directions()
            else:
                next_coord = self.bee.travel_route.pop(0)
                next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                self.bee.model.space.move_agent(self.bee, next_coord)

                if self.bee.travel_route == []:
                    if self.bee.is_long_jump:
                        self.get_new_directions()
                        self.bee.is_long_jump = False
                        return

                    below = self.bee.model.space.get_neighbors(self.bee.pos, 0, include_center=True)
                    plant_agents = [agent for agent in below if isinstance(agent, Plant)]
                    available_plants = [plant for plant in plant_agents if plant.bee_on is None and plant != self.bee.last_plant]

                    if available_plants and self.land_on_flower(self.bee):
                        plant = available_plants[0]
                        self.bee.forage_timer = self.bee.forage_timer_max
                        self.bee.on_plant = plant
                        plant.bee_on = self.bee
                        plant.visit_count += 1
                        plant.visit_timestamp.append(self.bee.model.steps)
                        plant.visited = True
                        self.memory_queue.append(1)
                    else:
                        self.memory_queue.append(0)
                        self.get_new_directions()

                    if len(self.memory_queue) > self.memory_size:
                        self.memory_queue.pop(0)
        else:
            self.bee.forage_timer -= 1
            if self.bee.forage_timer == 0:
                self.bee.last_plant = self.bee.on_plant
                self.bee.on_plant.bee_on = None
                self.bee.on_plant = None

    def get_new_directions(self):
        neighbors = self.bee.model.space.get_neighbors(pos=self.bee.pos, radius=self.bee.vision_range, include_center=True)
        plant_agents = [agent for agent in neighbors if isinstance(agent, Plant)]
        available_plants = [plant for plant in plant_agents if plant.bee_on is None and plant != self.bee.last_plant]

        if available_plants == []:
            if len(self.memory_queue) < self.memory_size:
                self.bee.travel_route, self.bee.is_long_jump = self.levy_truncated_sample(max_jump=self.LONG_JUMP_THRESHOLD)
            else:
                success_rate = sum(self.memory_queue) / self.memory_size
                if self.bee.random.random() > success_rate:
                    self.bee.travel_route, self.bee.is_long_jump = self.levy_truncated_sample()
                else:
                    self.bee.travel_route, self.bee.is_long_jump = self.levy_truncated_sample(max_jump=self.LONG_JUMP_THRESHOLD)
        else:
            random_plant = self.bee.random.choice(available_plants)
            self.bee.travel_route = [random_plant.pos]

    def levy_truncated_sample(self, mu=2.0, l_min=0.7, lambda_=0.01, max_jump=None):
        while True:
            theta = self.bee.random.uniform(0, 360)

            while True:
                U1 = self.bee.random.uniform(0, 1)
                candidate = l_min * (1 - U1) ** (-1 / mu)
                U2 = self.bee.random.uniform(0, 1)
                if U2 < math.exp(-lambda_ * candidate):
                    distance = candidate
                    break

            if max_jump is not None and distance >= max_jump:
                continue

            is_long_jump = distance > self.LONG_JUMP_THRESHOLD

            destination_coord = self.compute_endpoint(self.bee.pos[0], self.bee.pos[1], distance, theta)
            start_point = Point(self.bee.pos)
            bee_path = LineString([self.bee.pos, destination_coord])

            if self.bee.model.space.out_of_bounds(pos=destination_coord):
                max_distance = -1
                furthest_intersection = None
                for wall in self.bee.model.walls:
                    if bee_path.intersects(wall):
                        intersection = bee_path.intersection(wall)
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
                        elif intersection.geom_type == "LineString":
                            for pt_coords in [list(intersection.coords)[0], list(intersection.coords)[-1]]:
                                pt = Point(pt_coords)
                                dist = start_point.distance(pt)
                                if dist > max_distance:
                                    max_distance = dist
                                    furthest_intersection = pt
                if furthest_intersection:
                    bee_path = LineString([self.bee.pos, furthest_intersection])

            return self.sample_line(bee_path, 0.7), is_long_jump



    

class Bee(mesa.Agent):
    def __init__(self, model, beebehavior : BeeBehavior, beehive):
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
        self.space = self.model.space
        self.beehive = beehive
        self.beehive_id = beehive.my_id  # ID of the beehive the bee belongs to
        self.pos = None
        

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
        self.last_plant = None
        
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
        if self.beehive_id == 0:
            return {"color" : "white", "edgecolors" : "none"}
        elif self.beehive_id == 1:
            return {"color" : "black", "edgecolors" : "none"}

"""
Plant behavior class
"""
class Plant(mesa.Agent):
    def __init__(self, model):
        """
        Input:
        - model: the model instance of the environment
        - gender: gender of the flower
        - coord: coordinates of the flower in the environment
        
        """
        super().__init__(model)
        self.agent_type = "plant"
        
        self.model = model
        self.pos = None      # (x, y)
        self.bee_on = None    # Bee on the flower

        self.visit_count = 0
        self.visit_timestamp = []
        
        self.visited = False
        
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
        
        """
        return {"color" : "blue", "edgecolors" : "none"}
