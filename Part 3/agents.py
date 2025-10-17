import mesa
import math
from shapely.geometry import LineString, Point
import numpy as np
from collections import deque
import warnings
"""
Class for declaring bee and plant objects, and defining their behaviors

NOTE: use self.random to get a random seeded number


TODO: After new bee behavior is implemented and verified, do the following:
- [ ] Use class MyBeeBehavior as template, update PureLevyBehavior, TruncatedLevyBehavior, CompositeLevyBehavior and BiasedCompositeLevyBehavior

Currently MyBeeBehavior works as expected.
Currently temperature is hard coded into the simulation. Eg: 0 to 10m into the polytunnel, bees have 100% chance to land on a flower,
10m to 20m into the polytunnel, bees have 95% chance to land on a flower...so on so forth.
"""
class BeeBehavior():
    """
    Base class for bee behaviors. Follows the flow chart of bee behavior from
    paper "Simulation-governed design and tuning of greenhouse for successful
    bee pollination" by Alan Dorin.
    """
    def __init__(self, landing_prob):
        self.bee = None
        self.landing_prob = landing_prob
        self.LONG_JUMP_THRESHOLD = 3.0

    def step(self):
        if self.bee.forage_timer == 0:
            if not self.bee.travel_route:
                self.get_new_action()
            else:
                next_coord = self.bee.travel_route.pop(0)
                next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                self.bee.model.space.move_agent(self.bee, next_coord)

                if not self.bee.travel_route:
                    if self.bee.is_long_jump:
                        self.get_new_action()
                        self.bee.is_long_jump = False
                        if hasattr(self.bee, "reward_memory"): # EDIT
                            self.bee.reward_memory = []
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
                        plant.visited_this_step = True
                        if hasattr(self.bee, "reward_memory"): # EDIT
                            self.bee.reward_memory.append(1)
                    else:
                        if hasattr(self.bee, "reward_memory"): # EDIT
                            self.bee.reward_memory.append(0)
                        self.get_new_action()

                    if hasattr(self.bee, "reward_memory") and hasattr(self.bee, "reward_size"): # EDIT
                        if len(self.bee.reward_memory) > self.bee.reward_size:
                            self.bee.reward_memory.pop(0)
        else:
            self.bee.forage_timer -= 1
            if self.bee.forage_timer == 0:
                self.bee.last_plant = self.bee.on_plant
                self.bee.on_plant.bee_on = None
                self.bee.on_plant = None

    def get_new_action(self):
        raise NotImplementedError("Subclasses must implement get_new_action()")
 
    def land_on_flower(self, bee):
        return bee.random.random() < self.landing_prob

    def levy_pure_sample(self, mu=2.0, l_min=0.7):
        while True:
            theta = self.bee.random.uniform(0, 360)
            U = self.bee.random.uniform(0, 1)
            distance = l_min * (1 - U) ** (-1 / mu)
            is_long_jump = distance > self.LONG_JUMP_THRESHOLD

            destination_coord = self.compute_endpoint(self.bee.pos[0], self.bee.pos[1], distance, theta)
            bee_path = LineString([self.bee.pos, destination_coord])

            if self.bee.model.space.out_of_bounds(pos=destination_coord):
                bee_path = self.handle_wall_collisions(bee_path)

            return self.sample_line(bee_path, 0.7), is_long_jump

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
            bee_path = LineString([self.bee.pos, destination_coord])

            if self.bee.model.space.out_of_bounds(pos=destination_coord):
                bee_path = self.handle_wall_collisions(bee_path)

            return self.sample_line(bee_path, 0.7), is_long_jump

    def handle_wall_collisions(self, bee_path):
        start_point = Point(self.bee.pos)
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
            return LineString([self.bee.pos, furthest_intersection])
        else:
            return bee_path
    
    def clamp_position(self, pos, width, height):
        x = min(max(0, pos[0]), width - 0.001)
        y = min(max(0, pos[1]), height - 0.001)
        return (x, y)
    
    def sample_line(self, line: LineString, step: float) -> list:
        if line.length == 0:
            return [line.coords[0]]

        distances = np.arange(0, line.length, step).tolist()

        if distances[-1] < line.length:
            distances.append(line.length)

        points = [line.interpolate(distance) for distance in distances]
        return [tuple(point.coords[0]) for point in points]

    def compute_endpoint(self, x0, y0, distance, theta):
        angle_rad = math.radians(theta)
        x1 = x0 + distance * math.cos(angle_rad)
        y1 = y0 + distance * math.sin(angle_rad)
        return x1, y1


class Bee(mesa.Agent):
    def __init__(self, model, beebehavior: BeeBehavior, beehive):
        super().__init__(model)
        self.agent_type = "bee"
        self.model = model
        self.behavior = beebehavior
        self.behavior.bee = self
        self.space = self.model.space
        self.beehive = beehive
        self.beehive_id = beehive.my_id
        self.pos = None

        self.vision_range = 0.7

        self.forage_timer = 0
        self.forage_timer_max = 3

        self.has_pollen = False
        self.pollen_timer = 0
        self.pollen_timer_max = 25

        self.travel_route = []
        self.on_plant = None
        
        self.last_plant = None

        self.theta = None
        
    def step(self):
        self.behavior.step()

    def color_code(self):
        if self.beehive_id == 0:
            return {"color": "white", "edgecolors": "none"}
        elif self.beehive_id == 1:
            return {"color": "black", "edgecolors": "none"}


class MyBee(Bee):
    def __init__(self, model, behavior, beehive, reward_size):
        super().__init__(model, behavior, beehive)
        self.reward_size = reward_size
        self.reward_memory = deque(maxlen=self.reward_size)
        self.latest_visited_flowers = deque(maxlen=5)
        self.latest_visited_flowers_steps = deque(maxlen=5)
        self.stomach_capacity = 20.0 #! original was 20
        self.nectar_in_stomach = 0.0
        self.homing = False
        
        self.collection_rate = 3 # The amount of nectar to collect
        
        self.nectar_to_collect = 0.0
        
        self.target_plant = None
        self.nectar_reward_threshold = 1
        self.turning_angles = []
        self.jump_distances = []
        
        self.is_long_jump = False
        
        self.last_sugar_concentration = None
        
    def add_nectar(self, amount):
        # #! printing 
        # if self.unique_id == 5:
        #     print(f"added nectar {amount}, stomach has {min(self.nectar_in_stomach + amount, self.stomach_capacity)}")
        self.nectar_in_stomach = min(self.nectar_in_stomach + amount, self.stomach_capacity)

    def is_full(self):
        return self.nectar_in_stomach >= self.stomach_capacity

      
class Plant(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.agent_type = "plant"

        self.model = model
        self.pos = None
        self.bee_on = None

        self.visit_count = 0   #! remove
        
        self.information_duration = 60 * 5          # The amount of time a timestep is recorded in visit_timestamp
        self.visit_timestamp = deque(maxlen=60 * 5) # Records the number of bees in the last 5 minutes
        
        self.visited = False   #! check to remove
        self.visited_this_step = False #! check to remove

        
        # New parameter for plants
        self.sugar_concentration = None
    
    def on_placement(self):
        if self.pos is not None:
            x = self.pos[0]
            self.sugar_concentration = self._compute_sugar_concentration(x)

    def _compute_sugar_concentration(self, x):
        """
        ! Define how sugar concentration is distributed in the enviornment
        Sugar concentration depends on distance from tunnel ends (x = 10 or x = 110).
        x=10 or 110 -> 1.0
        x=20 or 100 -> 0.95
        x=30 or 90  -> 0.90
        ...
        x=60        -> 0.75
        """
        # Distance in units of 10 from nearest tunnel edge
        dist_from_edge = min(abs(x - 10), abs(x - 110))
        
        # Clamp to maximum of 50 units (center)
        capped_distance = min(dist_from_edge, 50)

        # Compute scaled concentration
        reward = 1.0 - (capped_distance // 10) * 0.05
        return round(reward, 2)
    
    def step(self):
        self.visited_this_step = self.bee_on is not None

    def color_code(self):
        return {"color": "blue", "edgecolors": "none"}


class MyPlant(Plant):
    def __init__(self, model, replenish_rate=0.0):
        super().__init__(model)
        self.scent_mark_timer = 0
        self.nectar = 13.0
        self.max_nectar = 13.0
        self.replenish_rate = replenish_rate
        self.reserved_by = None
        self.scent_duration = 45
        
        # self.START = 10
        # self.END = 30

    def has_scent(self):
        return self.scent_mark_timer > 0

    def leave_scent(self):
        self.scent_mark_timer = self.scent_duration

    def _filter_old_visits(self):
        """
        Remove timestamps older than information_duration from self.visit_timestamp.
        This is called at every simulation step.
        """
        current_time = self.model.steps  # or however you track global time
        while self.visit_timestamp and self.visit_timestamp[0] < current_time - self.information_duration:
            self.visit_timestamp.popleft()
    
    def step(self):
        #! Print
        # if self.model.steps in range(0, 3) and self.unique_id == 123:
        
        #     current_step = self.model.steps-1
        #     print(f"{current_step} plant {self.unique_id} has visited history {self.visit_timestamp}")  
        
        if self.scent_mark_timer > 0:
            self.scent_mark_timer -= 1

        if self.nectar < self.max_nectar:            
            self.nectar = min(self.nectar + self.replenish_rate, self.max_nectar)

        # Update visit_timestamps
        self._filter_old_visits()


class BeeHive(mesa.Agent):
    """
    Class representing a beehive in the simulation. Each beehive has bees assigned
    to it.
    """
    def __init__(self, model, my_id):
        super().__init__(model)
        self.agent_type = "beehive"
        self.model = model
        self.pos = None
        self.my_id = my_id # TODO: Check if this is needed, or if it should be removed (creating new agents in mesa should just assign a unique ID). If so, the go back to model.py and edit the code.

    def color_code(self):
        return {"color": "brown", "edgecolors": "none"}


class PureLevyBehavior(BeeBehavior):
    """
    Pure Levy flight behavior for bees. This behavior samples from a Levy distribution
    to determine the distance for the bee to travel. Direction is chosen uniformly.
    """
    def __init__(self, landing_prob):
        super().__init__(landing_prob=landing_prob)

    def get_new_action(self):
        neighbors = self.bee.model.space.get_neighbors(pos=self.bee.pos, radius=self.bee.vision_range, include_center=True)
        plant_agents = [agent for agent in neighbors if isinstance(agent, Plant)]
        available_plants = [plant for plant in plant_agents if plant.bee_on is None and plant != self.bee.last_plant]

        if available_plants == []:
            self.bee.travel_route, self.bee.is_long_jump = self.levy_pure_sample()
        else:
            random_plant = self.bee.random.choice(available_plants)
            self.bee.travel_route = [random_plant.pos]


class TruncatedLevyBehavior(BeeBehavior):
    """
    Similar to PureLevyBehavior, but uses a truncated Levy distribution.
    """
    def __init__(self, landing_prob):
        super().__init__(landing_prob=landing_prob)

    def get_new_action(self):
        neighbors = self.bee.model.space.get_neighbors(pos=self.bee.pos, radius=self.bee.vision_range, include_center=True)
        plant_agents = [agent for agent in neighbors if isinstance(agent, Plant)]
        available_plants = [plant for plant in plant_agents if plant.bee_on is None and plant != self.bee.last_plant]

        if available_plants == []:
            self.bee.travel_route, self.bee.is_long_jump = self.levy_truncated_sample()
        else:
            random_plant = self.bee.random.choice(available_plants)
            self.bee.travel_route = [random_plant.pos]


class CompositeLevyBehavior(BeeBehavior):
    """
    An extension of the truncated Levy behavior. It decides to make a long jump
    or short jump based on the reward memory queue. Everytime a flower is visited
    and the bee has collected nectar from it, then add 1 to the queue. Probability
    of making a short jump = sum(reward_memory) / queue size (when queue size is full).
    
    """
    def __init__(self, landing_prob, reward_size=5):
        super().__init__(landing_prob=landing_prob)
        self.reward_size = reward_size
        self.bee.reward_memory = []

    def get_new_action(self):
        neighbors = self.bee.model.space.get_neighbors(pos=self.bee.pos, radius=self.bee.vision_range, include_center=True)
        plant_agents = [agent for agent in neighbors if isinstance(agent, Plant)]
        available_plants = [plant for plant in plant_agents if plant.bee_on is None and plant != self.bee.last_plant]

        if available_plants == []:
            if len(self.bee.reward_memory) < self.bee.reward_size:
                self.bee.travel_route, self.bee.is_long_jump = self.levy_truncated_sample(max_jump=self.LONG_JUMP_THRESHOLD)
            else:
                success_rate = sum(self.bee.reward_memory) / self.bee.reward_size
                if self.bee.random.random() > success_rate:
                    self.bee.travel_route, self.bee.is_long_jump = self.levy_truncated_sample()
                else:
                    self.bee.travel_route, self.bee.is_long_jump = self.levy_truncated_sample(max_jump=self.LONG_JUMP_THRESHOLD)
        else:
            random_plant = self.bee.random.choice(available_plants)
            self.bee.travel_route = [random_plant.pos]


class BiasedCompositeLevyBehavior(BeeBehavior):
    """
    Extension of CompositeLevyBehavior where the direction of the jump is taken
    into account of the bee's current orientation. The biased distribtuion is sampled
    from the paper "A simulation to Assess Social Bee Visual Attention During Complex
    Search Tasks" by Zoe.
    """
    def __init__(self, landing_prob, reward_size=5, biased_distribution=None):
        super().__init__(landing_prob=landing_prob)
        self.reward_size = reward_size
        self.bee.reward_memory = []

        if biased_distribution is None:
            self.biased_distribution = lambda n=1: np.random.uniform(-180, 180, size=n)
        else:
            self.biased_distribution = biased_distribution

    def get_new_action(self):
        neighbors = self.bee.model.space.get_neighbors(self.bee.pos, self.bee.vision_range, include_center=True)
        plant_agents = [agent for agent in neighbors if isinstance(agent, Plant)]
        available_plants = [plant for plant in plant_agents if plant.bee_on is None and plant != self.bee.last_plant]

        if not available_plants:
            if len(self.bee.reward_memory) < self.bee.reward_size:
                self.bee.travel_route, self.bee.is_long_jump = self.sample_from_biased_distribution(max_jump=self.LONG_JUMP_THRESHOLD)
            else:
                success_rate = sum(self.bee.reward_memory) / self.bee.reward_size
                if self.bee.random.random() > success_rate:
                    self.bee.travel_route, self.bee.is_long_jump = self.sample_from_biased_distribution()
                else:
                    self.bee.travel_route, self.bee.is_long_jump = self.sample_from_biased_distribution(max_jump=self.LONG_JUMP_THRESHOLD)
        else:
            random_plant = self.bee.random.choice(available_plants)
            self.bee.travel_route = [random_plant.pos]

            dx = random_plant.pos[0] - self.bee.pos[0]
            dy = random_plant.pos[1] - self.bee.pos[1]
            theta_to_flower = math.degrees(math.atan2(dy, dx))

            if theta_to_flower > 180:
                theta_to_flower -= 360
            elif theta_to_flower < -180:
                theta_to_flower += 360

            self.bee.theta = theta_to_flower

    def sample_from_biased_distribution(self, max_jump=None):
        delta_theta = self.biased_distribution(self.bee.random, 1)[0]

        new_theta = (self.bee.theta + delta_theta) % 360
        if new_theta > 180:
            new_theta -= 360
        self.bee.theta = new_theta

        distance = self.bee.random.uniform(0.7, 3.5)

        if max_jump is not None and distance >= max_jump:
            distance = max_jump * 0.99

        is_long_jump = distance > self.LONG_JUMP_THRESHOLD

        destination_coord = self.compute_endpoint(self.bee.pos[0], self.bee.pos[1], distance, new_theta)
        bee_path = LineString([self.bee.pos, destination_coord])

        if self.bee.model.space.out_of_bounds(pos=destination_coord):
            bee_path = self.handle_wall_collisions(bee_path)

        return self.sample_line(bee_path, 0.7), is_long_jump

# NOTE: based class
class MyBeeBehavior(BeeBehavior):
    """
    An extension of BiasedCompositeLevyBehavior. The differences:
    - Homing behavior: Bee flies back to hive only when it's stomach is full
    - Flower memory: Each bee will have a memory of the last five flowers they have visited.
    - Flower behavior: Flower has a set amount of nectar. Each visit time a bee
                           visits a flower with nectar, regardless how much nectar it has, it will
                           always take 3 timesteps to forage.
    - Scent marking: When a bee lands on a flower, a scent will be left on the flower, regardless
                         whether the bee has foraged or not.
                         
    What I expect when running this behavior in an empty environment: the bee will perform five
    short jumps before making one long jump. After that, it will repeat the process.
    
    """
    def __init__(self, landing_prob, biased_distribution=None):
        super().__init__(landing_prob=landing_prob)
        
        self.START = 95
        self.END = 140
        
        self.BEE_IDS = [5]
        
        
        self.biased_distribution = biased_distribution

    def step(self):

        # 0. Check if bee is returning to the hive
        if self.bee.homing:
            if not self.bee.travel_route:
                
                # Arrived at hive
                self.bee.homing = False
                self.bee.nectar_in_stomach = 0.0
                self.bee.reward_memory.clear()
                self.bee.latest_visited_flowers.clear()
                self.bee.latest_visited_flowers_steps.clear()

                self.theta = None
                self.bee.last_sugar_concentration = None
                self.new_action_and_move()
                return
            else:
                # Continue travel toward the hive
                next_coord = self.bee.travel_route.pop(0)
                next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                self.bee.model.space.move_agent(self.bee, next_coord)
                return
        
        # 1. Spending time on foraging nectar?
        # 1.1 No, then check if the bee is at the destination
        if self.bee.forage_timer == 0 and self.bee.on_plant is None:
            

            # 2. Is the bee at the set destination?
            # 2.1 Yes, empty travel_route means it is at the destination, check if there is a plant below it
            if not self.bee.travel_route: 
                
                # If at the end of long jump, just get new_action and move
                if self.bee.is_long_jump:
                    # After finishing a long jump, get next directions.
                    # Move to 8
                    self.bee.is_long_jump = False
                    self.bee.reward_memory.clear()
                    self.new_action_and_move()
                    return
                
                plant_below_list = self.get_plant_below()
                # Check if there is a flower below
                if plant_below_list:
                    plant = plant_below_list[0]
                    self.bee.latest_visited_flowers.append(plant)
                    self.bee.latest_visited_flowers_steps.append(self.bee.model.steps)
                    
                    if plant.bee_on is None and not plant.has_scent():
                        
                        # Attempt to land on the flower
                        landing = self.land_on_flower(self.bee, plant)
                        
                        if landing:
                            self.bee.on_plant = plant
                            plant.bee_on = self.bee
                            
                            
                            # Update the timestamp when bee lands on the plant
                            plant.visit_timestamp.append(self.bee.model.steps)
                            
                            nectar_to_collect = min(self.bee.collection_rate, plant.nectar, self.bee.stomach_capacity - self.bee.nectar_in_stomach)
                            
                            if nectar_to_collect > 0:
                                self.bee.forage_timer = self.bee.forage_timer_max
                                self.bee.nectar_to_collect = nectar_to_collect
                                
                                # here is where to update the reward memroy
                                # self.bee.reward_memory.append(1)
                                self.update_reward_memory(self.bee)
                                
                                self.new_action_and_move()
                                
                            elif nectar_to_collect == 0:
                                self.bee.reward_memory.append(0)
                                
                                plant.bee_on = None
                                plant.leave_scent()
                                self.bee.on_plant = None
                                
                                self.new_action_and_move()
                                
                            else:
                                raise ValueError("Nectar to collect should not be negative 2")
                        # Did not land due to probability
                        elif not landing:
                            self.new_action_and_move()
                            
                        else:
                            raise ValueError("Landing logic error 2")

                    elif plant.bee_on and not plant.has_scent(): 
                        self.bee.reward_memory.append(1)
                        self.new_action_and_move()
                    
                    elif plant.has_scent():
                        self.bee.reward_memory.append(0)
                        self.new_action_and_move()
                    
                    else:
                        assert False, "Defensive programming 1, different reason why bee cannot land"       
                elif not plant_below_list:
                    self.bee.reward_memory.append(0)
                    self.new_action_and_move()
                else:
                    assert False, "Plant below detection problem"
            
            # 2.2 No, travel_route is not empty, move to the next location
            else:
                # If not long jumping, then scan while long jumping, if no plant is within range, then just travel
                available_plants = self.get_available_plants(vision_range=self.bee.vision_range)
                
                # Not performing a long jump and there are plants nearby, randomly choose one and fly towards it
                if not self.bee.is_long_jump and available_plants:
                    random_plant = self.bee.random.choice(available_plants)
                    self.bee.travel_route = [random_plant.pos]
                    
                    dx = random_plant.pos[0] - self.bee.pos[0]
                    dy = random_plant.pos[1] - self.bee.pos[1]
                    theta_to_flower = math.degrees(math.atan2(dy, dx))
                    theta_to_flower = (theta_to_flower + 180) % 360 - 180
                    self.bee.theta = theta_to_flower

                    next_coord = self.bee.travel_route.pop(0)
                    next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                    self.bee.model.space.move_agent(self.bee, next_coord)
                    return                    
                          
                else: # Continue down the path
                    next_coord = self.bee.travel_route.pop(0)
                    next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                    self.bee.model.space.move_agent(self.bee, next_coord)
   
   
   
        # 1.2 yes, currently still foraging
        elif self.bee.on_plant is not None or self.bee.forage_timer > 0:
            self.bee.forage_timer -= 1
            
            if self.bee.forage_timer == 0: # Finishe foraging
                # Update the plant nectar amount and bee stomach content
                self.bee.on_plant.nectar -=self.bee.nectar_to_collect
                self.bee.add_nectar(self.bee.nectar_to_collect)
                
                # Clear temp colleciton parameters
                self.bee.nectar_to_collect = 0.0
                
                # Wrap up interaction with flower
                self.bee.on_plant.bee_on = None
                self.bee.on_plant.leave_scent()
                self.bee.on_plant = None
                
                # If the bee stomach is full of nectar, then start heading home,
                # otherwise take enw action
                if self.bee.is_full():
                    self.bee.homing = True
                    bee_path = LineString([self.bee.pos, self.bee.beehive.pos])
                    self.bee.travel_route = self.sample_line(bee_path, 0.7)
                    dx = self.bee.beehive.pos[0] - self.bee.pos[0]
                    dy = self.bee.beehive.pos[1] - self.bee.pos[1]
                    theta = math.degrees(math.atan2(dy, dx))
                    if theta > 180: theta -= 360
                    elif theta < -180: theta += 360
                    self.bee.theta = theta
                else:
                    self.new_action_and_move()
        else:
            assert False, "Defensive programming, this should not be executed"

    
    def update_reward_memory(self, bee):
        current_sugar = bee.on_plant.sugar_concentration
        last_sugar = bee.last_sugar_concentration
        
        if last_sugar is None:
            reward = 1
            
        else:
            if current_sugar >= last_sugar:
                reward = 1
            else:
                drop = (last_sugar - current_sugar) / last_sugar
                reward = max(0, 1 - drop) # Reward between 0 and 1

        bee.reward_memory.append(reward)
        bee.last_sugar_concentration = current_sugar
            
    def get_new_action(self):
        """
        This function will update bee's travel_route. If there is a flower nearby, randomly choose one and
        update travel_route as [plant position]. Otherwise, a new direction and distance is sampled and travel_route
        is updated accordingly.
        """
        # 8. Are any nearby flowers not in the visited list?
        available_plants = self.get_available_plants(vision_range=self.bee.vision_range)

        # There is no plants nearby, decide whether to do long jump or not. 
        if not available_plants:
            if len(self.bee.reward_memory) < self.bee.reward_size:
                self.bee.travel_route, self.bee.is_long_jump = self.sample_from_biased_distribution(max_jump=self.LONG_JUMP_THRESHOLD)
            else:
                long_jump_rate = sum(self.bee.reward_memory) / self.bee.reward_size
                if self.bee.random.random() > long_jump_rate:
                    self.bee.travel_route, self.bee.is_long_jump = self.sample_from_biased_distribution()
                else:
                    self.bee.travel_route, self.bee.is_long_jump = self.sample_from_biased_distribution(max_jump=self.LONG_JUMP_THRESHOLD)
        
        # There are available palnts nearby
        else:
            random_plant = self.bee.random.choice(available_plants)
            self.bee.travel_route = [random_plant.pos]

            dx = random_plant.pos[0] - self.bee.pos[0]
            dy = random_plant.pos[1] - self.bee.pos[1]
            theta_to_flower = math.degrees(math.atan2(dy, dx))

            if theta_to_flower > 180:
                theta_to_flower -= 360
            elif theta_to_flower < -180:
                theta_to_flower += 360

            self.bee.theta = theta_to_flower

    def sample_from_biased_distribution(self, max_jump=None):
        ""
        # TODO: Move the biased turn angle function here
        delta_theta = self.biased_distribution(self.bee.random, 1)[0]

        ########### Debugging and Verification ###########
        self.bee.turning_angles.append(delta_theta)
        ##################################################
        
        new_theta = (self.bee.theta + delta_theta) % 360
        if new_theta > 180:
            new_theta -= 360
        self.bee.theta = new_theta

        # Replace uniform sampling with Truncated Lévy sampling
        alpha = 1.5 
        min_distance = 0.7
        max_distance = max_jump if max_jump is not None else 20.0  # set a cap if none given

        # Truncated Lévy sampling logic (same as in TruncatedLevyBehavior)
        while True:
            u = self.bee.random.random()
            distance = min_distance * ((1 - u) ** (-1 / alpha))
            if distance <= max_distance:
                break

        # Special case: force long jump if permitted but undershoots threshold
        if max_jump is None and distance < self.LONG_JUMP_THRESHOLD:
            distance += self.LONG_JUMP_THRESHOLD

        is_long_jump = distance > self.LONG_JUMP_THRESHOLD

        ########### Debugging and Verification ###########
        self.bee.jump_distances.append(distance)
        ##################################################
        
        destination_coord = self.compute_endpoint(self.bee.pos[0], self.bee.pos[1], distance, new_theta)
        bee_path = LineString([self.bee.pos, destination_coord])
        
        if self.bee.model.space.out_of_bounds(pos=destination_coord):
            bee_path = self.handle_wall_collisions(bee_path)

        return self.sample_line(bee_path, 0.7), is_long_jump


    def get_plant_below(self):
        """
        Return a list that contains the plant below the bee that is available.
        
        A plant is avaiable when there is no bee on the plant, and plant has no scent.
        """
        agents_below = self.bee.model.space.get_neighbors(self.bee.pos, 0, include_center = True)
        
        plant_agent = [agent for agent in agents_below if isinstance(agent, Plant)]
        
        return plant_agent

    def get_available_plants(self, vision_range=0.7):
        """
        Get a list of available plants that the bee can land on.
        
        A plant is available if:
        - It is within the bee's vision range
        - It does not have a bee on it (bee_on is None)
        - It is not in the bee's latest visited flowers list
        
        Note: it will still fly towards a flower even if there is a scent mark on it.
        """
        neighbors = self.bee.model.space.get_neighbors(self.bee.pos, vision_range, include_center=True)
        plant_agents = [agent for agent in neighbors if isinstance(agent, Plant)]
        available_plants = [plant for plant in plant_agents if plant.bee_on is None and plant not in self.bee.latest_visited_flowers]
        return available_plants
    
    def land_on_flower(self, bee, plant):
        """ 
        Take in the plant object and check if the bee can land on it or not.
        """
        return bee.random.random() < self.landing_prob and plant.has_scent() is False
    

    def new_action_and_move(self):
        self.get_new_action()
        
        if self.bee.travel_route:
            next_coord = self.bee.travel_route.pop(0)
            next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
            self.bee.model.space.move_agent(self.bee, next_coord)


class ComparativeRewards(BeeBehavior):
    """
    An extension of BiasedCompositeLevyBehavior. The differences:
    - Homing behavior: Bee flies back to hive only when it's stomach is full
    - Flower memory: Each bee will have a memory of the last five flowers they have visited.
    - Flower behavior: Flower has a set amount of nectar. Each visit time a bee
                           visits a flower with nectar, regardless how much nectar it has, it will
                           always take 3 timesteps to forage.
    - Scent marking: When a bee lands on a flower, a scent will be left on the flower, regardless
                         whether the bee has foraged or not.
                         
    Expected in an empty environment (original intent): five short jumps then one long jump, repeat.
    """

    def __init__(self, landing_prob, biased_distribution=None):
        super().__init__(landing_prob=landing_prob)
        
        self.START = 95
        self.END = 140
        self.BEE_IDS = [5]

        self.biased_distribution = biased_distribution

        # NEW: track last successful nectar collection amount for reward comparison
        self.last_nectar_collected = None

    def step(self):

        # 0. Check if bee is returning to the hive
        if self.bee.homing:
            if not self.bee.travel_route:
                # Arrived at hive
                self.bee.homing = False
                self.bee.nectar_in_stomach = 0.0
                self.bee.reward_memory.clear()
                self.bee.latest_visited_flowers.clear()
                self.bee.latest_visited_flowers_steps.clear()
                self.theta = None
                self.bee.last_sugar_concentration = None
                # NEW: reset nectar comparison baseline
                self.last_nectar_collected = None

                self.new_action_and_move()
                return
            else:
                # Continue travel toward the hive
                next_coord = self.bee.travel_route.pop(0)
                next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                self.bee.model.space.move_agent(self.bee, next_coord)
                return
        
        # 1. Spending time on foraging nectar?
        # 1.1 No, then check if the bee is at the destination
        if self.bee.forage_timer == 0 and self.bee.on_plant is None:

            # 2. Is the bee at the set destination?
            # 2.1 Yes, empty travel_route means it is at the destination, check if there is a plant below it
            if not self.bee.travel_route: 
                
                # If at the end of long jump, just get new_action and move
                if self.bee.is_long_jump:
                    # After finishing a long jump, get next directions.
                    # Move to 8
                    self.bee.is_long_jump = False
                    self.bee.reward_memory.clear()
                    # NEW: reset nectar comparison baseline when long jump cycle resets
                    self.last_nectar_collected = None

                    self.new_action_and_move()
                    return
                
                plant_below_list = self.get_plant_below()
                # Check if there is a flower below
                if plant_below_list:
                    plant = plant_below_list[0]
                    self.bee.latest_visited_flowers.append(plant)
                    self.bee.latest_visited_flowers_steps.append(self.bee.model.steps)
                    if plant.bee_on is None and not plant.has_scent():
                        
                        # Attempt to land on the flower
                        landing = self.land_on_flower(self.bee, plant)
                        
                        if landing:
                            self.bee.on_plant = plant
                            plant.bee_on = self.bee
                            
                            # Update the timestamp when bee lands on the plant
                            plant.visit_timestamp.append(self.bee.model.steps)
                            
                            nectar_to_collect = min(
                                self.bee.collection_rate,
                                plant.nectar,
                                self.bee.stomach_capacity - self.bee.nectar_in_stomach
                            )
                            
                            if nectar_to_collect > 0:
                                self.bee.forage_timer = self.bee.forage_timer_max
                                self.bee.nectar_to_collect = nectar_to_collect
                                
                                # NEW: update reward memory based on nectar amount vs last time
                                self.update_reward_memory(self.bee, nectar_to_collect)
                                
                                self.new_action_and_move()
                                
                            elif nectar_to_collect == 0:
                                # No nectar available to collect → treat as 0 (unchanged)
                                self.bee.reward_memory.append(0)
                                
                                plant.bee_on = None
                                plant.leave_scent()
                                self.bee.on_plant = None
                                
                                self.new_action_and_move()
                                
                            else:
                                raise ValueError("Nectar to collect should not be negative 2")

                        # Did not land due to probability
                        elif not landing:
                            # Unsuccessful attempt → keep original 0
                            self.new_action_and_move()
                            
                        else:
                            raise ValueError("Landing logic error 2")

                    elif plant.bee_on and not plant.has_scent(): 
                        # Another bee is on it (no scent yet) → keep original 1 or 0?
                        # Your original code appended 1 here; preserving that logic:
                        self.bee.reward_memory.append(1)
                        self.new_action_and_move()
                    
                    elif plant.has_scent():
                        # Scent present → consider as 0 (unchanged)
                        self.bee.reward_memory.append(0)
                        self.new_action_and_move()
                    
                    else:
                        assert False, "Defensive programming 1, different reason why bee cannot land"       

                elif not plant_below_list:
                    # No plant underfoot at destination → 0 (unchanged)
                    self.bee.reward_memory.append(0)
                    self.new_action_and_move()
                else:
                    assert False, "Plant below detection problem"
            
            # 2.2 No, travel_route is not empty, move to the next location
            else:
                # If not long jumping, then scan while long jumping, if no plant is within range, then just travel
                available_plants = self.get_available_plants(vision_range=self.bee.vision_range)
                
                # Not performing a long jump and there are plants nearby, randomly choose one and fly towards it
                if not self.bee.is_long_jump and available_plants:
                    random_plant = self.bee.random.choice(available_plants)
                    self.bee.travel_route = [random_plant.pos]
                    
                    dx = random_plant.pos[0] - self.bee.pos[0]
                    dy = random_plant.pos[1] - self.bee.pos[1]
                    theta_to_flower = math.degrees(math.atan2(dy, dx))
                    theta_to_flower = (theta_to_flower + 180) % 360 - 180
                    self.bee.theta = theta_to_flower

                    next_coord = self.bee.travel_route.pop(0)
                    next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                    self.bee.model.space.move_agent(self.bee, next_coord)
                    return                    
                          
                else:  # Continue down the path
                    next_coord = self.bee.travel_route.pop(0)
                    next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                    self.bee.model.space.move_agent(self.bee, next_coord)
   
        # 1.2 yes, currently still foraging
        elif self.bee.on_plant is not None or self.bee.forage_timer > 0:
            self.bee.forage_timer -= 1
            
            if self.bee.forage_timer == 0:  # Finished foraging
                # Update the plant nectar amount and bee stomach content
                self.bee.on_plant.nectar -= self.bee.nectar_to_collect
                self.bee.add_nectar(self.bee.nectar_to_collect)
                
                # Clear temp collection parameters
                self.bee.nectar_to_collect = 0.0
                
                # Wrap up interaction with flower
                self.bee.on_plant.bee_on = None
                self.bee.on_plant.leave_scent()
                self.bee.on_plant = None
                
                # If the bee stomach is full of nectar, then start heading home,
                # otherwise take new action
                if self.bee.is_full():
                    self.bee.homing = True
                    bee_path = LineString([self.bee.pos, self.bee.beehive.pos])
                    self.bee.travel_route = self.sample_line(bee_path, 0.7)
                    dx = self.bee.beehive.pos[0] - self.bee.pos[0]
                    dy = self.bee.beehive.pos[1] - self.bee.pos[1]
                    theta = math.degrees(math.atan2(dy, dx))
                    if theta > 180: theta -= 360
                    elif theta < -180: theta += 360
                    self.bee.theta = theta
                else:
                    self.new_action_and_move()
        else:
            assert False, "Defensive programming, this should not be executed"

    # --- NEW nectar-based reward memory ---
    def update_reward_memory(self, bee, nectar_collected: float):
        """
        Nectar-based reward memory:
        - If this collection > previous -> append 1
        - If this collection == previous -> append 1  (tie is success)
        - If this collection < previous -> append 0
        - Until a comparison can be made (first valid collection), append 1
        """
        if self.last_nectar_collected is None:
            reward = 1
        else:
            reward = 1 if nectar_collected >= self.last_nectar_collected else 0

        bee.reward_memory.append(reward)
        self.last_nectar_collected = nectar_collected

    def get_new_action(self):
        """
        This function will update bee's travel_route. If there is a flower nearby, randomly choose one and
        update travel_route as [plant position]. Otherwise, a new direction and distance is sampled and travel_route
        is updated accordingly.
        """
        # 8. Are any nearby flowers not in the visited list?
        available_plants = self.get_available_plants(vision_range=self.bee.vision_range)

        # There is no plants nearby, decide whether to do long jump or not. 
        if not available_plants:
            if len(self.bee.reward_memory) < self.bee.reward_size:
                self.bee.travel_route, self.bee.is_long_jump = self.sample_from_biased_distribution(max_jump=self.LONG_JUMP_THRESHOLD)
            else:
                long_jump_rate = sum(self.bee.reward_memory) / self.bee.reward_size
                if self.bee.random.random() > long_jump_rate:
                    self.bee.travel_route, self.bee.is_long_jump = self.sample_from_biased_distribution()
                else:
                    self.bee.travel_route, self.bee.is_long_jump = self.sample_from_biased_distribution(max_jump=self.LONG_JUMP_THRESHOLD)
        
        # There are available plants nearby
        else:
            random_plant = self.bee.random.choice(available_plants)
            self.bee.travel_route = [random_plant.pos]

            dx = random_plant.pos[0] - self.bee.pos[0]
            dy = random_plant.pos[1] - self.bee.pos[1]
            theta_to_flower = math.degrees(math.atan2(dy, dx))

            if theta_to_flower > 180:
                theta_to_flower -= 360
            elif theta_to_flower < -180:
                theta_to_flower += 360

            self.bee.theta = theta_to_flower

    def sample_from_biased_distribution(self, max_jump=None):
        ""
        # TODO: Move the biased turn angle function here
        delta_theta = self.biased_distribution(self.bee.random, 1)[0]

        ########### Debugging and Verification ###########
        self.bee.turning_angles.append(delta_theta)
        ##################################################
        
        new_theta = (self.bee.theta + delta_theta) % 360
        if new_theta > 180:
            new_theta -= 360
        self.bee.theta = new_theta

        # Replace uniform sampling with Truncated Lévy sampling
        alpha = 1.5 
        min_distance = 0.7
        max_distance = max_jump if max_jump is not None else 20.0  # set a cap if none given

        # Truncated Lévy sampling logic (same as in TruncatedLevyBehavior)
        while True:
            u = self.bee.random.random()
            distance = min_distance * ((1 - u) ** (-1 / alpha))
            if distance <= max_distance:
                break

        # Special case: force long jump if permitted but undershoots threshold
        if max_jump is None and distance < self.LONG_JUMP_THRESHOLD:
            distance += self.LONG_JUMP_THRESHOLD

        is_long_jump = distance > self.LONG_JUMP_THRESHOLD

        ########### Debugging and Verification ###########
        self.bee.jump_distances.append(distance)
        ##################################################
        
        destination_coord = self.compute_endpoint(self.bee.pos[0], self.bee.pos[1], distance, new_theta)
        bee_path = LineString([self.bee.pos, destination_coord])
        
        if self.bee.model.space.out_of_bounds(pos=destination_coord):
            bee_path = self.handle_wall_collisions(bee_path)

        return self.sample_line(bee_path, 0.7), is_long_jump

    def get_plant_below(self):
        """
        Return a list that contains the plant below the bee that is available.
        
        A plant is avaiable when there is no bee on the plant, and plant has no scent.
        """
        agents_below = self.bee.model.space.get_neighbors(self.bee.pos, 0, include_center=True)
        plant_agent = [agent for agent in agents_below if isinstance(agent, Plant)]
        return plant_agent

    def get_available_plants(self, vision_range=0.7):
        """
        Get a list of available plants that the bee can land on.
        
        A plant is available if:
        - It is within the bee's vision range
        - It does not have a bee on it (bee_on is None)
        - It is not in the bee's latest visited flowers list
        
        Note: it will still fly towards a flower even if there is a scent mark on it.
        """
        neighbors = self.bee.model.space.get_neighbors(self.bee.pos, vision_range, include_center=True)
        plant_agents = [agent for agent in neighbors if isinstance(agent, Plant)]
        available_plants = [plant for plant in plant_agents if plant.bee_on is None and plant not in self.bee.latest_visited_flowers]
        return available_plants
    
    def land_on_flower(self, bee, plant):
        """ 
        Take in the plant object and check if the bee can land on it or not.
        """
        return bee.random.random() < self.landing_prob and plant.has_scent() is False

    def new_action_and_move(self):
        self.get_new_action()
        if self.bee.travel_route:
            next_coord = self.bee.travel_route.pop(0)
            next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
            self.bee.model.space.move_agent(self.bee, next_coord)
            

class NearestFlower(BeeBehavior):
    """
    MyBeeBehavior variant:
    - Scanning picks the nearest available flower (not in latest_visited_flowers, regardless of scent)
    - Landing requires probability and NO scent (unchanged)
    - Keeps reward memory + biased sampler from MyBeeBehavior
    - CHANGED: Only record a flower in latest_visited_* after a SUCCESSFUL landing
    """

    def __init__(self, landing_prob, biased_distribution=None):
        super().__init__(landing_prob=landing_prob)
        self.START = 95
        self.END = 140
        self.BEE_IDS = [5]
        self.biased_distribution = biased_distribution

    def step(self):
        # 0. Homing
        if self.bee.homing:
            if not self.bee.travel_route:
                self.bee.homing = False
                self.bee.nectar_in_stomach = 0.0
                self.bee.reward_memory.clear()
                self.bee.latest_visited_flowers.clear()
                self.bee.latest_visited_flowers_steps.clear()
                self.theta = None
                self.bee.last_sugar_concentration = None
                self.new_action_and_move()
                return
            else:
                next_coord = self.bee.travel_route.pop(0)
                next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                self.bee.model.space.move_agent(self.bee, next_coord)
                return

        # 1. Not currently foraging
        if self.bee.forage_timer == 0 and self.bee.on_plant is None:
            if not self.bee.travel_route:
                # Finished a long jump → clear cycle & pick again
                if self.bee.is_long_jump:
                    self.bee.is_long_jump = False
                    self.bee.reward_memory.clear()
                    self.new_action_and_move()
                    return

                # Check plant directly underfoot (exact coord, as you prefer)
                plant_below_list = self.get_plant_below()
                if plant_below_list:
                    plant = plant_below_list[0]

                    # NOTE: CHANGED — do NOT push to latest_visited_* yet.
                    # We only append after a successful landing.

                    if plant.bee_on is None and not plant.has_scent():
                        # Try landing
                        landing = self.land_on_flower(self.bee, plant)
                        if landing:
                            # Successful landing → NOW record visit history (CHANGED)
                            self.bee.on_plant = plant
                            plant.bee_on = self.bee
                            plant.visit_timestamp.append(self.bee.model.steps)

                            if hasattr(self.bee, "latest_visited_flowers"):
                                self.bee.latest_visited_flowers.append(plant)
                            if hasattr(self.bee, "latest_visited_flowers_steps"):
                                self.bee.latest_visited_flowers_steps.append(self.bee.model.steps)

                            nectar_to_collect = min(
                                self.bee.collection_rate,
                                plant.nectar,
                                self.bee.stomach_capacity - self.bee.nectar_in_stomach
                            )

                            if nectar_to_collect > 0:
                                self.bee.forage_timer = self.bee.forage_timer_max
                                self.bee.nectar_to_collect = nectar_to_collect
                                # reward memory as in MyBeeBehavior (sugar-based)
                                self.update_reward_memory(self.bee)
                                self.new_action_and_move()
                            elif nectar_to_collect == 0:
                                self.bee.reward_memory.append(0)
                                plant.bee_on = None
                                plant.leave_scent()
                                self.bee.on_plant = None
                                self.new_action_and_move()
                            else:
                                raise ValueError("Nectar to collect should not be negative.")
                        else:
                            # Probabilistic miss → move on (no history recorded)
                            self.new_action_and_move()

                    elif plant.bee_on and not plant.has_scent():
                        # same as your logic
                        self.bee.reward_memory.append(1)
                        self.new_action_and_move()

                    elif plant.has_scent():
                        self.bee.reward_memory.append(0)
                        self.new_action_and_move()

                    else:
                        assert False, "Unexpected landing-state combination."

                else:
                    self.bee.reward_memory.append(0)
                    self.new_action_and_move()

            else:
                # En route: if not long-jumping and plants in range, choose the NEAREST one
                available_plants = self.get_available_plants(vision_range=self.bee.vision_range)
                if not self.bee.is_long_jump and available_plants:
                    target = self.nearest_plant(available_plants)
                    self.bee.travel_route = [target.pos]

                    dx = target.pos[0] - self.bee.pos[0]
                    dy = target.pos[1] - self.bee.pos[1]
                    theta_to_flower = math.degrees(math.atan2(dy, dx))
                    theta_to_flower = (theta_to_flower + 180) % 360 - 180
                    self.bee.theta = theta_to_flower

                    next_coord = self.bee.travel_route.pop(0)
                    next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                    self.bee.model.space.move_agent(self.bee, next_coord)
                    return
                else:
                    next_coord = self.bee.travel_route.pop(0)
                    next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                    self.bee.model.space.move_agent(self.bee, next_coord)

        # 2. Foraging countdown
        elif self.bee.on_plant is not None or self.bee.forage_timer > 0:
            self.bee.forage_timer -= 1
            if self.bee.forage_timer == 0:
                self.bee.on_plant.nectar -= self.bee.nectar_to_collect
                self.bee.add_nectar(self.bee.nectar_to_collect)
                self.bee.nectar_to_collect = 0.0

                self.bee.on_plant.bee_on = None
                self.bee.on_plant.leave_scent()
                self.bee.on_plant = None

                if self.bee.is_full():
                    self.bee.homing = True
                    bee_path = LineString([self.bee.pos, self.bee.beehive.pos])
                    self.bee.travel_route = self.sample_line(bee_path, 0.7)
                    dx = self.bee.beehive.pos[0] - self.bee.pos[0]
                    dy = self.bee.beehive.pos[1] - self.bee.pos[1]
                    theta = math.degrees(math.atan2(dy, dx))
                    if theta > 180: theta -= 360
                    elif theta < -180: theta += 360
                    self.bee.theta = theta
                else:
                    self.new_action_and_move()
        else:
            assert False, "Defensive programming, this should not be executed"

    # --- keep MyBeeBehavior’s sugar-based reward memory ---
    def update_reward_memory(self, bee):
        current_sugar = bee.on_plant.sugar_concentration
        last_sugar = bee.last_sugar_concentration
        if last_sugar is None:
            reward = 1
        else:
            if current_sugar >= last_sugar:
                reward = 1
            else:
                drop = (last_sugar - current_sugar) / last_sugar
                reward = max(0, 1 - drop)
        bee.reward_memory.append(reward)
        bee.last_sugar_concentration = current_sugar

    def get_new_action(self):
        # Same as MyBeeBehavior, except: choose nearest instead of random when plants available
        available_plants = self.get_available_plants(vision_range=self.bee.vision_range)

        if not available_plants:
            if len(self.bee.reward_memory) < self.bee.reward_size:
                self.bee.travel_route, self.bee.is_long_jump = self.sample_from_biased_distribution(max_jump=self.LONG_JUMP_THRESHOLD)
            else:
                long_jump_rate = sum(self.bee.reward_memory) / self.bee.reward_size
                if self.bee.random.random() > long_jump_rate:
                    self.bee.travel_route, self.bee.is_long_jump = self.sample_from_biased_distribution()
                else:
                    self.bee.travel_route, self.bee.is_long_jump = self.sample_from_biased_distribution(max_jump=self.LONG_JUMP_THRESHOLD)
        else:
            target = self.nearest_plant(available_plants)  # CHANGED (nearest)
            self.bee.travel_route = [target.pos]

            dx = target.pos[0] - self.bee.pos[0]
            dy = target.pos[1] - self.bee.pos[1]
            theta_to_flower = math.degrees(math.atan2(dy, dx))
            if theta_to_flower > 180:
                theta_to_flower -= 360
            elif theta_to_flower < -180:
                theta_to_flower += 360
            self.bee.theta = theta_to_flower

    def sample_from_biased_distribution(self, max_jump=None):
        delta_theta = self.biased_distribution(self.bee.random, 1)[0]
        self.bee.turning_angles.append(delta_theta)

        new_theta = (self.bee.theta + delta_theta) % 360
        if new_theta > 180:
            new_theta -= 360
        self.bee.theta = new_theta

        alpha = 1.5
        min_distance = 0.7
        max_distance = max_jump if max_jump is not None else 20.0
        while True:
            u = self.bee.random.random()
            distance = min_distance * ((1 - u) ** (-1 / alpha))
            if distance <= max_distance:
                break
        if max_jump is None and distance < self.LONG_JUMP_THRESHOLD:
            distance += self.LONG_JUMP_THRESHOLD

        is_long_jump = distance > self.LONG_JUMP_THRESHOLD
        self.bee.jump_distances.append(distance)

        destination_coord = self.compute_endpoint(self.bee.pos[0], self.bee.pos[1], distance, new_theta)
        bee_path = LineString([self.bee.pos, destination_coord])
        if self.bee.model.space.out_of_bounds(pos=destination_coord):
            bee_path = self.handle_wall_collisions(bee_path)
        return self.sample_line(bee_path, 0.7), is_long_jump

    def get_plant_below(self):
        agents_below = self.bee.model.space.get_neighbors(self.bee.pos, 0, include_center=True)
        return [a for a in agents_below if isinstance(a, Plant)]

    def get_available_plants(self, vision_range=0.7):
        neighbors = self.bee.model.space.get_neighbors(self.bee.pos, vision_range, include_center=True)
        plant_agents = [agent for agent in neighbors if isinstance(agent, Plant)]
        available_plants = [plant for plant in plant_agents if plant.bee_on is None and plant not in self.bee.latest_visited_flowers]
        return available_plants

    def land_on_flower(self, bee, plant):
        return bee.random.random() < self.landing_prob and plant.has_scent() is False

    def new_action_and_move(self):
        self.get_new_action()
        if self.bee.travel_route:
            next_coord = self.bee.travel_route.pop(0)
            next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
            self.bee.model.space.move_agent(self.bee, next_coord)

    def nearest_plant(self, plants, tol=1e-6):
        bx, by = self.bee.pos
        # squared distances
        d2_list = []
        for p in plants:
            px, py = p.pos
            d2 = (px - bx) * (px - bx) + (py - by) * (py - by)
            d2_list.append(d2)
        if not d2_list:
            return None
        min_d2 = min(d2_list)
        tied = [p for p, d2 in zip(plants, d2_list) if abs(d2 - min_d2) <= tol]
        return self.bee.random.choice(tied)


class LevyFlight(BeeBehavior):
    """
    Baseline behavior with scent-aware targeting:
    - Movement: TruncatedLevyBehavior logic, but ignores scented plants when targeting.
    - Plants/foraging/homing: MyBeeBehavior mechanics (scent on interaction, 3-step foraging, homing when full).
    - Memory: none (no reward_memory, no visited-flowers history).
    """

    def __init__(self, landing_prob, biased_distribution=None, **kwargs):
        # Accept biased_distribution and extra kwargs for API compatibility, but do not use them.
        super().__init__(landing_prob=landing_prob)

    def step(self):
        # 0) Homing
        if getattr(self.bee, "homing", False):
            if not self.bee.travel_route:
                self.bee.homing = False
                if hasattr(self.bee, "nectar_in_stomach"):
                    self.bee.nectar_in_stomach = 0.0
                self.bee.theta = None
                if hasattr(self.bee, "last_sugar_concentration"):
                    self.bee.last_sugar_concentration = None
                self.new_action_and_move()
                return
            else:
                next_coord = self.bee.travel_route.pop(0)
                next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                self.bee.model.space.move_agent(self.bee, next_coord)
                return

        # 1) Not foraging currently
        if self.bee.forage_timer == 0 and self.bee.on_plant is None:
            if not self.bee.travel_route:
                # Finished a long jump?
                if getattr(self.bee, "is_long_jump", False):
                    self.bee.is_long_jump = False
                    self.new_action_and_move()
                    return

                # Check plant underfoot
                under = self.get_plant_below()
                if under:
                    plant = under[0]
                    if plant.bee_on is None and not plant.has_scent():
                        if self.land_on_flower(self.bee, plant):
                            self.bee.on_plant = plant
                            plant.bee_on = self.bee
                            plant.visit_timestamp.append(self.bee.model.steps)

                            collection_rate = getattr(self.bee, "collection_rate", 3)
                            capacity = getattr(self.bee, "stomach_capacity", 20.0)
                            stomach = getattr(self.bee, "nectar_in_stomach", 0.0)
                            nectar_to_collect = min(collection_rate, plant.nectar, max(0.0, capacity - stomach))

                            if nectar_to_collect > 0:
                                self.bee.forage_timer = self.bee.forage_timer_max
                                self.bee.nectar_to_collect = nectar_to_collect
                                self.new_action_and_move()
                            else:
                                plant.leave_scent()
                                plant.bee_on = None
                                self.bee.on_plant = None
                                self.new_action_and_move()
                        else:
                            self.new_action_and_move()
                    else:
                        self.new_action_and_move()
                else:
                    self.new_action_and_move()
            else:
                # Mid-route snap to unscented & unoccupied plant if not long-jumping
                available = self.get_available_plants(self.bee.vision_range)
                if not getattr(self.bee, "is_long_jump", False) and available:
                    target = self.bee.random.choice(available)
                    self.bee.travel_route = [target.pos]
                    dx = target.pos[0] - self.bee.pos[0]
                    dy = target.pos[1] - self.bee.pos[1]
                    theta = math.degrees(math.atan2(dy, dx))
                    self.bee.theta = (theta + 180) % 360 - 180

                next_coord = self.bee.travel_route.pop(0)
                next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                self.bee.model.space.move_agent(self.bee, next_coord)

        # 2) Foraging
        elif self.bee.on_plant is not None or self.bee.forage_timer > 0:
            self.bee.forage_timer -= 1
            if self.bee.forage_timer == 0:
                amount = getattr(self.bee, "nectar_to_collect", 0.0)
                self.bee.on_plant.nectar -= amount
                if hasattr(self.bee, "add_nectar"):
                    self.bee.add_nectar(amount)
                self.bee.nectar_to_collect = 0.0

                self.bee.on_plant.leave_scent()
                self.bee.on_plant.bee_on = None
                self.bee.on_plant = None

                if hasattr(self.bee, "is_full") and self.bee.is_full():
                    self.bee.homing = True
                    bee_path = LineString([self.bee.pos, self.bee.beehive.pos])
                    self.bee.travel_route = self.sample_line(bee_path, 0.7)
                    dx = self.bee.beehive.pos[0] - self.bee.pos[0]
                    dy = self.bee.beehive.pos[1] - self.bee.pos[1]
                    theta = math.degrees(math.atan2(dy, dx))
                    self.bee.theta = (theta + 180) % 360 - 180
                else:
                    self.new_action_and_move()
        else:
            raise RuntimeError("LevyFlight.step(): unexpected state")


    def get_new_action(self):
        neighbors = self.bee.model.space.get_neighbors(
            pos=self.bee.pos, radius=self.bee.vision_range, include_center=True
        )
        plant_agents = [a for a in neighbors if isinstance(a, Plant)]
        available_plants = [p for p in plant_agents if (p.bee_on is None and not p.has_scent())]

        if not available_plants:
            self.bee.travel_route, self.bee.is_long_jump = self.levy_truncated_sample()
        else:
            target = self.bee.random.choice(available_plants)
            self.bee.travel_route = [target.pos]
            dx = target.pos[0] - self.bee.pos[0]
            dy = target.pos[1] - self.bee.pos[1]
            theta_to_flower = math.degrees(math.atan2(dy, dx))
            self.bee.theta = (theta_to_flower + 180) % 360 - 180


    def get_plant_below(self):
        agents_below = self.bee.model.space.get_neighbors(self.bee.pos, 0, include_center=True)
        return [a for a in agents_below if isinstance(a, Plant)]

    def get_available_plants(self, vision_range=0.7):
        neighbors = self.bee.model.space.get_neighbors(self.bee.pos, vision_range, include_center=True)
        plant_agents = [a for a in neighbors if isinstance(a, Plant)]
        return [p for p in plant_agents if (p.bee_on is None and not p.has_scent())]

    def land_on_flower(self, bee, plant):
        return bee.random.random() < self.landing_prob and not plant.has_scent()

    def new_action_and_move(self):
        self.get_new_action()
        if self.bee.travel_route:
            next_coord = self.bee.travel_route.pop(0)
            next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
            self.bee.model.space.move_agent(self.bee, next_coord)
