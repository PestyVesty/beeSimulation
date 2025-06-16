import mesa
import math
from shapely.geometry import LineString, Point
import numpy as np
from collections import deque

"""
Class for declaring bee and plant objects, and defining their behaviors

NOTE: use self.random to get a random seeded number


TODO: After new bee behavior is implemented and verified, do the following:
- [ ] Update the documentation in old behaviors
- [ ] Based on the new behavior, correct the old behaviors (for example, does not skip flowers on the way when doing long jumps)
"""

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
        self.my_id = my_id

    def color_code(self):
        return {"color": "brown", "edgecolors": "none"}


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
                self.get_new_directions()
            else:
                next_coord = self.bee.travel_route.pop(0)
                next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                self.bee.model.space.move_agent(self.bee, next_coord)

                if not self.bee.travel_route:
                    if self.bee.is_long_jump:
                        self.get_new_directions()
                        self.bee.is_long_jump = False
                        if hasattr(self, "reward_memory_queue"):
                            self.reward_memory_queue = []
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
                        if hasattr(self, "reward_memory_queue"):
                            self.reward_memory_queue.append(1)
                    else:
                        if hasattr(self, "reward_memory_queue"):
                            self.reward_memory_queue.append(0)
                        self.get_new_directions()

                    if hasattr(self, "reward_memory_queue") and hasattr(self, "reward_size"):
                        if len(self.reward_memory_queue) > self.reward_size:
                            self.reward_memory_queue.pop(0)
        else:
            self.bee.forage_timer -= 1
            if self.bee.forage_timer == 0:
                self.bee.last_plant = self.bee.on_plant
                self.bee.on_plant.bee_on = None
                self.bee.on_plant = None

    def get_new_directions(self):
        raise NotImplementedError("Subclasses must implement get_new_directions()")



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


class PureLevyBehavior(BeeBehavior):
    """
    Pure Levy flight behavior for bees. This behavior samples from a Levy distribution
    to determine the distance for the bee to travel. Direction is chosen uniformly.
    """
    def __init__(self, landing_prob):
        super().__init__(landing_prob=landing_prob)

    def get_new_directions(self):
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

    def get_new_directions(self):
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
    of making a short jump = sum(reward_memory_queue) / queue size (when queue size is full).
    
    """
    def __init__(self, landing_prob, reward_size=5):
        super().__init__(landing_prob=landing_prob)
        self.reward_size = reward_size
        self.reward_memory_queue = []

    def get_new_directions(self):
        neighbors = self.bee.model.space.get_neighbors(pos=self.bee.pos, radius=self.bee.vision_range, include_center=True)
        plant_agents = [agent for agent in neighbors if isinstance(agent, Plant)]
        available_plants = [plant for plant in plant_agents if plant.bee_on is None and plant != self.bee.last_plant]

        if available_plants == []:
            if len(self.reward_memory_queue) < self.reward_size:
                self.bee.travel_route, self.bee.is_long_jump = self.levy_truncated_sample(max_jump=self.LONG_JUMP_THRESHOLD)
            else:
                success_rate = sum(self.reward_memory_queue) / self.reward_size
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
        self.reward_memory_queue = []

        if biased_distribution is None:
            self.biased_distribution = lambda n=1: np.random.uniform(-180, 180, size=n)
        else:
            self.biased_distribution = biased_distribution

    def get_new_directions(self):
        neighbors = self.bee.model.space.get_neighbors(self.bee.pos, self.bee.vision_range, include_center=True)
        plant_agents = [agent for agent in neighbors if isinstance(agent, Plant)]
        available_plants = [plant for plant in plant_agents if plant.bee_on is None and plant != self.bee.last_plant]

        if not available_plants:
            if len(self.reward_memory_queue) < self.reward_size:
                self.bee.travel_route, self.bee.is_long_jump = self.sample_from_biased_distribution(max_jump=self.LONG_JUMP_THRESHOLD)
            else:
                success_rate = sum(self.reward_memory_queue) / self.reward_size
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
        self.found_flower_during_short_jump = False
        
    def step(self):
        self.behavior.step()

    def color_code(self):
        if self.beehive_id == 0:
            return {"color": "white", "edgecolors": "none"}
        elif self.beehive_id == 1:
            return {"color": "black", "edgecolors": "none"}


class Plant(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)
        self.agent_type = "plant"

        self.model = model
        self.pos = None
        self.bee_on = None

        self.visit_count = 0
        self.visit_timestamp = []

        self.visited = False
        self.visited_this_step = False

    def step(self):
        self.visited_this_step = self.bee_on is not None

    def color_code(self):
        return {"color": "blue", "edgecolors": "none"}

################################################################################
################################################################################
################################################################################
class MyBee(Bee):
    def __init__(self, model, behavior, beehive=None):
        super().__init__(model, behavior, beehive)
        self.last_visited_flowers = deque(maxlen=5)
        self.stomach_capacity = 10.0 # maximum nectar capacity of the bee's stomach
        self.nectar_in_stomach = 0.0 # current nectar in the bee's stomach
        self.homing = False          # whether the bee is currently homing back to the hive
        
        self.unsuccessful_visits = 0 # count visits to empty flowers
    def add_nectar(self, amount):
        """
        Add nectar to the stomach, without exceeding capacity
        """
        self.nectar_in_stomach = min(self.nectar_in_stomach + amount, self.stomach_capacity)
        
    def is_full(self):
        return self.nectar_in_stomach >= self.stomach_capacity
    
class MyPlant(Plant):
    def __init__(self, model):
        super().__init__(model)
        self.scent_mark_timer = 0
        self.nectar = 8.0 # So flower will be visited at most 3 times
        
    def has_scent(self):
        return self.scent_mark_timer > 0
    
    def leave_scent(self, duration=5):
        self.scent_mark_timer = duration
        
    def step(self):
        if self.scent_mark_timer > 0:
            self.scent_mark_timer -= 1


class MyBeeBehavior(BeeBehavior):
    """
    An extension of BiasedCompositeLevyBehavior. The differences:
    - [x] Homing behavior: Bee flies back to hive either: 
            1) the stomach is full
            2) unsuccessful visits reaches 15.
            After bee reaches the hive, it will be released back into the environment immediately.
    - [x] Flower memory: Each bee will have a memory of the last five flowers they have visited.
    - [x] Flower behavior: Set the flower has 8 micro liters of nectar. Each visits by
                           bee it will take 1 micro liters of nectar. Each visit time
                           is set to 3 steps.
    - [x] Scent marking: after a bee has successfully foraged, it will leave a decaying
                         scent mark. Other bees will be able to detect the scent mark when
                         they are above it and does not land on it. After 5 steps, the scent
                         mark will disappear.
    
    """
    def __init__(self, landing_prob, reward_size=5, biased_distribution=None):
        super().__init__(landing_prob=landing_prob)
        self.reward_size = reward_size
        self.reward_memory_queue = deque(maxlen=self.reward_size)

        if biased_distribution is None:
            self.biased_distribution = lambda n=1: np.random.uniform(-180, 180, size=n)
        else:
            self.biased_distribution = biased_distribution

    def step(self):
        
        # 0. Homing behavior: bee is returning to the hive
        if self.bee.homing:
            if not self.bee.travel_route:
                # Arrived at hive
                self.bee.homing = False
                self.bee.nectar_in_stomach = 0.0
                self.reward_memory_queue.clear()
                self.bee.unsuccessful_visits = 0
                self.bee.last_visited_flowers.clear()
                self.theta = self.bee.random.uniform(-180, 180)
                self.get_new_directions() # Bees resumes foraging
                return
            
            else:
                # Contue traveling toward hive
                next_coord = self.bee.travel_route.pop(0)
                next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                self.bee.model.space.move_agent(self.bee, next_coord)
                return
            
            
        # 1. Spending turns on current task?
        # 1.1 No
        if self.bee.forage_timer == 0:

            # 2. Have a set destination cell?
            # 2.1 No
            if not self.bee.travel_route:
                self.get_new_directions()

            # 2.2 Yes
            else:
                # 3. Move to next cell
                next_coord = self.bee.travel_route.pop(0)
                next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                self.bee.model.space.move_agent(self.bee, next_coord)

                # 5. Is it at the destination cell?
                # 5.1 Yes
                if not self.bee.travel_route:

                    # Check: Is the jump declared as a "long jump"? If so, get new directions
                    # and reset the reward memory queue.
                    # Bees do not stop for flowers when long jumping.
                    if self.bee.is_long_jump:
                        # After finishing a long jump, get next directions.
                        # Move to 8
                        self.get_new_directions()
                        self.bee.is_long_jump = False
                        self.reward_memory_queue = deque(maxlen=self.reward_size)
                        self.bee.found_flower_during_short_jump = False  # reset flag after jump completes
                        return

                    # Checkpoint: reaching here means the bee completed a short jump.
                    # Checking below if there is a flower below it land on.
                    below = self.bee.model.space.get_neighbors(self.bee.pos, 0, include_center=True)
                    plant_agents = [agent for agent in below if isinstance(agent, MyPlant)]
                    available_plants = [plant for plant in plant_agents if plant.bee_on is None and 
                                        plant not in self.bee.last_visited_flowers]

                    # 6. Land on flower below?
                    land_decision = self.land_on_flower(self.bee)
                    # 6.1 Yes
                    if available_plants and land_decision and not available_plants[0].has_scent() and available_plants[0].nectar > 0:
                        # 7. Land and harvest on flower
                        plant = available_plants[0]
                        self.bee.forage_timer = self.bee.forage_timer_max
                        self.bee.on_plant = plant
                        plant.bee_on = self.bee
                        plant.visit_count += 1
                        plant.visit_timestamp.append(self.bee.model.steps)
                        plant.visited = True
                        plant.visited_this_step = True
                        self.bee.last_visited_flowers.append(plant)
                        self.reward_memory_queue.append(1)
                        self.bee.found_flower_during_short_jump = False  # reset flag after landing
                    # 6.2 Yes, but flower has scent or no nectar
                    elif available_plants and land_decision and (available_plants[0].has_scent() or available_plants[0].nectar == 0):
                        self.reward_memory_queue.append(0)
                        self.bee.last_visited_flowers.append(available_plants[0])
                        self.get_new_directions()
                    else:
                        if not self.bee.found_flower_during_short_jump:
                            self.get_new_directions()
                        self.bee.found_flower_during_short_jump = False  # reset after reaching end of route

                # 5.2 No, but bee is scanning while moving during short jump
                else:
                    # Scan for flowers before continuing along short-jump route
                    if not self.bee.found_flower_during_short_jump:
                        neighbors = self.bee.model.space.get_neighbors(self.bee.pos, self.bee.vision_range, include_center=False)
                        plant_agents = [agent for agent in neighbors if isinstance(agent, MyPlant)]
                        available_plants = [p for p in plant_agents if p.bee_on is None and p not in self.bee.last_visited_flowers]

                        if available_plants:
                            # Redirect to the closest or random flower
                            target_flower = self.bee.random.choice(available_plants)
                            self.bee.travel_route = [target_flower.pos]  # Overwrite remaining route
                            self.bee.found_flower_during_short_jump = True  # Prevent switching again this trip

                            # Optional: update direction (theta) toward flower
                            dx = target_flower.pos[0] - self.bee.pos[0]
                            dy = target_flower.pos[1] - self.bee.pos[1]
                            theta = math.degrees(math.atan2(dy, dx))
                            if theta > 180:
                                theta -= 360
                            elif theta < -180:
                                theta += 360
                            self.bee.theta = theta
                    return

        # 1.2 Yes (still harvesting)
        else:
            self.bee.forage_timer -= 1
            if self.bee.forage_timer == 0:

                self.bee.last_plant = self.bee.on_plant
                
                # Add nectar to bee's stomach
                nectar_collected = min(3.0, self.bee.on_plant.nectar)  # Collect nectar from flower
                self.bee.add_nectar(nectar_collected)
                self.bee.on_plant.nectar -= nectar_collected

                # Leave scent mark on the flower
                self.bee.on_plant.leave_scent(duration=5)

                self.bee.on_plant.bee_on = None
                self.bee.on_plant = None
                
                # If no nectar was collected, increment unsuccessful visits
                if nectar_collected == 0:
                    self.bee.unsuccessful_visits += 1
                
                # if stomach is full, bee will return to the hive
                # Or, if visited 15 empty flowers, bee will return to the hive
                if self.bee.is_full() or self.bee.unsuccessful_visits >= 15:
                    self.bee.homing = True
                    self.bee.travel_route = [self.bee.beehive.pos]
                    
                    dx = self.bee.beehive.pos[0] - self.bee.pos[0]
                    dy = self.bee.beehive.pos[1] - self.bee.pos[1]
                    theta = math.degrees(math.atan2(dy, dx))
                    if theta > 180:
                        theta -= 360
                    elif theta < -180:
                        theta += 360
                    self.bee.theta = theta
                
    def get_new_directions(self):
        # 8. Are any nearby flowers unvisited targets?
        neighbors = self.bee.model.space.get_neighbors(self.bee.pos, self.bee.vision_range, include_center=False)
        plant_agents = [agent for agent in neighbors if isinstance(agent, MyPlant)]
        available_plants = [plant for plant in plant_agents if plant.bee_on is None and plant not in self.bee.last_visited_flowers]
        
        # 8.1 No, leads to 9
        if not available_plants:
            if len(self.reward_memory_queue) < self.reward_size:
                self.bee.travel_route, self.bee.is_long_jump = self.sample_from_biased_distribution(max_jump=self.LONG_JUMP_THRESHOLD)
            else:
                success_rate = sum(self.reward_memory_queue) / self.reward_size
                if self.bee.random.random() > success_rate:
                    self.bee.travel_route, self.bee.is_long_jump = self.sample_from_biased_distribution()
                else:
                    self.bee.travel_route, self.bee.is_long_jump = self.sample_from_biased_distribution(max_jump=self.LONG_JUMP_THRESHOLD)
        # 8.2 Yes, leads to 10
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

        # Replace uniform sampling with Truncated Lévy sampling
        alpha = 1.5  # Or whatever parameter you were using
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

        destination_coord = self.compute_endpoint(self.bee.pos[0], self.bee.pos[1], distance, new_theta)
        bee_path = LineString([self.bee.pos, destination_coord])

        if self.bee.model.space.out_of_bounds(pos=destination_coord):
            bee_path = self.handle_wall_collisions(bee_path)

        return self.sample_line(bee_path, 0.7), is_long_jump