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
        self.my_id = my_id # TODO: Check if this is needed, or if it should be removed (creating new agents in mesa should just assign a unique ID). If so, the go back to model.py and edit the code.

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
                self.get_new_action()
            else:
                next_coord = self.bee.travel_route.pop(0)
                next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                self.bee.model.space.move_agent(self.bee, next_coord)

                if not self.bee.travel_route:
                    if self.bee.is_long_jump:
                        self.get_new_action()
                        self.bee.is_long_jump = False
                        if hasattr(self, "reward_memory"):
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
                        if hasattr(self, "reward_memory"):
                            self.bee.reward_memory.append(1)
                    else:
                        if hasattr(self, "reward_memory"):
                            self.bee.reward_memory.append(0)
                        self.get_new_action()

                    if hasattr(self, "reward_memory") and hasattr(self, "reward_size"):
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

        self.have_target = False
        self.target_flower = None
        
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
def debug_bee(msg, bee):
    uid = getattr(bee, "unique_id", None)
    if uid in (5, 38) and 15 <= bee.model.steps <= 30:
        print(f"[t={bee.model.steps}][Bee {uid}] {msg}")

class MyBee(Bee):
    def __init__(self, model, behavior, beehive=None):
        super().__init__(model, behavior, beehive)
        self.reward_size = 5
        self.reward_memory = deque(maxlen=self.reward_size)
        self.latest_visited_flowers = deque(maxlen=5)
        self.stomach_capacity = 20.0
        self.nectar_in_stomach = 0.0
        self.homing = False
        self.collection_rate = 3
        self.nectar_to_collect = 0.0
        self.target_plant = None
        self.nectar_reward_threshold = 1
        self.turning_angles = []
        self.jump_distances = []
        
        self.is_long_jump = False
    def add_nectar(self, amount):
        self.nectar_in_stomach = min(self.nectar_in_stomach + amount, self.stomach_capacity)

    def is_full(self):
        return self.nectar_in_stomach >= self.stomach_capacity

class MyPlant(Plant):
    def __init__(self, model):
        super().__init__(model)
        self.scent_mark_timer = 0
        self.nectar = 13.0
        self.max_nectar = 13.0
        self.replenish_rate = 0.25
        self.reserved_by = None
        
        self.START = 10
        self.END = 30

    def has_scent(self):
        return self.scent_mark_timer > 0

    def leave_scent(self, duration=45):
        self.scent_mark_timer = duration

    def step(self):
        # current_step = getattr(self.model, "steps", None)

        # # Print out bee_on if bee 5 or 38 is on this plant between step 15 and 30
        # if (
        #     self.bee_on is not None 
        #     and getattr(self.bee_on, "unique_id", None) in (5, 38) 
        #     and 15 <= current_step <= 30
        # ):
        #     print(
        #         f"[PLANT {self.unique_id}][step={current_step}] bee_on={self.bee_on.unique_id}, "
        #         f"bee_on.on_plant={self.bee_on.on_plant}, nectar={self.nectar}"
        #     )
        #     # Optionally check for inconsistencies
        #     if self.bee_on.on_plant != self:
        #         print(
        #             f"-> Inconsistency: Plant {self.unique_id} thinks bee {self.bee_on.unique_id} is here, but bee does not!"
        #         )
        #         # Optionally heal
        #         self.bee_on = None

        
        if self.model.steps in range(self.START, self.END) and self.pos == (13.0, 2.85):
            if self.bee_on:
                print(f"Plant at {self.pos} has bee {self.bee_on.unique_id} at step {self.model.steps -1}")
        
        
        # Usual scent and nectar management
        if self.scent_mark_timer > 0:
            self.scent_mark_timer -= 1

        if self.nectar < self.max_nectar:
            self.nectar = min(self.nectar + self.replenish_rate, self.max_nectar)

class MyBeeBehavior(BeeBehavior):
    """
    An extension of BiasedCompositeLevyBehavior. The differences:
    - [ ] Homing behavior: Bee flies back to hive either: 
            - [x] 1) the stomach is full
            - [ ] 2) unsuccessful visits reaches 15.
            After bee reaches the hive, wait for 5 steps before releasing it back
            into the environment as a "new" bee.
    - [x] Flower memory: Each bee will have a memory of the last five flowers they have visited.
    - [x] Flower behavior: Set the flower has 3 micro liters of nectar. Each visits by
                           bee it will take 1 micro liters of nectar. Each visit time
                           is set to 3 steps.
    - [x] Scent marking: after a bee has successfully foraged, it will leave a decaying
                         scent mark. Other bees will be able to detect the scent mark when
                         they are above it and does not land on it. After 5 steps, the scent
                         mark will disappear.
                         
    What I expect when running this behavior in an empty environment: the bee will perform five
    short jumps before making one long jump. After that, it will repeat the process.
    
    """
    def __init__(self, landing_prob, biased_distribution=None):
        super().__init__(landing_prob=landing_prob)
        
        self.START = 95
        self.END = 140
        
        # self.START = 65
        # self.END = 80
        
        self.BEE_IDS = [5]
        
        if biased_distribution is None:
            self.biased_distribution = lambda n=1: np.random.uniform(-180, 180, size=n)
        else:
            self.biased_distribution = biased_distribution

    def step(self):
        
        # #! between steps START and END, print states for bee with unique_id
        # if self.bee.model.steps in range(self.START, self.END) and self.bee.unique_id in self.BEE_IDS:
               
        #     current_step = self.bee.model.steps-1
        #     print(f"{current_step} ######################") 
        #     #! Position
        #     print(f"bee position {self.bee.pos}")
            
        #     # #! Is this a long jump
        #     print(f"Long jump: {self.bee.is_long_jump}")

        #     #! Printing plant below
        #     plant_below_list = self.get_plant_below()
        #     if plant_below_list:
        #         plant = plant_below_list[0]
        #         # print(f"{current_step} bee {self.bee.unique_id} found plant at {plant.pos}, bee on {self.bee.on_plant.bee_on.unique_id}")                
        #         print(f"!!! bee {self.bee.unique_id} found plant at {plant.pos}")                
            
        #     #! Current memory state
        #     print(f"bee visited memory: {self.bee.latest_visited_flowers}")

        # 0. Check if bee is returning to the hive
        if self.bee.homing:
            if not self.bee.travel_route:
                # Arrived at hive
                self.bee.homing = False
                self.bee.nectar_in_stomach = 0.0
                self.bee.reward_memory.clear()
                self.bee.latest_visited_flowers.clear()
                self.theta = None
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
                    
                    if plant.bee_on is None and not plant.has_scent():
                        
                        # Attempt to land on the flower
                        landing = self.land_on_flower(self.bee, plant)
                        
                        if landing:
                            self.bee.on_plant = plant
                            plant.bee_on = self.bee 
                            
                            nectar_to_collect = min(self.bee.collection_rate, plant.nectar, self.bee.stomach_capacity - self.bee.nectar_in_stomach)
                            
                            if nectar_to_collect > 0:
                                self.bee.forage_timer = self.bee.forage_timer_max
                                self.nectar_to_collect = nectar_to_collect
                                self.bee.reward_memory.append(1)
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
                    # elif plant.bee_on and not plant.has_scent(): # TODO: implement this defensive logic
                    elif plant.bee_on and not plant.has_scent(): 
                        self.bee.reward_memory.append(1)
                        self.new_action_and_move()
                    
                    # elif plant.has_scent() and not plant.bee_on: # TODO: implement this defensive logic 
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
                    print("Stomach is full")
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

        # #! End of the step
        # if self.bee.model.steps in range(self.START, self.END) and self.bee.unique_id in self.BEE_IDS:
        #     print(f"{current_step} end ######################")
        #     print(f"Travel route after moving: {self.bee.travel_route}")
        #     print(f"Bee_on status: {self.bee.on_plant}")
            
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
    
        if self.bee.model.steps in range(0, 23) and self.bee.unique_id == 5:
            print(f"Destination Coord {destination_coord}")
        
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
        
        #! Got new action
        if self.bee.model.steps in range(self.START, self.END) and self.bee.unique_id == 5:
            print(f"new route {self.bee.travel_route}")
        
        # #! New route
        # if self.bee.model.steps in range(18, 23) and self.bee.unique_id == 5:
        #     print(f"Step {self.bee.model.steps} new route {self.bee.travel_route}")
        if self.bee.travel_route:
            next_coord = self.bee.travel_route.pop(0)
            next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
            self.bee.model.space.move_agent(self.bee, next_coord)
        
        # #! updated new route        
        # if self.bee.model.steps in range(self.START, self.END) and self.bee.unique_id == 5:    
        #     print(f"Travel route after moving: {self.bee.travel_route}")

class MyBeeBehavior_v2_best(BeeBehavior):
    """
    ****** Fixed the logic problem, however redundant
    
    An extension of BiasedCompositeLevyBehavior. The differences:
    - [ ] Homing behavior: Bee flies back to hive either: 
            - [x] 1) the stomach is full
            - [ ] 2) unsuccessful visits reaches 15.
            After bee reaches the hive, wait for 5 steps before releasing it back
            into the environment as a "new" bee.
    - [x] Flower memory: Each bee will have a memory of the last five flowers they have visited.
    - [x] Flower behavior: Set the flower has 3 micro liters of nectar. Each visits by
                           bee it will take 1 micro liters of nectar. Each visit time
                           is set to 3 steps.
    - [x] Scent marking: after a bee has successfully foraged, it will leave a decaying
                         scent mark. Other bees will be able to detect the scent mark when
                         they are above it and does not land on it. After 5 steps, the scent
                         mark will disappear.
                         
    What I expect when running this behavior in an empty environment: the bee will perform five
    short jumps before making one long jump. After that, it will repeat the process.
    
    """
    def __init__(self, landing_prob, biased_distribution=None):
        super().__init__(landing_prob=landing_prob)
        
        self.START = 17
        self.END = 25
        
        # self.START = 65
        # self.END = 80
        
        self.BEE_IDS = [5]
        
        if biased_distribution is None:
            self.biased_distribution = lambda n=1: np.random.uniform(-180, 180, size=n)
        else:
            self.biased_distribution = biased_distribution

    # TODO: when arrived at destination, it should not be visited, update memory immediately
    # TODO: check when is memory updated in the code below
    def step(self):
        
        #! between steps START and END, print states for bee with unique_id
        if self.bee.model.steps in range(self.START, self.END) and self.bee.unique_id in self.BEE_IDS:
               
            current_step = self.bee.model.steps-1
            print(f"{current_step} ######################") 
            #! Position
            print(f"bee position {self.bee.pos}")
            
            # #! Is this a long jump
            # print(f"{current_step} is long jump {self.bee.is_long_jump}")

            #! Printing plant below
            plant_below_list = self.get_plant_below()
            if plant_below_list:
                plant = plant_below_list[0]
                # print(f"{current_step} bee {self.bee.unique_id} found plant at {plant.pos}, bee on {self.bee.on_plant.bee_on.unique_id}")                
                print(f"!!! bee {self.bee.unique_id} found plant at {plant.pos}")                
            
            #! Current memory state
            print(f"bee visited memory: {self.bee.latest_visited_flowers}")

        # 0. Check if bee is returning to the hive
        if self.bee.homing:
            if not self.bee.travel_route:
                # Arrived at hive
                self.bee.homing = False
                self.bee.nectar_in_stomach = 0.0
                self.bee.reward_memory.clear()
                self.bee.latest_visited_flowers.clear()
                self.theta = None
                self.new_action_and_move()
                return
            else:
                # Continue travel toward the hive
                next_coord = self.bee.travel_route.pop(0)
                next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                self.bee.model.space.move_agent(self.bee, next_coord)
                return

        # #! Print bee 5 forage_timer and on_plant
        # if self.bee.model.steps == 19 and self.bee.unique_id == 5:
        #     print(f"!!!!! forage_timer: {self.bee.forage_timer}, on_plant {self.bee.on_plant}" )
        #     # Pass
        
        # 1. Spending time on foraging nectar?
        # 1.1 No, then check if the bee is at the destination
        if self.bee.forage_timer == 0 and self.bee.on_plant is None:

            if self.bee.model.steps == 20 and self.bee.unique_id == 5:
                print(f"!!! Forager_timer is 0 and bee is not on plant, travel_route is: {self.bee.travel_route}")
            # 2. Is the bee at the set destination?
            # 2.1 Yes, empty travel_route means it is at the destination, check if there is a plant below it
            if not self.bee.travel_route: 
                
                plant_below_list = self.get_plant_below()
                # Check if there is a flower below
                if plant_below_list:
                    plant = plant_below_list[0]
                    self.bee.latest_visited_flowers.append(plant)
                    
                    if plant.bee_on is None and not plant.has_scent():
                        
                        # Attempt to land on the flower
                        landing = self.land_on_flower(self.bee, plant)
                        
                        if landing:
                            self.bee.on_plant = plant
                            plant.bee_on = self.bee 
                            
                            nectar_to_collect = min(self.bee.collection_rate, plant.nectar, self.bee.stomach_capacity - self.bee.nectar_in_stomach)
                            
                            if nectar_to_collect > 0:
                                self.bee.forage_timer = self.bee.forage_timer_max
                                self.nectar_to_collect = nectar_to_collect
                                self.bee.reward_memory.append(1)
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
                    # elif plant.bee_on and not plant.has_scent(): # TODO: implement this defensive logic
                    elif plant.bee_on and not plant.has_scent(): 
                        self.bee.reward_memory.append(1)
                        self.new_action_and_move()
                    
                    # elif plant.has_scent() and not plant.bee_on: # TODO: implement this defensive logic 
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
                # 3. Move to next location
                next_coord = self.bee.travel_route.pop(0)
                next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                self.bee.model.space.move_agent(self.bee, next_coord)

                # 5. Is it at the destination location?
                # 5.1 Yes
                if not self.bee.travel_route:
                    
                    # #! Or do they enter this first?
                    # if self.bee.model.steps in range(self.START, self.END) and self.bee.unique_id == 5:
                    #     print(f"{self.bee.model.steps} arrived at destination {self.bee.pos}")
                    #     print(f"Is long jump? {self.bee.is_long_jump}")
                        
                    # Check: Is the jump declared as a "long jump"? If so, get new directions
                    # and reset the reward memory queue. Bees do not stop for flowers when long jumping.
                    if self.bee.is_long_jump:
                        # After finishing a long jump, get next directions.
                        # Move to 8
                        self.bee.is_long_jump = False
                        self.bee.reward_memory.clear()
                        self.new_action_and_move()

                        return

                    # Checkpoint: reaching here means the bee completed a short jump.

                    # Check if there is a flower below
                    plant_below_list = self.get_plant_below()
                    
                    # #! Print list of plant below bee 5
                    # if self.bee.model.steps in range(self.START, self.END) and self.bee.unique_id == 5:
                    #     print(f"Step {self.bee.model.steps} plant below list: {plant_below_list}")
                        
                        
                    # There is a plant below the bee
                    if plant_below_list:
                        plant = plant_below_list[0]
                        self.bee.latest_visited_flowers.append(plant) # Add plant below to the memory
                        
                        #! Find when bee 5 lands on flower 5
                        # if self.bee.model.steps in range(self.START, self.END) and self.bee.unique_id == 5:
                        #     print(f"landing")
                        
                        # Check if can land on the flower
                        # Can land on flower: there is no bee on flower and flower has no scent
                        if plant.bee_on is None and not plant.has_scent():
                            
                            # Attempt to land on the flower
                            landing = self.land_on_flower(self.bee, plant)
                            
                            # Determine whether to land based on probability
                            if landing:
                                # Set bee_on and on_plant
                                self.bee.on_plant = plant
                                plant.bee_on = self.bee
                                
                                # self.bee.latest_visited_flowers.append(self.bee.on_plant)
                                
                                # TODO: Following parameters can be removed
                                plant.visit_count += 1
                                plant.visited = True  
                                plant.visited_this_step = True
                                plant.visit_timestamp.append(self.bee.model.steps)
                                                                
                                # Calculate the remaining amount of nectar to collect for the following fixed amount of timesteps
                                nectar_to_collect = min(self.bee.collection_rate, plant.nectar, self.bee.stomach_capacity - self.bee.nectar_in_stomach)
                                
                                # When there is nectar, regardless of amount  
                                if nectar_to_collect > 0: # TODO: update this to threshold, change foraging logic
                                    self.bee.forage_timer = self.bee.forage_timer_max
                                    self.nectar_to_collect = nectar_to_collect
                                    self.bee.reward_memory.append(1)
                                    self.new_action_and_move()

                                # There is no nectar
                                elif nectar_to_collect == 0: # TODO: update this to threshold, change foraging logic
                                    # Unsuccessful visit
                                    self.bee.reward_memory.append(0)

                                    # Clean up and leave
                                    plant.bee_on = None
                                    plant.leave_scent()
                                    self.bee.on_plant = None

                                    # Take new action
                                    self.new_action_and_move()

                                # Defensive programming
                                else:
                                    raise ValueError("Nectar to collect should not be negative")
                            
                            # Bee decided not to land on the plant based on probability
                            # Add the flower tot he latest_visited_flowers
                            # NOTE: Reward memory is not updated when not landing
                            elif not landing:
                                # self.bee.latest_visited_flowers.append(plant)
                                self.new_action_and_move()
                                
                            # Defensive programming
                            else:
                                raise ValueError("Landing logic error")
                            
                        # Did not land on the flower because there is a bee
                        elif plant.bee_on:
                            self.bee.reward_memory.append(1)
                            self.new_action_and_move()
                        
                        # Did not land on flower because there is scent detected
                        elif plant.has_scent():
                            self.bee.reward_memory.append(0)
                            self.new_action_and_move()
                        
                        # Defensive programming                            
                        else: 
                            assert False, "Defensive programming 0, different reason why bee cannot land on flower"

                    # There is no flower below bee
                    elif not plant_below_list: 
                        self.bee.reward_memory.append(0)
                        self.new_action_and_move()
                    else:
                        assert False, "Plant below detection problem"
                        
                # 5.2 No, not at destined location, and bee is scanning while moving during short jump
                else:
                    # If not a long jump, scan
                    if not self.bee.is_long_jump:
                        available_plants = self.get_available_plants(vision_range=self.bee.vision_range)
                        if available_plants:
                            random_plant = self.bee.random.choice(available_plants)
                            self.bee.travel_route = [random_plant.pos]
                            
                            dx = random_plant.pos[0] - self.bee.pos[0]
                            dy = random_plant.pos[1] - self.bee.pos[1]
                            theta_to_flower = math.degrees(math.atan2(dy, dx))
                            theta_to_flower = (theta_to_flower + 180) % 360 - 180
                            self.bee.theta = theta_to_flower
                    # else, no flowers nearby, continue with the current direction
                    return
                
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

        #! End of the step
        if self.bee.model.steps in range(self.START, self.END) and self.bee.unique_id in self.BEE_IDS:
            print(f"{current_step} end ######################")
            print(f"Travel route after moving: {self.bee.travel_route}")
            print(f"Bee_on status: {self.bee.on_plant}")
            
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
    
        if self.bee.model.steps in range(0, 23) and self.bee.unique_id == 5:
            print(f"Destination Coord {destination_coord}")
        
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
        
        #! Got new action
        if self.bee.model.steps in range(self.START, self.END) and self.bee.unique_id == 5:
            print(f"new route {self.bee.travel_route}")
        
        # #! New route
        # if self.bee.model.steps in range(18, 23) and self.bee.unique_id == 5:
        #     print(f"Step {self.bee.model.steps} new route {self.bee.travel_route}")
        if self.bee.travel_route:
            next_coord = self.bee.travel_route.pop(0)
            next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
            self.bee.model.space.move_agent(self.bee, next_coord)
        
        # #! updated new route        
        # if self.bee.model.steps in range(self.START, self.END) and self.bee.unique_id == 5:    
        #     print(f"Travel route after moving: {self.bee.travel_route}")

class MyBeeBehavior_v1_debugging(BeeBehavior):
    """
    An extension of BiasedCompositeLevyBehavior. The differences:
    - [ ] Homing behavior: Bee flies back to hive either: 
            - [x] 1) the stomach is full
            - [ ] 2) unsuccessful visits reaches 15.
            After bee reaches the hive, wait for 5 steps before releasing it back
            into the environment as a "new" bee.
    - [x] Flower memory: Each bee will have a memory of the last five flowers they have visited.
    - [x] Flower behavior: Set the flower has 3 micro liters of nectar. Each visits by
                           bee it will take 1 micro liters of nectar. Each visit time
                           is set to 3 steps.
    - [x] Scent marking: after a bee has successfully foraged, it will leave a decaying
                         scent mark. Other bees will be able to detect the scent mark when
                         they are above it and does not land on it. After 5 steps, the scent
                         mark will disappear.
                         
    What I expect when running this behavior in an empty environment: the bee will perform five
    short jumps before making one long jump. After that, it will repeat the process.
    
    """
    def __init__(self, landing_prob, biased_distribution=None):
        super().__init__(landing_prob=landing_prob)
        
        # self.START = 10
        # self.END = 30
        
        self.START = 65
        self.END = 80
        # self.BEE_IDS = [5]
        
        if biased_distribution is None:
            self.biased_distribution = lambda n=1: np.random.uniform(-180, 180, size=n)
        else:
            self.biased_distribution = biased_distribution

    # TODO: when arrived at destination, it should not be visited, update memory immediately
    # TODO: check when is memory updated in the code below
    def step(self):
        #!
        if self.bee.model.steps in range(self.START, self.END) and self.bee.unique_id == 5:
            print(f"{self.bee.model.steps} is long jump {self.bee.is_long_jump}")
        
        if self.bee.model.steps == 23 and self.bee.unique_id in (5, 38):
            plant_below_list = self.get_plant_below()
            if plant_below_list:
                plant = plant_below_list[0]
                print(f"{self.bee.model.steps} [BEE {self.bee.unique_id}] Found plant below at {plant.pos}, plant_id={plant.unique_id}, object_id={id(plant)}")
                if plant.bee_on is not None:
                    print(f"{self.bee.model.steps} [BEE {self.bee.unique_id}] plant.bee_on = BEE {plant.bee_on.unique_id}, object_id={id(plant.bee_on)}")
                else:
                    print(f"{self.bee.model.steps} [BEE {self.bee.unique_id}] plant.bee_on = None")
            else:
                print(f"{self.bee.model.steps} [BEE {self.bee.unique_id}] No plant detected below bee at pos {self.bee.pos}")
            
        # 0. Check if bee is returning to the hive
        if self.bee.homing:
            if not self.bee.travel_route:
                # Arrived at hive
                self.bee.homing = False
                self.bee.nectar_in_stomach = 0.0
                self.bee.reward_memory.clear()
                self.bee.latest_visited_flowers.clear()
                self.theta = None
                self.new_action_and_move()
                return
            else:
                # Continue travel toward the hive
                next_coord = self.bee.travel_route.pop(0)
                next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                self.bee.model.space.move_agent(self.bee, next_coord)
                return

        # #! Print bee 5 forage_timer and on_plant
        # if self.bee.model.steps in range(18, 23) and self.bee.unique_id == 5:
        #     print(f"Step {self.bee.model.steps}, forage_timer: {self.bee.forage_timer}, on_plant {self.bee.on_plant}" )

        # 1. Spending time on foraging nectar?
        # 1.1 No
        if self.bee.forage_timer == 0 and self.bee.on_plant is None:
            
            # #! Testing if ever entered
            # if self.bee.model.steps in range(18, 23) and self.bee.unique_id == 5:
            #     print("Yeet")
            # #! Yes does enter
            
            # 2. Have a set destination location?
            # 2.1 No, meaning the bee arrived at its destination or the bee is leaving the hive
            if not self.bee.travel_route: 
                #! Do they enter this first? 
                if self.bee.model.steps in range(self.START, self.END) and self.bee.unique_id == 5:
                    print(f"{self.bee.model.steps} got new action, new route {self.bee.travel_route}")
                self.new_action_and_move()
            # 2.2 Yes
            else:
                # 3. Move to next location
                next_coord = self.bee.travel_route.pop(0)
                next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                self.bee.model.space.move_agent(self.bee, next_coord)

                # 5. Is it at the destination location?
                # 5.1 Yes
                if not self.bee.travel_route:
                    
                    #! Or do they enter this first?
                    if self.bee.model.steps in range(self.START, self.END) and self.bee.unique_id == 5:
                        print(f"{self.bee.model.steps} arrived at destination {self.bee.pos}")
                        print(f"Is long jump? {self.bee.is_long_jump}")
                        
                    # Check: Is the jump declared as a "long jump"? If so, get new directions
                    # and reset the reward memory queue. Bees do not stop for flowers when long jumping.
                    if self.bee.is_long_jump:
                        # After finishing a long jump, get next directions.
                        # Move to 8
                        self.bee.is_long_jump = False
                        self.bee.reward_memory.clear()
                        self.new_action_and_move()

                        return

                    # Checkpoint: reaching here means the bee completed a short jump.

                    # Check if there is a flower below
                    plant_below_list = self.get_plant_below()
                    
                    #! Print list of plant below bee 5
                    if self.bee.model.steps in range(self.START, self.END) and self.bee.unique_id == 5:
                        print(f"Step {self.bee.model.steps} plant below list: {plant_below_list}")
                        
                        
                    # There is a plant below the bee
                    if plant_below_list:
                        plant = plant_below_list[0]
                        self.bee.latest_visited_flowers.append(plant) # Add plant below to the memory
                        
                        #! Find when bee 5 lands on flower 5
                        # if self.bee.model.steps in range(self.START, self.END) and self.bee.unique_id == 5:
                        #     print(f"landing")
                        
                        # Check if can land on the flower
                        # Can land on flower: there is no bee on flower and flower has no scent
                        if plant.bee_on is None and not plant.has_scent():
                            
                            # Attempt to land on the flower
                            landing = self.land_on_flower(self.bee, plant)
                            
                            # Determine whether to land based on probability
                            if landing:
                                # Set bee_on and on_plant
                                self.bee.on_plant = plant
                                plant.bee_on = self.bee
                                # self.bee.latest_visited_flowers.append(self.bee.on_plant)
                                
                                # TODO: Following parameters can be removed
                                plant.visit_count += 1
                                plant.visited = True  
                                plant.visited_this_step = True
                                plant.visit_timestamp.append(self.bee.model.steps)
                                                                
                                # Calculate the remaining amount of nectar to collect for the following fixed amount of timesteps
                                nectar_to_collect = min(self.bee.collection_rate, plant.nectar, self.bee.stomach_capacity - self.bee.nectar_in_stomach)
                                
                                # When there is nectar, regardless of amount  
                                if nectar_to_collect > 0: # TODO: update this to threshold, change foraging logic
                                    self.bee.forage_timer = self.bee.forage_timer_max
                                    self.nectar_to_collect = nectar_to_collect
                                    self.bee.reward_memory.append(1)
                                    

                                # There is no nectar
                                elif nectar_to_collect == 0: # TODO: update this to threshold, change foraging logic
                                    # Unsuccessful visit
                                    self.bee.reward_memory.append(0)

                                    # Clean up and leave
                                    plant.bee_on = None
                                    plant.leave_scent()
                                    self.bee.on_plant = None

                                    # Take new action
                                    self.new_action_and_move()

                                # Defensive programming
                                else:
                                    raise ValueError("Nectar to collect should not be negative")
                            
                            # Bee decided not to land on the plant based on probability
                            # Add the flower tot he latest_visited_flowers
                            # NOTE: Reward memory is not updated when not landing
                            elif not landing:
                                # self.bee.latest_visited_flowers.append(plant)
                                self.new_action_and_move()
                                
                            # Defensive programming
                            else:
                                raise ValueError("Landing logic error")
                            
                        # Did not land on the flower because there is a bee
                        elif plant.bee_on:
                            self.bee.reward_memory.append(1)
                            self.new_action_and_move()
                        
                        # Did not land on flower because there is scent detected
                        elif plant.has_scent():
                            self.bee.reward_memory.append(0)
                            self.new_action_and_move()
                        
                        # Defensive programming                            
                        else: 
                            assert False, "Defensive programming 0, different reason why bee cannot land on flower"

                    # There is no flower below bee
                    elif not plant_below_list: 
                        self.bee.reward_memory.append(0)
                        self.new_action_and_move()
                    else:
                        assert False, "Plant below detection problem"
                        
                # 5.2 No, not at destined location, and bee is scanning while moving during short jump
                else:
                    # If not a long jump, scan
                    if not self.bee.is_long_jump:
                        available_plants = self.get_available_plants(vision_range=self.bee.vision_range)
                        if available_plants:
                            random_plant = self.bee.random.choice(available_plants)
                            self.bee.travel_route = [random_plant.pos]
                            
                            dx = random_plant.pos[0] - self.bee.pos[0]
                            dy = random_plant.pos[1] - self.bee.pos[1]
                            theta_to_flower = math.degrees(math.atan2(dy, dx))
                            theta_to_flower = (theta_to_flower + 180) % 360 - 180
                            self.bee.theta = theta_to_flower
                    # else, no flowers nearby, continue with the current direction
                    return
                
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
    
        if self.bee.model.steps in range(0, 23) and self.bee.unique_id == 5:
            print(f"Destination Coord {destination_coord}")
        
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
        
        #! Got new action
        if self.bee.model.steps in range(self.START, self.END) and self.bee.unique_id == 5:
            print(f"{self.bee.model.steps} got new action after long jump")
            print(f"{self.bee.model.steps} new route {self.bee.travel_route}")
        
        #! New route
        if self.bee.model.steps in range(18, 23) and self.bee.unique_id == 5:
            print(f"Step {self.bee.model.steps} new route {self.bee.travel_route}")
        if self.bee.travel_route:
            next_coord = self.bee.travel_route.pop(0)
            next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
            self.bee.model.space.move_agent(self.bee, next_coord)

class MyBeeBehavior_new(BeeBehavior):
    """
    An extension of BiasedCompositeLevyBehavior. The differences:
    - [ ] Homing behavior: Bee flies back to hive either: 
            - [x] 1) the stomach is full
            - [ ] 2) unsuccessful visits reaches 15.
            After bee reaches the hive, wait for 5 steps before releasing it back
            into the environment as a "new" bee.
    - [x] Flower memory: Each bee will have a memory of the last five flowers they have visited.
    - [x] Flower behavior: Set the flower has 3 micro liters of nectar. Each visits by
                           bee it will take 1 micro liters of nectar. Each visit time
                           is set to 3 steps.
    - [x] Scent marking: after a bee has successfully foraged, it will leave a decaying
                         scent mark. Other bees will be able to detect the scent mark when
                         they are above it and does not land on it. After 5 steps, the scent
                         mark will disappear.
                         
    What I expect when running this behavior in an empty environment: the bee will perform five
    short jumps before making one long jump. After that, it will repeat the process.
    
    """
    def __init__(self, landing_prob, biased_distribution=None):
        super().__init__(landing_prob=landing_prob)
        

        if biased_distribution is None:
            self.biased_distribution = lambda n=1: np.random.uniform(-180, 180, size=n)
        else:
            self.biased_distribution = biased_distribution

    # TODO: when arrived at destination, it should not be visited, update memory immediately
    # TODO: check when is memory updated in the code below
    def step(self):
        # 0. Check if bee is returning to the hive
        if self.bee.homing:
            if not self.bee.travel_route:
                # Arrived at hive
                self.bee.homing = False
                self.bee.nectar_in_stomach = 0.0
                self.bee.reward_memory.clear()
                self.bee.latest_visited_flowers.clear()
                self.theta = None
                self.new_action_and_move()
                return
            else:
                # Continue travel toward the hive
                next_coord = self.bee.travel_route.pop(0)
                next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                self.bee.model.space.move_agent(self.bee, next_coord)
                return

        # 1. Spending time on foraging nectar?
        # 1.1 No
        if self.bee.forage_timer == 0 and self.bee.on_plant is None:
            # 2. Have a set destination location?
            # 2.1 No, meaning the bee arrived at its destination or the bee is leaving the hive
            if not self.bee.travel_route:
                self.new_action_and_move()
            # 2.2 Yes
            else:
                # 3. Move to next location
                next_coord = self.bee.travel_route.pop(0)
                next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                self.bee.model.space.move_agent(self.bee, next_coord)

                # 5. Is it at the destination location?
                # 5.1 Yes
                if not self.bee.travel_route:
                    # Check: Is the jump declared as a "long jump"? If so, get new directions
                    # and reset the reward memory queue. Bees do not stop for flowers when long jumping.
                    if self.bee.is_long_jump:
                        # After finishing a long jump, get next directions.
                        # Move to 8
                        self.bee.is_long_jump = False
                        self.bee.reward_memory.clear()
                        self.new_action_and_move()
                        return

                    # Checkpoint: reaching here means the bee completed a short jump.

                    # Check if there is a flower below
                    plant_below_list = self.get_plant_below()

                    # There is a plant below the bee
                    if plant_below_list:
                        plant = plant_below_list[0]
                        self.bee.latest_visited_flowers.append(plant) # Add plant below to the memory
                        
                        # Check if can land on the flower
                        # Can land on flower: there is no bee on flower and flower has no scent
                        if plant.bee_on is None and not plant.has_scent():
                            
                            # Attempt to land on the flower
                            landing = self.land_on_flower(self.bee, plant)
                            
                            # Determine whether to land based on probability
                            if landing:
                                # Set bee_on and on_plant
                                self.bee.on_plant = plant
                                plant.bee_on = self.bee
                                # self.bee.latest_visited_flowers.append(self.bee.on_plant)
                                
                                # TODO: Following parameters can be removed
                                plant.visit_count += 1
                                plant.visited = True  
                                plant.visited_this_step = True
                                plant.visit_timestamp.append(self.bee.model.steps)
                                                                
                                # Calculate the remaining amount of nectar to collect for the following fixed amount of timesteps
                                nectar_to_collect = min(self.bee.collection_rate, plant.nectar, self.bee.stomach_capacity - self.bee.nectar_in_stomach)
                                
                                # When there is nectar, regardless of amount  
                                if nectar_to_collect > 0: # TODO: update this to threshold, change foraging logic
                                    self.bee.forage_timer = self.bee.forage_timer_max
                                    self.nectar_to_collect = nectar_to_collect
                                    self.bee.reward_memory.append(1)
                                    

                                # There is no nectar
                                elif nectar_to_collect == 0: # TODO: update this to threshold, change foraging logic
                                    # Unsuccessful visit
                                    self.bee.reward_memory.append(0)

                                    # Clean up and leave
                                    plant.bee_on = None
                                    plant.leave_scent()
                                    self.bee.on_plant = None

                                    # Take new action
                                    self.new_action_and_move()

                                # Defensive programming
                                else:
                                    raise ValueError("Nectar to collect should not be negative")
                            
                            # Bee decided not to land on the plant based on probability
                            # Add the flower tot he latest_visited_flowers
                            # NOTE: Reward memory is not updated when not landing
                            elif not landing:
                                # self.bee.latest_visited_flowers.append(plant)
                                self.new_action_and_move()
                                
                            # Defensive programming
                            else:
                                raise ValueError("Landing logic error")
                            
                        # Did not land on the flower because there is a bee
                        elif plant.bee_on:
                            self.bee.reward_memory.append(1)
                            self.new_action_and_move()
                        
                        # Did not land on flower because there is scent detected
                        elif plant.has_scent():
                            self.bee.reward_memory.append(0)
                            self.new_action_and_move()
                        
                        # Defensive programming                            
                        else: 
                            assert False, "Defensive programming 0, different reason why bee cannot land on flower"

                    # There is no flower below bee
                    elif not plant_below_list: 
                        self.bee.reward_memory.append(0)
                        self.new_action_and_move()
                    else:
                        assert False, "Plant below detection problem"
                        
                # 5.2 No, not at destined location, and bee is scanning while moving during short jump
                else:
                    # If not a long jump, scan
                    if not self.bee.is_long_jump:
                        available_plants = self.get_available_plants(vision_range=self.bee.vision_range)
                        if available_plants:
                            random_plant = self.bee.random.choice(available_plants)
                            self.bee.travel_route = [random_plant.pos]
                            
                            dx = random_plant.pos[0] - self.bee.pos[0]
                            dy = random_plant.pos[1] - self.bee.pos[1]
                            theta_to_flower = math.degrees(math.atan2(dy, dx))
                            theta_to_flower = (theta_to_flower + 180) % 360 - 180
                            self.bee.theta = theta_to_flower
                    # else, no flowers nearby, continue with the current direction
                    return
                
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

class MyBeeBehavior_v1(BeeBehavior):
    """
    An extension of BiasedCompositeLevyBehavior. The differences:
    - [ ] Homing behavior: Bee flies back to hive either: 
            - [x] 1) the stomach is full
            - [ ] 2) unsuccessful visits reaches 15.
            After bee reaches the hive, wait for 5 steps before releasing it back
            into the environment as a "new" bee.
    - [x] Flower memory: Each bee will have a memory of the last five flowers they have visited.
    - [x] Flower behavior: Set the flower has 3 micro liters of nectar. Each visits by
                           bee it will take 1 micro liters of nectar. Each visit time
                           is set to 3 steps.
    - [x] Scent marking: after a bee has successfully foraged, it will leave a decaying
                         scent mark. Other bees will be able to detect the scent mark when
                         they are above it and does not land on it. After 5 steps, the scent
                         mark will disappear.
                         
    What I expect when running this behavior in an empty environment: the bee will perform five
    short jumps before making one long jump. After that, it will repeat the process.
    
    """
    def __init__(self, landing_prob, biased_distribution=None):
        super().__init__(landing_prob=landing_prob)
        

        if biased_distribution is None:
            self.biased_distribution = lambda n=1: np.random.uniform(-180, 180, size=n)
        else:
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
                self.theta = None
                self.new_action_and_move()
                return
            else:
                # Continue travel toward the hive
                next_coord = self.bee.travel_route.pop(0)
                next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                self.bee.model.space.move_agent(self.bee, next_coord)
                return

        # 1. Spending time on foraging nectar?
        # 1.1 No
        if self.bee.forage_timer == 0 and self.bee.on_plant is None:
            # 2. Have a set destination cell?
            # 2.1 No, meaning the bee arrived at its destination or the bee is leaving the hive
            if not self.bee.travel_route:
                self.new_action_and_move()
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
                    # and reset the reward memory queue. Bees do not stop for flowers when long jumping.
                    if self.bee.is_long_jump:
                        # After finishing a long jump, get next directions.
                        # Move to 8
 
                        self.bee.is_long_jump = False
                        self.bee.reward_memory.clear()
                        self.new_action_and_move()
                        return

                    # Checkpoint: reaching here means the bee completed a short jump.

                    # Checking below if there is a flower below it can land on.
                    available_plants = self.get_available_plants(vision_range=0)

                    # Reset target regardless of whether flower is found — destination reached
                    self.bee.have_target = False
                    self.bee.target_flower = None

                    # Land on plant if there is an available plant and there are no scent marks on the flower
                    if available_plants:
                        plant = self.bee.random.choice(available_plants)
                        if plant.bee_on is None and plant.reserved_by is None and not plant.has_scent():
                            # Reserve first
                            plant.reserved_by = self.bee
                            # Try to land
                            landing = self.land_on_flower(self.bee, plant=plant)
                            if landing:
                                # Only now set bee_on and on_plant!
                                self.bee.on_plant = plant
                                plant.bee_on = self.bee
                                plant.visit_count += 1
                                plant.visit_timestamp.append(self.bee.model.steps)
                                plant.visited = True
                                plant.visited_this_step = True
                                self.bee.latest_visited_flowers.append(self.bee.on_plant)

                                nectar_to_collect = min(self.bee.collection_rate, plant.nectar, self.bee.stomach_capacity - self.bee.nectar_in_stomach)
                                if nectar_to_collect > 0:
                                    self.bee.forage_timer = self.bee.forage_timer_max
                                    self.bee.target_plant = plant
                                    self.bee.nectar_to_collect = nectar_to_collect
                                    self.bee.reward_memory.append(1)
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

                                elif nectar_to_collect == 0: #TODO: make this <= to some value, so that it indicates the replenish rate is low
                                    # Unsuccessful visit, update reward memory queue
                                    self.bee.reward_memory.append(0)
                                    # Clean up and leave
                                    plant.bee_on = None
                                    plant.reserved_by = None
                                    plant.leave_scent()
                                    self.bee.on_plant = None
                                    self.new_action_and_move()
                                else:
                                    raise ValueError("Nectar to collect should not be negative.")
                            else:
                                # Failed landing due to random chance
                                self.bee.latest_visited_flowers.append(plant)
                                if plant.bee_on is not None:
                                    self.bee.reward_memory.append(1)
                                elif plant.has_scent():
                                    self.bee.reward_memory.append(0)
                                else:
                                    self.bee.reward_memory.append(0)
                                plant.reserved_by = None
                                self.new_action_and_move()
                        else:
                            # Plant is not available (bee_on or scent)
                            self.bee.latest_visited_flowers.append(plant)
                            if plant.bee_on is not None:
                                self.bee.reward_memory.append(1)
                            elif plant.has_scent():
                                self.bee.reward_memory.append(0)
                            else:
                                self.bee.reward_memory.append(0)
                            plant.reserved_by = None
                            self.new_action_and_move()
                    else:
                        # No available plant
                        self.new_action_and_move()

                # 5.2 No, but bee is scanning while moving during short jump
                else:
                    # Check if there is a flower nearby
                    available_plants = self.get_available_plants(vision_range=self.bee.vision_range)
                    if available_plants and not self.bee.have_target:
                        random_plant = self.bee.random.choice(available_plants)
                        self.bee.have_target = True
                        self.bee.target_flower = random_plant
                        self.bee.travel_route = [random_plant.pos]

                        dx = random_plant.pos[0] - self.bee.pos[0]
                        dy = random_plant.pos[1] - self.bee.pos[1]
                        theta_to_flower = math.degrees(math.atan2(dy, dx))
                        theta_to_flower = (theta_to_flower + 180) % 360 - 180
                        self.bee.theta = theta_to_flower
                    # else, no flowers nearby, continue with the current direction
                    return

        # 1.2 yes, currently still foraging
        elif self.bee.on_plant is not None:
            self.bee.forage_timer -= 1
            if self.bee.forage_timer == 0:  # Max forage timer is 3, meaning three time steps
                # Update the plant nectar amount and bee stomach content
                if self.bee.target_plant is not None:
                    self.bee.target_plant.nectar -= self.bee.nectar_to_collect
                    self.bee.add_nectar(self.bee.nectar_to_collect)

                # Clear temp collection parameters
                self.bee.nectar_to_collect = 0.0
                self.bee.target_plant = None

                # Wrap up interaction with flower
                self.bee.on_plant.bee_on = None
                if hasattr(self.bee.on_plant, 'reserved_by'):
                    self.bee.on_plant.reserved_by = None
                self.bee.on_plant.leave_scent()
                self.bee.on_plant = None

                self.get_new_action()

                if self.bee.travel_route:
                    next_coord = self.bee.travel_route.pop(0)
                    next_coord = self.clamp_position(next_coord, self.bee.model.space.width, self.bee.model.space.height)
                    self.bee.model.space.move_agent(self.bee, next_coord)

    def get_new_action(self):
        # 8. Are any nearby flowers not in the visited list?
        available_plants = self.get_available_plants(vision_range=self.bee.vision_range)

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
    """
    Things to check and verify
    - [ ] Bee navigation logic (levy flight)
        - [x] Collect and plot the jump distances, ensure it is a negative exponential distribution  
        - [x] Collect and plot the turn angles, ensure it matches to the defined distribution 
        - [ ] What is a long jump, what is a short jump? Determine this in a different file.
    - [x] Bee scanning logic
        - A bee will fly and scan for flowers during regular jumps
        - A bee will not scan during a "long jump". A jump is defined as a long jump based on the reward_memory queue.
    - [x] Bee landing logic
        - Bees will land on a flower based on a probability, however I have set the probability to 100%
        - A bee will not land when there is a scent mark on the flower or a bee is on the flower
    - [x] Bee foraging logic
        - Bee will forage nectar from a flower for a fixed amount of time steps, then leaves a scent on the flower when they are done
        - If the flower does not have nectar, they will leave a scent and move on
    - [x] Update on latest_visited_flowers:
        - Bees will update this memory when they land on a flower, regardless whether the flower has nectar or not.
    - [x] Bee nectar collection logic
        - Bees will collect nectar from the flower based, the amount is fixed at 3 micro liters, and the amount of time it takes to
          forage is also fixed
    - [x] Scent marking logic
        - Checked via visualisation, where scent mark timer is assigned when bees leave a flower after visiting, regardless if the flower has nectar or not.
        - Time decrements correctly
    - [x] Landing logic
        - Bees does not land on flowers with scent marks
        - Bees does not land on flowers that have a bee on top of it, which is checked by parameter bee_on that registers the bee that is on the flower
    - [x] Homing behavior
        - [x] Bees only returns to hive when stomach is full
    - [x] Check if the bee Levy flight is working correctly



    - [ ] Removed reward_memory from beebehavior class to bee itself and reward_size to bee class


    TODO for other behaviors. 
    - [ ] After confirming my_behavior is working as intended, refactor all other behaviors to use the same logic as MyBeeBehavior
    """
    
################################################################################
####Above: my_behavior##########################################################
################################################################################
"""
# Sudo code for reimplementaion
def step(self):
    # 0. If bee is in homing mode (returning to hive):
    if bee.homing:
        if bee.travel_route is empty:
            # Arrived at hive
            bee.homing = False
            bee.nectar_in_stomach = 0
            bee.reward_memory.clear()
            bee.latest_visited_flowers.clear()
            bee.theta = None
            self.new_action_and_move()
            return
        else:
            # Move towards hive along travel route
            move_bee_along_route()
            return

    # 1. If bee is NOT currently foraging on a plant (forage_timer == 0 and not on_plant):
    if bee.forage_timer == 0 and bee.on_plant is None:
        if bee.travel_route is empty:
            # Destination reached, or no target set — plan new action and start moving
            self.new_action_and_move()
        else:
            # Move one step along current route
            move_bee_along_route()

            if bee.travel_route is now empty:  # Arrived at destination
                if bee.is_long_jump:
                    # Finished a long jump: reset jump state and reward memory
                    bee.is_long_jump = False
                    bee.reward_memory.clear()
                    self.new_action_and_move()
                    return

                # Otherwise, check for available flowers directly beneath bee
                available_plants = self.get_available_plants(vision_range=0)
                bee.have_target = False
                bee.target_flower = None

                if available_plants:
                    plant = random.choice(available_plants)
                    # Only land if not already occupied and no scent
                    if plant.bee_on is None and not plant.has_scent():
                        # Attempt landing with probability (landing_prob)
                        if self.land_on_flower(bee, plant):
                            bee.on_plant = plant
                            plant.bee_on = bee
                            update_visit_metadata(plant, bee)
                            bee.latest_visited_flowers.append(plant)
                            nectar_to_collect = min(bee.collection_rate, plant.nectar, bee.stomach_capacity - bee.nectar_in_stomach)
                            if nectar_to_collect > 0:
                                bee.forage_timer = bee.forage_timer_max
                                bee.target_plant = plant
                                bee.nectar_to_collect = nectar_to_collect
                                bee.reward_memory.append(1)
                                # If bee's stomach full, switch to homing mode and set path to hive
                                if bee.is_full():
                                    bee.homing = True
                                    bee.travel_route = path_to_hive(bee)
                                    bee.theta = direction_to_hive(bee)
                            else:
                                # No nectar to collect — unsuccessful, update memory and leave
                                bee.reward_memory.append(0)
                                plant.bee_on = None
                                plant.leave_scent()
                                bee.on_plant = None
                                self.new_action_and_move()
                        else:
                            # Failed landing by chance
                            bee.latest_visited_flowers.append(plant)
                            update_reward_memory_failed(plant, bee)
                            self.new_action_and_move()
                    else:
                        # Flower not available (occupied or scented)
                        bee.latest_visited_flowers.append(plant)
                        update_reward_memory_failed(plant, bee)
                        self.new_action_and_move()
                else:
                    # No flower available at current spot
                    self.new_action_and_move()
            else:
                # Bee is mid-jump, check for target flowers within vision range
                available_plants = self.get_available_plants(vision_range=bee.vision_range)
                if available_plants and not bee.have_target:
                    random_plant = random.choice(available_plants)
                    bee.have_target = True
                    bee.target_flower = random_plant
                    bee.travel_route = [random_plant.pos]
                    bee.theta = direction_to_flower(bee, random_plant)
                return  # Otherwise, keep moving on current route

    # 2. If bee IS currently foraging (on_plant is not None)
    elif bee.on_plant is not None:
        bee.forage_timer -= 1
        if bee.forage_timer == 0:
            # Finished foraging: update plant nectar, bee's stomach
            if bee.target_plant:
                bee.target_plant.nectar -= bee.nectar_to_collect
                bee.add_nectar(bee.nectar_to_collect)
            bee.nectar_to_collect = 0
            bee.target_plant = None
            # Clean up plant state
            bee.on_plant.bee_on = None
            bee.on_plant.leave_scent()
            bee.on_plant = None
            self.get_new_action()
            if bee.travel_route:
                move_bee_along_route()

# (Helper functions referenced above would include: move_bee_along_route, update_visit_metadata, 
# path_to_hive, direction_to_hive, direction_to_flower, update_reward_memory_failed, etc.)
"""