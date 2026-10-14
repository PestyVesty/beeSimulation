# Overview
Simulation replicates specifically the plant layout "MFMFMFMF" only. The detailed
environment settings can be found in the research paper above. There are two major
differences:
    1. The implemented environment is continuous not grid
    2. Origianl grid environment has cells 0.35m x 0.35m. This seems to be a typo
       or error (from last time talking with the first author, should of been 
       0.7m x 0.7m). If overlapping the grid and continuous environment, plants
       from the continuous environment will land in the middle of the cells from
       the grid environment (where the cell represents a plant).

In the paper, Levy flight was simulated using Cauchy Distribution. I did not find
the exact parameters used for the Cauchy Distribution, so I used the standard
values. Otherwise, all plant and bee behavior should replicate the behaviors defined
in the research paper.

# Documentation
Reimplementation of the environment of research paper "Simulation-governed design
and tuning of greenhouses for successful bee pollination". Main differences:
    - Continuous environment: allows for cleaner integration of geometric contstraints like baffles
    - Cell size: 0.35m x 0.35m -> 0.7m x 0.7m
    - Bee Movement: Moore neighborhood + cell-based Levy -> Real-valued positions + continuous Levy


## Enviornment
    - Dimensions: 72 x 18 cells -> 72x0.7m x 18x0.7m
    - Four beds placed 2x0.7m away from each other and away from each other
    - Beds are 6x0.7m away from the width of greenhouse
    - Beds closest to the length of greenhouse are 2x0.7m away
    - Each bed consists of two rows, where each row has 60 plants
  
## Plant Object Attributes
Attributes of flowers:
  - gender: gender of the flower
  - has_pollen: for male flowers, boolean to indicate whether pollen are available
  - pollinated: for female flower, boolean to indicate whether flower is pollinated
  - pos: position of the flower in the simulation enviornment
  - bee_on: reference to the bee object that is on top of the flower
  - visited: boolean to indicate whether the plower has been visited or not. This
             is to simulate repellent scent marking. 

## Bee Object
Key attributes for bees:
  - behavior: Corresponds to the class BeeBehavior. Designed this way to allow
              the option to include different bee behavior.
  - walls:    LineString representation of the walls in the simulation.
  - pos:      Position of the bee in the environment.
  - theta:    Direction of the bee is facing. This was not used in the simulation
              and does not influence bee behavior.
  - vision_range:     Visual range of the bee.
  - forage_timer:     Foraging time left for the bee when it is on a flower.
  - forage_timer_max: Default 3, value from the research paper above.
  - has_pollen  :     Boolean to determine whether bee has pollen or not.
  - pollen_timer:     Timer for pollen sticking to bee. When reaches 0 no pollen
                      is left on the bee.
  - pollen_timer_max: How long a pollen sticks to a bee.

  - self.travel_route: A list, containing coordinates for bees to travel along.
  - self.on_plant:     The plant object the bee landed on.

    
