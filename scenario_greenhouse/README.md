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
    - type: 'male' or 'female'
    - coord: x, y position
    - pollinated: flag status for female flowers
    - has_pollen: flag status for males

## Bee Object
Attributes for bees:
    - coord: x, y position
    - theta: direction bee facing
    - has_pollen: boolean
    - pollen_timer: countdown for pollen viability
    - destination: (x, y) target or None


# TODO:
    - [x] Setup dependencies
    - [x] Create structure
    - [x] Environment
      - [x] Initiate environment
      - [x] Place the plants
      - [x] Place the bees
    - [ ] Plant object
      - [x] define attributes
      - [ ] Define plant behavior
    - [ ] Bee object
      - [x] Define attributes
      - [ ] Define bee behavior
    - [ ] Execution of the model
      - [x] Create run.py
      - [x] Create an environment and run 100 steps with no errors (no behavior for either agents)
    - [ ] Data collection
      - [x] Collect location of the agents
      - [ ] Visualize change the color or shape of the agent based on the state
    - [ ] Visualisation
      - [ ] Implement visualisation using matplotlib
      - [ ] Add in additional color coding for visualisation
      - [ ] Basic visualisation
    
# Optional Todos:
    - [ ] If creating thousands of agents, consider using Mesa-frames (https://github.com/projectmesa/mesa-frames)