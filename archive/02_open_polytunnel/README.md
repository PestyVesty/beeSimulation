# Checkpoints and notes.
- agents.py: Only the behavior MyBeeBehavior() is tested and works as expected. 
  All other bee behaviors needs to be edited based on MyBeeBehavior and tested.

- visualization: Everything before section "3. Blank" does not need to be edited.

- data_visualisation: Currently visualizes the average nectar amount of each column
  and averaged bee visitation rate of each column at 10min intervals.
  Current problem: saves the same simulation scenario where the replenish rate is 0.


Creating new file: Refactoring code based on MyBeeBehavior. Refactoring code to expland on
bee behavior and incorporating other mechanism, such as sugar concentration and temperature.

Start by:
    - Reduce the code size for agents and model.py
    - Creating a polytunnel class
    - Check and verify how bees behave with walls