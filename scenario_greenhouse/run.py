import model
import pickle

if __name__ == "__main__":
    model = model.GreenhouseModel(seed=0)  # Initialize the model with a seed
    for i in range(10):  # Run the model for 3 steps
        model.step()
    
    
    df = model.datacollector.get_agent_vars_dataframe()

    with open("simulation_data.pkl", "wb") as f:
        pickle.dump(df, f)
    print("Model finished running.")
    # The model will run for 3 steps, and the output will be printed to the console.