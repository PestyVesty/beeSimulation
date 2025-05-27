import model
import pickle
import tqdm as tqdm

if __name__ == "__main__":
    model = model.OpenCoverModel(seed=0)  # Initialize the model with a seed
    for i in tqdm.tqdm(range(7 * 60 * 60)):  # Run the model for 500 steps
        model.step()
        
    df = model.datacollector.get_agent_vars_dataframe()
    
    with open("simulation_data.pkl", "wb") as f:
        pickle.dump(df, f)
    print("Model finished running.")