import pandas as pd
from data_dictionary import generate_data_dictionary, print_data_dictionary
import os

def main():
    """
    Generate data dictionaries for the specified datasets and save them to JSON files.
    """
    # Define the datasets to analyze
    datasets = [
        "cleaned_tm_players_dataset_v3_with_features",
        "cleaned_fifa_dataset_v3_with_features"
    ]
    
    # Create output directory if it doesn't exist
    output_dir = "data_dictionaries"
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset_name in datasets:
        print(f"\n{'='*50}")
        print(f"Analyzing dataset: {dataset_name}")
        print(f"{'='*50}")
        
        try:
            # Read the dataset
            df = pd.read_csv(f"datasets/{dataset_name}.csv")
            
            # Generate data dictionary
            output_file = os.path.join(output_dir, f"{dataset_name}_dictionary.json")
            data_dict = generate_data_dictionary(
                df,
                output_file=output_file,
                include_sample_values=True,
                sample_size=5
            )
            
            # Print the data dictionary
            print_data_dictionary(data_dict)
            
            print(f"\nData dictionary saved to: {output_file}")
            
        except FileNotFoundError:
            print(f"Error: Could not find dataset file: {dataset_name}.csv")
        except Exception as e:
            print(f"Error processing {dataset_name}: {str(e)}")

if __name__ == "__main__":
    main() 
