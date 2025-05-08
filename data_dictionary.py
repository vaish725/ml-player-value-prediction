# This code is inspired by:
# 1. pandas-profiling library: https://github.com/pandas-profiling/pandas-profiling
# 2. scikit-learn's data dictionary utilities
# 3. Streamlit's data exploration examples: https://docs.streamlit.io/
# Modified and enhanced by Vaishnavi Kamdi for the Football Player Analysis project

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Union
import json

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def generate_data_dictionary(df: pd.DataFrame, 
                           output_file: str = None,
                           include_sample_values: bool = True,
                           sample_size: int = 5) -> Dict[str, Any]:
    """
    Generate a comprehensive data dictionary for a pandas DataFrame.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input DataFrame to analyze
    output_file : str, optional
        If provided, saves the data dictionary to this JSON file
    include_sample_values : bool, default True
        Whether to include sample values for each column
    sample_size : int, default 5
        Number of sample values to include if include_sample_values is True
    
    Returns:
    --------
    Dict[str, Any]
        A dictionary containing the data dictionary information
    """
    # Initialize the data dictionary
    data_dict = {
        "dataset_info": {
            "number_of_rows": len(df),
            "number_of_columns": len(df.columns),
            "memory_usage": f"{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        },
        "columns": {}
    }
    
    # Analyze each column
    for column in df.columns:
        # Get basic column information
        column_info = {
            "data_type": str(df[column].dtype),
            "missing_values": {
                "count": int(df[column].isnull().sum()),
                "percentage": f"{(df[column].isnull().sum() / len(df) * 100):.2f}%"
            },
            "unique_values": int(df[column].nunique())
        }
        
        # Add descriptive statistics based on data type
        if pd.api.types.is_numeric_dtype(df[column]):
            column_info["statistics"] = {
                "mean": float(df[column].mean()) if not df[column].isnull().all() else None,
                "median": float(df[column].median()) if not df[column].isnull().all() else None,
                "std": float(df[column].std()) if not df[column].isnull().all() else None,
                "min": float(df[column].min()) if not df[column].isnull().all() else None,
                "max": float(df[column].max()) if not df[column].isnull().all() else None
            }
        elif pd.api.types.is_categorical_dtype(df[column]) or pd.api.types.is_object_dtype(df[column]):
            # Get value counts for categorical/object columns
            value_counts = {str(k): int(v) for k, v in df[column].value_counts().head(10).to_dict().items()}
            column_info["top_values"] = value_counts
        
        # Add sample values if requested
        if include_sample_values:
            # Get non-null sample values
            sample_values = df[column].dropna().sample(min(sample_size, len(df))).tolist()
            # Convert numpy types to Python native types
            sample_values = [float(x) if isinstance(x, np.floating) else int(x) if isinstance(x, np.integer) else x for x in sample_values]
            column_info["sample_values"] = sample_values
        
        # Add column to data dictionary
        data_dict["columns"][column] = column_info
    
    # Save to file if output_file is provided
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(data_dict, f, indent=4, cls=NumpyEncoder)
    
    return data_dict

def print_data_dictionary(data_dict: Dict[str, Any]) -> None:
    """
    Print the data dictionary in a readable format.
    
    Parameters:
    -----------
    data_dict : Dict[str, Any]
        The data dictionary to print
    """
    print("\n=== Dataset Information ===")
    print(f"Number of rows: {data_dict['dataset_info']['number_of_rows']}")
    print(f"Number of columns: {data_dict['dataset_info']['number_of_columns']}")
    print(f"Memory usage: {data_dict['dataset_info']['memory_usage']}")
    print("\n=== Column Details ===")
    
    for column, info in data_dict['columns'].items():
        print(f"\nColumn: {column}")
        print(f"Data type: {info['data_type']}")
        print(f"Missing values: {info['missing_values']['count']} ({info['missing_values']['percentage']})")
        print(f"Unique values: {info['unique_values']}")
        
        if 'statistics' in info:
            print("\nStatistics:")
            for stat, value in info['statistics'].items():
                if value is not None:
                    print(f"  {stat}: {value:.2f}")
        
        if 'top_values' in info:
            print("\nTop values:")
            for value, count in info['top_values'].items():
                print(f"  {value}: {count}")
        
        if 'sample_values' in info:
            print("\nSample values:")
            print(f"  {info['sample_values']}")

# Example usage:
if __name__ == "__main__":
    # Create a sample DataFrame
    sample_data = {
        'age': [25, 30, 35, None, 28],
        'salary': [50000, 60000, 75000, 45000, 55000],
        'department': ['IT', 'HR', 'IT', 'Finance', 'HR']
    }
    df = pd.DataFrame(sample_data)
    
    # Generate and print data dictionary
    data_dict = generate_data_dictionary(df)
    print_data_dictionary(data_dict)
    
    # Save to file
    generate_data_dictionary(df, output_file="data_dictionary.json") 
