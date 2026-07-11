import pandas as pd
import os
import logging

# Configure logging to track execution
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def load_transaction_data(file_path: str) -> pd.DataFrame:
    """
    Loads the credit card dataset from a CSV file.
    
    Args:
        file_path (str): Path to the csv file.
        
    Returns:
        pd.DataFrame: The loaded dataframe.
        
    Raises:
        FileNotFoundError: If the data is missing.
    """
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}")
        
        df = pd.read_csv(file_path)
        logging.info(f"Successfully loaded data with {df.shape[0]} rows and {df.shape[1]} columns.")
        return df
    
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

if __name__ == "__main__":
    # Point this to your data folder
    DATA_PATH = "data/creditcard.csv"
    raw_data = load_transaction_data(DATA_PATH)
    
    # Socratic Question: Why do we look at the first 5 rows immediately?
    print(raw_data.head())