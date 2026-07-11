from sklearn.preprocessing import StandardScaler
import pandas as pd
from data_loader import load_transaction_data

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Scales the 'Amount' and 'Time' columns.
    """
    scaler = StandardScaler()
    
    # We use double brackets because scaler expects a 2D array/DataFrame
    df['scaled_amount'] = scaler.fit_transform(df[['Amount']])
    df['scaled_time'] = scaler.fit_transform(df[['Time']])
    
    # Drop the original unscaled columns to keep the data clean
    df = df.drop(['Time', 'Amount'], axis=1)
    
    return df

if __name__ == "__main__":
    DATA_PATH = "data/creditcard.csv"
    raw_df = load_transaction_data(DATA_PATH)
    processed_df = preprocess_data(raw_df)
    
    print("Columns after preprocessing:")
    print(processed_df.columns)
    print("\nFirst 5 rows of scaled columns:")
    print(processed_df[['scaled_amount', 'scaled_time']].head())