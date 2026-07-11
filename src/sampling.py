import pandas as pd
from data_loader import load_transaction_data
from preprocessor import preprocess_data

def create_balanced_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates a 50/50 balanced dataset using Random Under-Sampling.
    """
    # 1. Shuffle the data just in case it's ordered
    df = df.sample(frac=1, random_state=42)
    
    # 2. Split Fraud and Non-Fraud
    fraud_df = df.loc[df['Class'] == 1]
    non_fraud_df = df.loc[df['Class'] == 0]
    
    # 3. Take a random sample of non-fraud cases equal to the number of fraud cases
    # We have 492 frauds, so we take 492 non-frauds.
    non_fraud_sample = non_fraud_df.sample(n=len(fraud_df), random_state=42)
    
    # TASK: Join the two dataframes together 
    # Hint: Use pd.concat([df1, df2])
    balanced_df = pd.concat([fraud_df, non_fraud_sample])
    
    # Shuffle again so the model doesn't see all frauds first
    balanced_df = balanced_df.sample(frac=1, random_state=42)
    
    return balanced_df

if __name__ == "__main__":
    DATA_PATH = "data/creditcard.csv"
    raw_df = load_transaction_data(DATA_PATH)
    processed_df = preprocess_data(raw_df)
    
    balanced_df = create_balanced_sample(processed_df)
    
    print("Balanced Dataset Distribution:")
    print(balanced_df['Class'].value_counts())
    print(f"\nTotal rows in balanced dataset: {len(balanced_df)}")