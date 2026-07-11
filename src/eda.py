import pandas as pd
from data_loader import load_transaction_data

def analyze_class_distribution(df: pd.DataFrame):
    """
    Analyzes the balance between Fraud (1) and Normal (0) transactions.
    """
    # Calculate counts
    counts = df['Class'].value_counts()
    
    # Calculate percentages
    # TASK: Fill in the logic for 'percent_fraud' 
    # (Hint: it is (number of fraud / total rows) * 100)
    percent_fraud = (counts[1] / len(df)) * 100
    
    print("-" * 30)
    print(f"Normal Transactions (Class 0): {counts[0]}")
    print(f"Fraudulent Transactions (Class 1): {counts[1]}")
    print(f"Percentage of Fraud: {percent_fraud:.4f}%")
    print("-" * 30)

if __name__ == "__main__":
    DATA_PATH = "data/creditcard.csv"
    df = load_transaction_data(DATA_PATH)
    analyze_class_distribution(df)