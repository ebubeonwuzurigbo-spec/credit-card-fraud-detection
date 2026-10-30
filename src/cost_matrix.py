import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from data_loader import load_transaction_data


COST_FP=15
COST_FN=300

def optimize_threshold():
    df = load_transaction_data('data/creditcard.csv')
    model=joblib.load("models/fraud_model.pkl")
    X=df.drop(columns=["Class"])
    #Rename columns to match the model's expected input features
    X=X.rename(columns={"Time": "scaled_time", "Amount": "scaled_amount"})
    #Force the order of columns to match the model's expected input features
    X=X[model.get_booster().feature_names]
    y=df["Class"]
    X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    probabilities=model.predict_proba(X_test)[:,1]
    best_threshold=0.5
    lower_cost=float('inf') # Initialize with infinity to ensure any calculated cost will be lower
    thresholds=np.arange(0.01,1.00,0.01)
    for tau in thresholds:
        #Convert probabilities to binary predictions based on the current threshold, 1 for fraud or 0 for safe
        predictions=(probabilities>=tau).astype(int)
        #Calculate the number of false positives and false negatives
        fp=np.sum((predictions==1) & (y_test==0))
        fn=np.sum((predictions==0) & (y_test==1))
        #Calculate the total cost based on the defined costs for false positives and false negatives
        total_cost=(fp*COST_FP)+(fn*COST_FN)

        if total_cost<lower_cost:
            lower_cost=total_cost
            best_threshold=tau

    print(f"Optimal threshold: {best_threshold:.2f}")
    print(f"Lowest Financial Loss: £{lower_cost:,.2f}")

if __name__=="__main__":
    optimize_threshold()