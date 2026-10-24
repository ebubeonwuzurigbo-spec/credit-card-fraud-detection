import joblib
import pandas as pd
import shap
import os
import matplotlib
# Tell matplotlib to use a non-interactive backend (so we can save plots without needing a display)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from data_loader import load_transaction_data

def explain_fraud_prediction():

    # 1. Load the trained XGBoost model and dataset
    print("Loading model and sample data...")
    model = joblib.load("models/fraud_model.pkl")
    df = load_transaction_data('data/creditcard.csv')

    # Isolate features (drop target)
    X = df.drop(columns=["Class"])
    
    # Grab the very first known fraud transaction to inspect
    fraud_sample = X[df["Class"] == 1].iloc[[0]]

    # 2. Initialize TreeSHAP on our XGBoost model
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(fraud_sample)

    # 3. Extract the top 3 features driving this specific fraud decision
    feature_names = fraud_sample.columns
    contributions = pd.Series(shap_values.values[0], index=feature_names)
    
    # Sort by strongest risk drivers (positive SHAP values push towards fraud)
    top_drivers = contributions.sort_values(ascending=False).head(3)

    print("\n==============================")
    print("SHAP FRAUD DECISION EXPLANATION")
    print("==============================")
    print("Top 3 Factors Driving This Fraud Alert:\n")
    for rank, (feature, impact) in enumerate(top_drivers.items(), 1):
        print(f" {rank}. {feature}: +{impact:.4f} risk contribution")
    print("==============================\n")

def generate_global_summary_plot():
    """
    Generates a global SHAP summary plot for the entire dataset.
    """
    print("Generating global SHAP summary plot...")
    
    # Load model and data
    model = joblib.load("models/fraud_model.pkl")
    df = load_transaction_data('data/creditcard.csv')
    
    # Isolate features (drop target)
    X = df.drop(columns=["Class"])
    X_sample= X.sample(n=200, random_state=42)  # Sample for faster plotting
    
    # Initialize TreeSHAP
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    
    # Create summary plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    
    # Save the plot
    if not os.path.exists('plots'):
        os.makedirs('plots')
        
    plot_path = 'plots/global_shap_summary.png'
    plt.savefig(plot_path, bbox_inches='tight')
    plt.close()
    
    print(f"Global SHAP summary plot saved to {plot_path}")


if __name__ == "__main__":
    explain_fraud_prediction()
    generate_global_summary_plot()