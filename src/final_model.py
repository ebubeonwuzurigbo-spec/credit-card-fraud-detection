import joblib
import os
from sklearn.linear_model import LogisticRegression
from data_loader import load_transaction_data
from preprocessor import preprocess_data
from sampling import create_balanced_sample
from model_trainer import prepare_data_for_training

def save_best_model(X_train, y_train):
    """
    Trains the winning model and saves it to disk.
    """
    print("Training final production model...")
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Create models directory if it doesn't exist
    if not os.path.exists('models'):
        os.makedirs('models')
        
    # Save the model
    model_path = 'models/fraud_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved successfully to {model_path}")

if __name__ == "__main__":
    # The full pipeline
    df = load_transaction_data("data/creditcard.csv")
    df = preprocess_data(df)
    df = create_balanced_sample(df)
    X_train, X_test, y_train, y_test = prepare_data_for_training(df)
    
    save_best_model(X_train, y_train)