

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from data_loader import load_transaction_data
from preprocessor import preprocess_data
from sampling import create_balanced_sample
from sklearn.metrics import confusion_matrix, classification_report

def prepare_data_for_training(df: pd.DataFrame):
    """
    Separates features and targets, then splits into train and test sets.
    """
    # X is the features. We drop the 'Class' column.
    X = df.drop('Class', axis=1)
    
    # y is the target. We ONLY want the 'Class' column.
    y = df['Class']
    
    # TASK 1: Use train_test_split here. 
    # - Pass X and y.
    # - Set test_size to 0.2 (20% for testing).
    # - Set random_state to 42 (so we get the same random split every time).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # YOUR CODE GOES HERE
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # 1. Pipeline execution
    raw_df = load_transaction_data("data/creditcard.csv")
    processed_df = preprocess_data(raw_df)
    balanced_df = create_balanced_sample(processed_df)
    
    # 2. Split the data
    X_train, X_test, y_train, y_test = prepare_data_for_training(balanced_df)
    
    # 3. Verification
    print("-" * 30)
    print(f"Training Features Shape (X_train): {X_train.shape}")
    print(f"Testing Features Shape (X_test): {X_test.shape}")
    print(f"Training Target Shape (y_train): {y_train.shape}")
    print(f"Testing Target Shape (y_test): {y_test.shape}")
    print("-" * 30)



def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Trains a Logistic Regression model and evaluates its baseline accuracy.
    """
    print("\nInitializing Logistic Regression Model...")
    model = LogisticRegression(random_state=42)
    
    # TASK 1: "Train" the model. 
    # Hint: In Scikit-Learn, training a model is always done using the .fit() method on your training data.
    # YOUR CODE HERE
    model.fit(X_train, y_train)
    
    # TASK 2: Make predictions.
    # Hint: You want the model to predict outcomes for the testing features. Use the .predict() method.
    # predictions = # YOUR CODE HERE
    predictions = model.predict(X_test)
    # Calculate basic accuracy
    acc = accuracy_score(y_test, predictions)
    print(f"Baseline Model Accuracy: {acc * 100:.2f}%")
    
    # --- NEW EVALUATION SECTION ---
    print("\n" + "="*30)
    print("CONFUSION MATRIX:")
    print(confusion_matrix(y_test, predictions))
    
    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_test, predictions))
    print("="*30)

    return model

def compare_models(X_train, X_test, y_train, y_test):
    """
    Trains and compares Logistic Regression vs Random Forest.
    """
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        
        print(f"{name} Classification Report:")
        print(classification_report(y_test, predictions))
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, predictions)}")

if __name__ == "__main__":
    # 1. Pipeline execution
    raw_df = load_transaction_data("data/creditcard.csv")
    processed_df = preprocess_data(raw_df)
    balanced_df = create_balanced_sample(processed_df)
    
    # 2. Split the data
    X_train, X_test, y_train, y_test = prepare_data_for_training(balanced_df)
    
    # 3. Train and Evaluate
    trained_model = train_and_evaluate(X_train, X_test, y_train, y_test)

    # 4. Compare Models
    raw_df = load_transaction_data("data/creditcard.csv")
    processed_df = preprocess_data(raw_df)
    balanced_df = create_balanced_sample(processed_df)
    X_train, X_test, y_train, y_test = prepare_data_for_training(balanced_df)
    compare_models(X_train, X_test, y_train, y_test)

