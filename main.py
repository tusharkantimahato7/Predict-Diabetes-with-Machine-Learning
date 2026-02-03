from src.data.data_loader import load_data, save_data
from src.features.preprocessing import split_features_target, split_train_test, scale_features
from src.models.model import create_model, train_model, predict
from src.evaluation.evaluation import evaluate_model
from src.utils.helpers import print_separator, ensure_dir

def main():
    print_separator("Diabetes Prediction ML Pipeline")
    
    # Ensure directories exist
    ensure_dir('data/processed')
    
    # Load data
    print("\n[1] Loading data...")
    data = load_data('data/raw/diabetes.csv')
    
    # Save cleaned data (same as raw in this simple case)
    save_data(data, 'data/processed/cleaned_diabetes.csv')
    
    # Split features and target
    print("\n[2] Splitting features and target...")
    X, y = split_features_target(data)
    
    # Train-test split
    print("\n[3] Creating train-test split...")
    X_train, X_test, y_train, y_test = split_train_test(X, y)
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Scale features
    print("\n[4] Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Create and train model
    print("\n[5] Training model...")
    model = create_model()
    model = train_model(model, X_train_scaled, y_train)
    
    # Make predictions
    print("\n[6] Making predictions...")
    predictions = predict(model, X_test_scaled)
    
    # Evaluate
    print_separator("Model Evaluation")
    evaluate_model(y_test, predictions)
    
    print_separator("Pipeline Complete")

if __name__ == "__main__":
    main()