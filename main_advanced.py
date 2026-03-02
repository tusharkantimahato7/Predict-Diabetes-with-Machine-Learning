"""
Advanced ML Pipeline with Cross-Validation and Hyperparameter Tuning
"""

from src.data.data_loader import load_data, save_data
from src.features.preprocessing import split_features_target, split_train_test, scale_features
from src.models.model import create_model, train_model, predict, get_feature_importance
from src.models.tuning import tune_hyperparameters
from src.evaluation.evaluation import evaluate_model
from src.evaluation.cross_validation import compare_models_cv
from src.utils.helpers import print_separator, ensure_dir
from src.utils.visualization import (plot_confusion_matrix, plot_roc_curve,plot_feature_importance, plot_model_comparison)


def main():
    print_separator("Advanced Diabetes Prediction Pipeline")
    
    # Ensure directories
    ensure_dir('data/processed')
    ensure_dir('models')
    ensure_dir('outputs')
    
    # 1. Load data
    print("\n[1] Loading data...")
    data = load_data('data/raw/diabetes.csv')
    save_data(data, 'data/processed/cleaned_diabetes.csv')
    
    # 2. Prepare features
    print("\n[2] Preparing features...")
    X, y = split_features_target(data)
    feature_names = X.columns.tolist()
    
    # 3. Train-test split
    print("\n[3] Creating train-test split...")
    X_train, X_test, y_train, y_test = split_train_test(X, y, test_size=0.3)
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")
    
    # 4. Scale features
    print("\n[4] Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # 5. Model comparison using cross-validation
    print_separator("Model Comparison")
    models = {
        'Logistic Regression': create_model('logistic'),
        'Random Forest': create_model('random_forest'),
        'SVM': create_model('svm')
    }
    cv_results = compare_models_cv(models, X_train_scaled, y_train, cv=5)
    
    # 6. Select best model and tune hyperparameters
    print_separator("Hyperparameter Tuning")
    best_model_name = max(cv_results, key=lambda k: cv_results[k]['mean'])
    print(f"\nBest model from CV: {best_model_name}")
    
    base_model = create_model('logistic')  # You can change this based on CV results
    tuned_model, tuning_results = tune_hyperparameters(
        base_model, X_train_scaled, y_train, cv=5
    )
    
    # 7. Final training
    print_separator("Final Model Training")
    print("Training tuned model on full training set...")
    final_model = train_model(tuned_model, X_train_scaled, y_train)
    
    # 8. Predictions
    print("\n[8] Making predictions...")
    predictions = predict(final_model, X_test_scaled)
    
    # 9. Evaluation
    print_separator("Model Evaluation")
    accuracy = evaluate_model(y_test, predictions)
    
    # 10. Feature importance
    importance = get_feature_importance(final_model, feature_names)
    
    # 11. Visualizations
    print_separator("Generating Visualizations")
    
    print("\n[11.1] Confusion Matrix...")
    plot_confusion_matrix(y_test, predictions, 'outputs/confusion_matrix.png')
    
    print("\n[11.2] ROC Curve...")
    roc_auc = plot_roc_curve(final_model, X_test_scaled, y_test, 'outputs/roc_curve.png')
    
    if importance:
        print("\n[11.3] Feature Importance...")
        plot_feature_importance(importance, 'outputs/feature_importance.png')
    
    print("\n[11.4] Model Comparison...")
    comparison = {name: res['mean'] for name, res in cv_results.items()}
    plot_model_comparison(comparison, 'outputs/model_comparison.png')
    
    # 12. Save final model
    from src.models.model import save_model
    save_model(final_model, 'models/final_tuned_model.pkl')
    
    print_separator("Pipeline Complete")
    print(f"\n✓ Final Accuracy: {accuracy:.4f}")
    if roc_auc:
        print(f"✓ ROC AUC Score: {roc_auc:.4f}")
    print(f"✓ Best Parameters: {tuning_results['best_params']}")
    print(f"✓ Visualizations saved to 'outputs/' folder")
    print(f"✓ Model saved to 'models/final_tuned_model.pkl'")


if __name__ == "__main__":
    main()