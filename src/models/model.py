"""
Machine Learning Models Module

This module contains functions for creating, training, and using
machine learning models for diabetes prediction.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pickle
import os


def create_model(model_type='logistic'):
    """
    Create and return a machine learning model.
    
    Args:
        model_type (str): Type of model to create
            - 'logistic': Logistic Regression (default)
            - 'random_forest': Random Forest Classifier
            - 'svm': Support Vector Machine
    
    Returns:
        model: Initialized sklearn model
    """
    if model_type == 'logistic':
        model = LogisticRegression(
            max_iter=200, 
            random_state=42,
            solver='lbfgs'
        )
        print(f"Created Logistic Regression model")
    
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        print(f"Created Random Forest model")
    
    elif model_type == 'svm':
        model = SVC(
            kernel='rbf',
            random_state=42,
            probability=True
        )
        print(f"Created SVM model")
    
    else:
        print(f"Unknown model type '{model_type}'. Using Logistic Regression.")
        model = LogisticRegression(max_iter=200, random_state=42)
    
    return model


def train_model(model, X_train, y_train):
    """
    Train the model on training data.
    
    Args:
        model: sklearn model object
        X_train: Training features (numpy array or pandas DataFrame)
        y_train: Training labels (numpy array or pandas Series)
    
    Returns:
        model: Trained model
    """
    print(f"Training {model.__class__.__name__}...")
    model.fit(X_train, y_train)
    print(f"✓ Model training complete")
    return model


def predict(model, X_test):
    """
    Make predictions on test data.
    
    Args:
        model: Trained sklearn model
        X_test: Test features (numpy array or pandas DataFrame)
    
    Returns:
        predictions: Predicted class labels (numpy array)
    """
    predictions = model.predict(X_test)
    print(f"Made {len(predictions)} predictions")
    return predictions


def predict_proba(model, X_test):
    """
    Get prediction probabilities.
    
    Args:
        model: Trained sklearn model
        X_test: Test features (numpy array or pandas DataFrame)
    
    Returns:
        probabilities: Prediction probabilities (numpy array)
            Shape: (n_samples, n_classes)
    """
    if hasattr(model, 'predict_proba'):
        probabilities = model.predict_proba(X_test)
        return probabilities
    else:
        print("Warning: Model doesn't support probability predictions")
        return None


def save_model(model, filepath='models/trained_model.pkl'):
    """
    Save trained model to disk.
    
    Args:
        model: Trained sklearn model
        filepath (str): Path where model will be saved
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to {filepath}")


def load_model(filepath='models/trained_model.pkl'):
    """
    Load trained model from disk.
    
    Args:
        filepath (str): Path to saved model
    
    Returns:
        model: Loaded sklearn model
    """
    if not os.path.exists(filepath):
        print(f"Error: Model file not found at {filepath}")
        return None
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    print(f"✓ Model loaded from {filepath}")
    return model


def get_feature_importance(model, feature_names=None):
    """
    Get feature importance from the model (if available).
    
    Args:
        model: Trained sklearn model
        feature_names (list): List of feature names (optional)
    
    Returns:
        dict: Feature names and their importance scores
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]
        
        # Create dictionary of feature:importance
        feature_dict = dict(zip(feature_names, importances))
        
        # Sort by importance
        sorted_features = sorted(
            feature_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        print("\nFeature Importances:")
        for feature, importance in sorted_features:
            print(f"  {feature}: {importance:.4f}")
        
        return dict(sorted_features)
    
    elif hasattr(model, 'coef_'):
        # For linear models (like Logistic Regression)
        coefficients = model.coef_[0]
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(coefficients))]
        
        # Get absolute values for importance
        abs_coef = abs(coefficients)
        feature_dict = dict(zip(feature_names, abs_coef))
        
        # Sort by importance
        sorted_features = sorted(
            feature_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        print("\nFeature Importance (Absolute Coefficients):")
        for feature, coef in sorted_features:
            print(f"  {feature}: {coef:.4f}")
        
        return dict(sorted_features)
    
    else:
        print("Model doesn't support feature importance")
        return None


# Example usage
if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_classification
    
    # Generate sample data
    X, y = make_classification(
        n_samples=100, 
        n_features=8, 
        n_classes=2, 
        random_state=42
    )
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Create and train model
    print("=== Testing Model Module ===\n")
    model = create_model('logistic')
    model = train_model(model, X_train, y_train)
    
    # Make predictions
    predictions = predict(model, X_test)
    print(f"Sample predictions: {predictions[:5]}")
    
    # Get probabilities
    probs = predict_proba(model, X_test)
    if probs is not None:
        print(f"Sample probabilities:\n{probs[:3]}")
    
    # Get feature importance
    feature_names = [f"Feature_{i+1}" for i in range(8)]
    get_feature_importance(model, feature_names)
    
    # Save model
    save_model(model, 'test_model.pkl')
    
    # Load model
    loaded_model = load_model('test_model.pkl')
    
    print("\n✓ All model functions working correctly!")