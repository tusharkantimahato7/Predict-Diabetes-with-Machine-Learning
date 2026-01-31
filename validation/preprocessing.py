"""
Data Preprocessing Module

This module handles all data cleaning and transformation tasks.
We deal with invalid zero values, split features from target,
and standardize the features for better model performance.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def clean_data(data):
    """
    Clean the dataset by handling zero values that don't make medical sense.
    
    Args:
        data (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
        
    Why this step?
    - In medical data, zero values for Glucose, BP, BMI etc. are impossible
    - These are actually missing values coded as 0
    - We replace them with the median of that column (robust to outliers)
    """
    data_clean = data.copy()
    
    # These columns cannot logically have zero values
    zero_not_allowed = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    
    print("\nCleaning zero values...")
    for col in zero_not_allowed:
        # Count zeros before replacement
        zero_count = (data_clean[col] == 0).sum()
        if zero_count > 0:
            # Replace zeros with NaN
            data_clean[col] = data_clean[col].replace(0, np.nan)
            # Fill NaN with median
            median_val = data_clean[col].median()
            data_clean[col].fillna(median_val, inplace=True)
            print(f"  {col}: replaced {zero_count} zeros with median ({median_val:.2f})")
    
    print("✓ Data cleaning completed")
    return data_clean


def prepare_features_and_target(data):
    """
    Separate features (X) and target variable (y).
    
    Args:
        data (pd.DataFrame): Cleaned dataset
        
    Returns:
        tuple: (X, y) where X is features and y is target
        
    Why this step?
    - ML models need input (X) and output (y) separated
    - Standard practice in supervised learning
    - Makes code cleaner and easier to understand
    """
    X = data.drop('Outcome', axis=1)
    y = data['Outcome']
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Feature columns: {list(X.columns)}")
    
    return X, y


def split_dataset(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and testing sets.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target variable
        test_size (float): Proportion of test set (default 20%)
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
        
    Why this step?
    - We need separate data to train and evaluate our model
    - Training set: teaches the model
    - Test set: checks if model works on unseen data
    - 80-20 split is industry standard for medium datasets
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Ensures same class distribution in train and test
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Train positive class: {y_train.sum()} ({y_train.sum()/len(y_train)*100:.1f}%)")
    print(f"Test positive class: {y_test.sum()} ({y_test.sum()/len(y_test)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):
    """
    Standardize features to have mean=0 and std=1.
    
    Args:
        X_train (pd.DataFrame): Training features
        X_test (pd.DataFrame): Testing features
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, scaler)
        
    Why this step?
    - Features have different ranges (Age: 20-80, Glucose: 0-200, etc.)
    - ML algorithms work better when features are on similar scales
    - Prevents features with large values from dominating the model
    - StandardScaler: transforms to mean=0, std=1
    
    Important: We fit scaler ONLY on training data to prevent data leakage!
    """
    scaler = StandardScaler()
    
    # Fit and transform training data
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Transform test data using the same scaler
    X_test_scaled = scaler.transform(X_test)
    
    print("\n✓ Features scaled using StandardScaler")
    print(f"  Scaler fitted on training set only")
    
    return X_train_scaled, X_test_scaled, scaler


def preprocess_data(data):
    """
    Complete preprocessing pipeline - calls all preprocessing functions.
    
    Args:
        data (pd.DataFrame): Raw dataset
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler)
        
    Why this function?
    - Single entry point for all preprocessing
    - Ensures steps happen in correct order
    - Makes main script cleaner
    """
    print("\n" + "="*60)
    print("PREPROCESSING PIPELINE")
    print("="*60)
    
    # Step 1: Clean data
    data_clean = clean_data(data)
    
    # Step 2: Separate features and target
    X, y = prepare_features_and_target(data_clean)
    
    # Step 3: Split into train and test
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    
    # Step 4: Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    print("\n✓ Preprocessing completed successfully")
    print("="*60)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


if __name__ == "__main__":
    # Test the module independently
    from data_loader import load_dataset
    
    data = load_dataset("data/diabetes.csv")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    print(f"\nFinal shapes:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_test: {X_test.shape}")