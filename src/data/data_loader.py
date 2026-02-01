"""
Data Loader Module

This module handles loading the diabetes dataset and performing basic validation.
We check if the file exists, load it into a pandas DataFrame, and display
essential information about the data structure.
"""

import pandas as pd
import os


def load_dataset(filepath):
    """
    Load the diabetes dataset from a CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Loaded dataset
        
    Why this function?
    - Centralizes data loading logic
    - Makes it easy to switch data sources later
    - Handles file validation before loading
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    data = pd.read_csv(filepath)
    print(f"Dataset loaded successfully from {filepath}")
    return data


def display_basic_info(data):
    """
    Display basic information about the dataset.
    
    Args:
        data (pd.DataFrame): The dataset to analyze
        
    Why this function?
    - Helps us understand data structure quickly
    - Identifies potential issues early (missing values, data types)
    - Good practice before any preprocessing
    """
    print("\n" + "="*60)
    print("DATASET OVERVIEW")
    print("="*60)
    
    print(f"\nDataset Shape: {data.shape[0]} rows, {data.shape[1]} columns")
    
    print("\nFirst 5 Rows:")
    print(data.head())
    
    print("\nColumn Data Types:")
    print(data.dtypes)
    
    print("\nMissing Values:")
    print(data.isnull().sum())
    
    print("\nBasic Statistics:")
    print(data.describe())
    
    print("\nTarget Variable Distribution:")
    if 'Outcome' in data.columns:
        print(data['Outcome'].value_counts())
    
    print("="*60)


def validate_dataset(data):
    """
    Validate that the dataset has expected structure.
    
    Args:
        data (pd.DataFrame): Dataset to validate
        
    Returns:
        bool: True if valid, raises exception otherwise
        
    Why this function?
    - Ensures we're working with the correct dataset
    - Catches data issues before they cause errors later
    - Makes code more robust
    """
    expected_columns = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
    ]
    
    missing_cols = set(expected_columns) - set(data.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    print("✓ Dataset validation passed")
    return True


if __name__ == "__main__":
    # Test the module independently
    data = load_dataset("data/diabetes.csv")
    validate_dataset(data)
    display_basic_info(data)