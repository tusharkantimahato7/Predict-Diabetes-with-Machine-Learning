"""
Main Execution Script 

This is the entry point of the project.
Run this file to execute the complete ML pipeline: 
1. Load data 
2. Preprossing 
3.Train Model 
4.Evaluate 
5.Make Predictions

Usage: python main.py
"""

import sys
import os

# Add src to path so we can import our modules 
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))