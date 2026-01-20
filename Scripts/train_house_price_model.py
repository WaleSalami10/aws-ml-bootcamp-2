#!/usr/bin/env python3
"""
House Price Prediction Training Task
Extends the base training template for linear regression
"""

import sys
import numpy as np
import pandas as pd
from train_model import MLTrainingTask

class HousePriceTrainingTask(MLTrainingTask):
    def __init__(self):
        super().__init__(
            model_name="house_price_linear_regression",
            data_path="../Data/house_prices.csv"
        )
    
    def preprocess_data(self):
        """Feature scaling using min-max normalization"""
        feature_cols = ['size_sqft', 'bedrooms', 'age_years']
        
        for col in feature_cols:
            col_min = self.data[col].min()
            col_max = self.data[col].max()
            self.data[f'{col}_scaled'] = (self.data[col] - col_min) / (col_max - col_min)
        
        print("Applied min-max scaling to features")
    
    def prepare_features_target(self, data):
        """Prepare scaled features and target variable"""
        # Use scaled features
        feature_cols = ['size_sqft_scaled', 'bedrooms_scaled', 'age_years_scaled']
        X = data[feature_cols].values
        y = data['price'].values
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        
        return X, y

def main():
    """Run house price model training"""
    print("🏠 House Price Prediction Training Task")
    print("=" * 50)
    
    # Create and run training task
    task = HousePriceTrainingTask()
    
    try:
        metrics = task.run_training_pipeline()
        
        print("\n✅ Training completed successfully!")
        print(f"Final R² Score: {metrics['r2_score']:.4f}")
        print(f"Final MSE: {metrics['mse']:.4f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())