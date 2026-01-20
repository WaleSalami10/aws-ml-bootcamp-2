#!/usr/bin/env python3
"""
ML Training Task Template
Follows the project's educational approach with from-scratch implementations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

class MLTrainingTask:
    def __init__(self, model_name, data_path, output_dir="models"):
        self.model_name = model_name
        self.data_path = data_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Training parameters
        self.learning_rate = 0.01
        self.num_iterations = 1000
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def load_data(self):
        """Load and prepare training data"""
        print(f"Loading data from {self.data_path}")
        self.data = pd.read_csv(self.data_path)
        print(f"Dataset shape: {self.data.shape}")
        return self.data
    
    def preprocess_data(self):
        """Implement feature scaling and data preparation"""
        # Override this method for specific preprocessing
        pass
    
    def initialize_parameters(self, n_features):
        """Initialize model parameters"""
        self.weights = np.random.randn(n_features) * 0.01
        self.bias = 0.0
        print(f"Initialized {n_features} weights and bias")
    
    def forward_pass(self, X):
        """Compute predictions - override for specific models"""
        return np.dot(X, self.weights) + self.bias
    
    def compute_cost(self, y_true, y_pred):
        """Compute cost function - override for specific models"""
        return np.mean((y_true - y_pred) ** 2) / 2
    
    def compute_gradients(self, X, y_true, y_pred):
        """Compute gradients - override for specific models"""
        m = len(y_true)
        dw = (1/m) * np.dot(X.T, (y_pred - y_true))
        db = (1/m) * np.sum(y_pred - y_true)
        return dw, db
    
    def update_parameters(self, dw, db):
        """Update model parameters"""
        self.weights = self.weights - self.learning_rate * dw
        self.bias = self.bias - self.learning_rate * db
    
    def train(self, X, y):
        """Main training loop"""
        print(f"Starting training for {self.model_name}")
        print(f"Learning rate: {self.learning_rate}, Iterations: {self.num_iterations}")
        
        for i in range(self.num_iterations):
            # Forward pass
            y_pred = self.forward_pass(X)
            
            # Compute cost
            cost = self.compute_cost(y, y_pred)
            self.cost_history.append(cost)
            
            # Compute gradients
            dw, db = self.compute_gradients(X, y, y_pred)
            
            # Update parameters
            self.update_parameters(dw, db)
            
            # Progress reporting
            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")
        
        print(f"Training completed! Final cost: {self.cost_history[-1]:.4f}")
    
    def evaluate(self, X, y):
        """Evaluate model performance"""
        y_pred = self.forward_pass(X)
        
        # R² Score
        residual_ss = np.sum((y - y_pred) ** 2)
        total_ss = np.sum((y - np.mean(y)) ** 2)
        r2 = 1 - (residual_ss / total_ss)
        
        # MSE
        mse = np.mean((y - y_pred) ** 2)
        
        print(f"Model Evaluation:")
        print(f"R² Score: {r2:.4f}")
        print(f"MSE: {mse:.4f}")
        
        return {"r2_score": r2, "mse": mse}
    
    def plot_learning_curve(self):
        """Plot training progress"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title(f'{self.model_name} - Learning Curve')
        plt.grid(True)
        
        # Save plot
        plot_path = self.output_dir / f"{self.model_name}_learning_curve.png"
        plt.savefig(plot_path)
        print(f"Learning curve saved to {plot_path}")
        plt.show()
    
    def save_model(self):
        """Save trained model parameters"""
        model_data = {
            'model_name': self.model_name,
            'weights': self.weights.tolist(),
            'bias': float(self.bias),
            'learning_rate': self.learning_rate,
            'num_iterations': self.num_iterations,
            'final_cost': self.cost_history[-1]
        }
        
        model_path = self.output_dir / f"{self.model_name}_model.json"
        import json
        with open(model_path, 'w') as f:
            json.dump(model_data, f, indent=2)
        
        print(f"Model saved to {model_path}")
    
    def run_training_pipeline(self):
        """Complete training pipeline"""
        # Load data
        data = self.load_data()
        
        # Preprocess (implement in subclass)
        self.preprocess_data()
        
        # Prepare features and target (implement in subclass)
        X, y = self.prepare_features_target(data)
        
        # Initialize parameters
        self.initialize_parameters(X.shape[1])
        
        # Train model
        self.train(X, y)
        
        # Evaluate
        metrics = self.evaluate(X, y)
        
        # Visualize
        self.plot_learning_curve()
        
        # Save model
        self.save_model()
        
        return metrics

# Example usage
if __name__ == "__main__":
    # This would be implemented by specific algorithm classes
    print("ML Training Task Template")
    print("Extend this class for specific algorithms (Linear Regression, Logistic Regression, etc.)")