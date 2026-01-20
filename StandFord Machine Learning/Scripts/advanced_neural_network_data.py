#!/usr/bin/env python3
"""
Advanced Neural Network Dataset Generator

This script generates complex datasets for training advanced neural networks.
It creates non-linear, multi-dimensional datasets that require deep learning to solve.

Usage:
    python Scripts/advanced_neural_network_data.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def generate_complex_regression_dataset(n_samples=1000, noise_level=0.1, random_state=42):
    """
    Generate a complex 4D regression dataset with non-linear patterns
    
    This dataset combines:
    - Trigonometric functions
    - Polynomial terms
    - Exponential decay
    - Interaction terms
    
    Much more complex than simple linear relationships!
    """
    np.random.seed(random_state)
    
    # Generate 4D input features in different ranges
    X = np.zeros((n_samples, 4))
    X[:, 0] = np.random.uniform(-2, 2, n_samples)    # Feature 1: [-2, 2]
    X[:, 1] = np.random.uniform(-1, 3, n_samples)    # Feature 2: [-1, 3]
    X[:, 2] = np.random.uniform(0, 4, n_samples)     # Feature 3: [0, 4]
    X[:, 3] = np.random.uniform(-3, 1, n_samples)    # Feature 4: [-3, 1]
    
    # Complex non-linear target function
    y = (
        # Trigonometric interactions
        0.3 * np.sin(X[:, 0] * X[:, 1]) +
        0.25 * np.cos(X[:, 2] * 0.5) * np.sin(X[:, 3]) +
        
        # Polynomial terms
        0.2 * (X[:, 0] ** 2 + X[:, 1] ** 2) * 0.1 +
        0.15 * X[:, 2] * X[:, 3] +
        
        # Exponential decay
        0.1 * np.exp(-0.5 * (X[:, 2] ** 2 + X[:, 3] ** 2)) +
        
        # Linear terms for baseline
        0.1 * X[:, 0] + 0.05 * X[:, 1] - 0.08 * X[:, 2] + 0.12 * X[:, 3]
    )
    
    # Add noise
    y += np.random.normal(0, noise_level, n_samples)
    
    # Normalize output to [0, 1] range for sigmoid output
    y_min, y_max = y.min(), y.max()
    y_normalized = (y - y_min) / (y_max - y_min)
    
    return X, y_normalized.reshape(-1, 1), y_min, y_max

def generate_classification_dataset(n_samples=1000, random_state=42):
    """
    Generate a complex 4D classification dataset
    
    Creates non-linearly separable classes that require deep learning
    """
    np.random.seed(random_state)
    
    # Generate features
    X = np.random.uniform(-2, 2, (n_samples, 4))
    
    # Complex decision boundary
    decision_value = (
        np.sin(X[:, 0] + X[:, 1]) +
        np.cos(X[:, 2] * X[:, 3]) +
        0.5 * (X[:, 0] ** 2 + X[:, 1] ** 2 - X[:, 2] ** 2 - X[:, 3] ** 2)
    )
    
    # Create binary labels
    y = (decision_value > np.median(decision_value)).astype(float)
    
    return X, y.reshape(-1, 1)

def generate_time_series_dataset(n_samples=1000, sequence_length=10, random_state=42):
    """
    Generate a time series dataset for sequence learning
    
    Creates patterns that change over time
    """
    np.random.seed(random_state)
    
    # Generate time series with multiple components
    t = np.linspace(0, 4*np.pi, n_samples)
    
    # Multiple sine waves with different frequencies and phases
    series = (
        np.sin(t) +
        0.5 * np.sin(3*t + np.pi/4) +
        0.3 * np.sin(5*t + np.pi/2) +
        0.1 * np.random.normal(0, 1, n_samples)  # noise
    )
    
    # Create sequences
    X, y = [], []
    for i in range(len(series) - sequence_length):
        X.append(series[i:i+sequence_length])
        y.append(series[i+sequence_length])
    
    return np.array(X), np.array(y).reshape(-1, 1)

def save_datasets():
    """Generate and save all datasets"""
    
    # Create Data directory if it doesn't exist
    data_dir = Path("Data")
    data_dir.mkdir(exist_ok=True)
    
    print("🔄 Generating advanced neural network datasets...")
    
    # 1. Complex Regression Dataset
    print("📊 Generating complex regression dataset...")
    X_reg, y_reg, y_min, y_max = generate_complex_regression_dataset(n_samples=1200)
    
    # Split into train/test
    train_size = 800
    X_train_reg = X_reg[:train_size]
    y_train_reg = y_reg[:train_size]
    X_test_reg = X_reg[train_size:]
    y_test_reg = y_reg[train_size:]
    
    # Save regression data
    reg_train_df = pd.DataFrame(X_train_reg, columns=['feature_1', 'feature_2', 'feature_3', 'feature_4'])
    reg_train_df['target'] = y_train_reg
    reg_train_df.to_csv(data_dir / "advanced_regression_train.csv", index=False)
    
    reg_test_df = pd.DataFrame(X_test_reg, columns=['feature_1', 'feature_2', 'feature_3', 'feature_4'])
    reg_test_df['target'] = y_test_reg
    reg_test_df.to_csv(data_dir / "advanced_regression_test.csv", index=False)
    
    print(f"   ✅ Saved regression data: {train_size} train, {len(X_test_reg)} test samples")
    
    # 2. Classification Dataset
    print("📊 Generating complex classification dataset...")
    X_cls, y_cls = generate_classification_dataset(n_samples=1200)
    
    # Split into train/test
    X_train_cls = X_cls[:train_size]
    y_train_cls = y_cls[:train_size]
    X_test_cls = X_cls[train_size:]
    y_test_cls = y_cls[train_size:]
    
    # Save classification data
    cls_train_df = pd.DataFrame(X_train_cls, columns=['feature_1', 'feature_2', 'feature_3', 'feature_4'])
    cls_train_df['target'] = y_train_cls
    cls_train_df.to_csv(data_dir / "advanced_classification_train.csv", index=False)
    
    cls_test_df = pd.DataFrame(X_test_cls, columns=['feature_1', 'feature_2', 'feature_3', 'feature_4'])
    cls_test_df['target'] = y_test_cls
    cls_test_df.to_csv(data_dir / "advanced_classification_test.csv", index=False)
    
    print(f"   ✅ Saved classification data: {train_size} train, {len(X_test_cls)} test samples")
    
    # 3. Time Series Dataset
    print("📊 Generating time series dataset...")
    X_ts, y_ts = generate_time_series_dataset(n_samples=1000, sequence_length=10)
    
    # Save time series data
    ts_df = pd.DataFrame(X_ts, columns=[f'lag_{i+1}' for i in range(10)])
    ts_df['target'] = y_ts
    ts_df.to_csv(data_dir / "time_series_sequences.csv", index=False)
    
    print(f"   ✅ Saved time series data: {len(X_ts)} sequences")
    
    return {
        'regression': (X_train_reg, y_train_reg, X_test_reg, y_test_reg),
        'classification': (X_train_cls, y_train_cls, X_test_cls, y_test_cls),
        'time_series': (X_ts, y_ts)
    }

def visualize_datasets(datasets):
    """Create visualizations of the generated datasets"""
    
    print("\n📈 Creating dataset visualizations...")
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    
    # Regression dataset
    X_train_reg, y_train_reg, X_test_reg, y_test_reg = datasets['regression']
    
    for i in range(4):
        axes[0, i].scatter(X_train_reg[:, i], y_train_reg, alpha=0.5, s=10, label='Train')
        axes[0, i].scatter(X_test_reg[:, i], y_test_reg, alpha=0.5, s=10, label='Test')
        axes[0, i].set_xlabel(f'Feature {i+1}')
        axes[0, i].set_ylabel('Target')
        axes[0, i].set_title(f'Regression: Feature {i+1} vs Target')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
    
    # Classification dataset
    X_train_cls, y_train_cls, X_test_cls, y_test_cls = datasets['classification']
    
    for i in range(4):
        # Color by class
        class_0_mask = y_train_cls.flatten() == 0
        class_1_mask = y_train_cls.flatten() == 1
        
        axes[1, i].scatter(X_train_cls[class_0_mask, i], X_train_cls[class_0_mask, (i+1)%4], 
                          alpha=0.5, s=10, label='Class 0', c='blue')
        axes[1, i].scatter(X_train_cls[class_1_mask, i], X_train_cls[class_1_mask, (i+1)%4], 
                          alpha=0.5, s=10, label='Class 1', c='red')
        axes[1, i].set_xlabel(f'Feature {i+1}')
        axes[1, i].set_ylabel(f'Feature {(i+1)%4 + 1}')
        axes[1, i].set_title(f'Classification: Features {i+1} vs {(i+1)%4 + 1}')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    # Time series dataset
    X_ts, y_ts = datasets['time_series']
    
    # Show first few sequences
    for i in range(4):
        axes[2, i].plot(X_ts[i*50], 'b-', linewidth=2, label='Input Sequence')
        axes[2, i].axhline(y=y_ts[i*50], color='red', linestyle='--', linewidth=2, label='Target')
        axes[2, i].set_xlabel('Time Step')
        axes[2, i].set_ylabel('Value')
        axes[2, i].set_title(f'Time Series Example {i+1}')
        axes[2, i].legend()
        axes[2, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Neural Networks/images/advanced_datasets_overview.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("   ✅ Saved visualization to 'Neural Networks/images/advanced_datasets_overview.png'")

def main():
    """Main function to generate all datasets and visualizations"""
    
    print("🚀 Advanced Neural Network Dataset Generator")
    print("=" * 50)
    
    # Generate and save datasets
    datasets = save_datasets()
    
    # Create visualizations
    visualize_datasets(datasets)
    
    # Print summary
    print("\n📋 DATASET SUMMARY")
    print("=" * 30)
    print("Generated datasets:")
    print("  📊 Complex Regression (4D → 1D)")
    print("     - Non-linear patterns with trigonometric and polynomial terms")
    print("     - 800 training samples, 400 test samples")
    print("     - Saved as: advanced_regression_train.csv, advanced_regression_test.csv")
    
    print("  🎯 Complex Classification (4D → Binary)")
    print("     - Non-linearly separable classes")
    print("     - 800 training samples, 400 test samples")
    print("     - Saved as: advanced_classification_train.csv, advanced_classification_test.csv")
    
    print("  📈 Time Series Sequences (10D → 1D)")
    print("     - Multi-frequency sine waves with noise")
    print("     - 990 sequences of length 10")
    print("     - Saved as: time_series_sequences.csv")
    
    print("\n🎯 These datasets are perfect for testing your advanced neural networks!")
    print("   They require deep learning to achieve good performance.")
    
    print("\n✅ All datasets generated successfully!")

if __name__ == "__main__":
    main()