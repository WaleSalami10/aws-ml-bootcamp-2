#!/usr/bin/env python3
"""
Neural Network Comparison Script
Demonstrates the improvements made to the sigmoid neural network implementation.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Set random seed for reproducibility
np.random.seed(42)

def sigmoid(x):
    """Sigmoid activation function with numerical stability."""
    x_clipped = np.clip(x, -500, 500)
    return 1 / (1 + np.exp(-x_clipped))

def sigmoid_derivative(x):
    """Derivative of sigmoid function."""
    return x * (1 - x)

class SimpleNeuralNetwork:
    """Simple neural network implementation for comparison."""
    
    def __init__(self, input_size, hidden_size, output_size, use_xavier=True):
        if use_xavier:
            # Xavier initialization
            self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
            self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
        else:
            # Uniform initialization (original approach)
            self.W1 = np.random.uniform(size=(input_size, hidden_size))
            self.W2 = np.random.uniform(size=(hidden_size, output_size))
        
        self.losses = []
    
    def forward(self, X):
        """Forward propagation."""
        self.z1 = np.dot(X, self.W1)
        self.a1 = sigmoid(self.z1)
        self.z2 = np.dot(self.a1, self.W2)
        self.a2 = sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        """Backward propagation."""
        # Output layer
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)
        
        # Hidden layer
        hidden_error = np.dot(output_delta, self.W2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.a1)
        
        return output_delta, hidden_delta
    
    def update_weights(self, X, output_delta, hidden_delta, learning_rate):
        """Update network weights."""
        self.W2 += learning_rate * np.dot(self.a1.T, output_delta)
        self.W1 += learning_rate * np.dot(X.T, hidden_delta)
    
    def compute_loss(self, y_true, y_pred):
        """Compute mean squared error loss."""
        return np.mean((y_true - y_pred) ** 2)
    
    def train(self, X, y, epochs, learning_rate):
        """Train the network."""
        for epoch in range(epochs):
            # Forward pass
            output = self.forward(X)
            
            # Compute loss
            loss = self.compute_loss(y, output)
            self.losses.append(loss)
            
            # Backward pass
            output_delta, hidden_delta = self.backward(X, y, output)
            
            # Update weights
            self.update_weights(X, output_delta, hidden_delta, learning_rate)
        
        return output

def create_comparison_visualization():
    """Create visualization comparing original vs improved implementation."""
    
    # Training data
    X = np.array([[0, 1]])
    y = np.array([[1]])
    
    # Training parameters
    epochs = 1000
    learning_rate = 0.1
    
    # Train original network (uniform initialization)
    print("Training original network (uniform initialization)...")
    original_net = SimpleNeuralNetwork(2, 3, 1, use_xavier=False)
    original_output = original_net.train(X, y, epochs, learning_rate)
    
    # Train improved network (Xavier initialization)
    print("Training improved network (Xavier initialization)...")
    improved_net = SimpleNeuralNetwork(2, 3, 1, use_xavier=True)
    improved_output = improved_net.train(X, y, epochs, learning_rate)
    
    # Create comprehensive visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Loss comparison
    ax1.plot(original_net.losses, 'r-', linewidth=2, label='Original (Uniform Init)', alpha=0.8)
    ax1.plot(improved_net.losses, 'b-', linewidth=2, label='Improved (Xavier Init)', alpha=0.8)
    ax1.set_title('Training Loss Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Mean Squared Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Sigmoid function properties
    x = np.linspace(-10, 10, 1000)
    sigmoid_vals = sigmoid(x)
    sigmoid_deriv = sigmoid_vals * (1 - sigmoid_vals)
    
    ax2.plot(x, sigmoid_vals, 'b-', linewidth=2, label='Sigmoid σ(x)')
    ax2.plot(x, sigmoid_deriv, 'r--', linewidth=2, label="Derivative σ'(x)")
    ax2.set_title('Sigmoid Function Properties', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Input (x)')
    ax2.set_ylabel('Output')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    
    # 3. Weight initialization comparison
    np.random.seed(42)
    uniform_weights = np.random.uniform(size=1000)
    xavier_weights = np.random.randn(1000) * np.sqrt(1.0 / 2)
    
    ax3.hist(uniform_weights, bins=30, alpha=0.7, label='Uniform [0,1]', color='red')
    ax3.hist(xavier_weights, bins=30, alpha=0.7, label='Xavier Init', color='blue')
    ax3.set_title('Weight Initialization Comparison', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Weight Value')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Network architecture and results
    ax4.text(0.5, 0.9, 'Neural Network Analysis', ha='center', fontsize=16, fontweight='bold')
    ax4.text(0.5, 0.8, 'Architecture: 2 → 3 → 1', ha='center', fontsize=14)
    ax4.text(0.5, 0.7, 'Activation: Sigmoid', ha='center', fontsize=14)
    ax4.text(0.5, 0.6, 'Task: Learn [0,1] → [1]', ha='center', fontsize=14)
    
    ax4.text(0.5, 0.45, 'Results:', ha='center', fontsize=14, fontweight='bold')
    ax4.text(0.5, 0.35, f'Original Final Output: {original_output[0][0]:.6f}', 
             ha='center', fontsize=12, color='red')
    ax4.text(0.5, 0.25, f'Improved Final Output: {improved_output[0][0]:.6f}', 
             ha='center', fontsize=12, color='blue')
    ax4.text(0.5, 0.15, f'Target: {y[0][0]:.6f}', ha='center', fontsize=12)
    
    ax4.text(0.5, 0.05, f'Final Loss - Original: {original_net.losses[-1]:.6f}', 
             ha='center', fontsize=10, color='red')
    ax4.text(0.5, 0.01, f'Final Loss - Improved: {improved_net.losses[-1]:.6f}', 
             ha='center', fontsize=10, color='blue')
    
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    
    plt.tight_layout()
    
    # Save the plot
    os.makedirs('Neural Networks/images', exist_ok=True)
    plt.savefig('Neural Networks/images/neural_network_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\nComparison Results:")
    print(f"Original Network - Final Output: {original_output[0][0]:.6f}, Final Loss: {original_net.losses[-1]:.6f}")
    print(f"Improved Network - Final Output: {improved_output[0][0]:.6f}, Final Loss: {improved_net.losses[-1]:.6f}")
    print(f"Target: {y[0][0]}")
    print("\nVisualization saved to Neural Networks/images/neural_network_comparison.png")

def demonstrate_xor_problem():
    """Demonstrate the XOR problem solution."""
    
    # XOR dataset
    X_xor = np.array([[0, 0],
                      [0, 1],
                      [1, 0],
                      [1, 1]])
    
    y_xor = np.array([[0],
                      [1],
                      [1],
                      [0]])
    
    print("\nDemonstrating XOR Problem Solution:")
    print("Input (X1, X2) -> Target")
    for i in range(len(X_xor)):
        print(f"({X_xor[i][0]}, {X_xor[i][1]}) -> {y_xor[i][0]}")
    
    # Create and train network for XOR
    xor_net = SimpleNeuralNetwork(2, 4, 1, use_xavier=True)
    
    # Train on XOR problem
    epochs = 2000
    learning_rate = 1.0
    
    for epoch in range(epochs):
        total_loss = 0
        for i in range(len(X_xor)):
            # Forward pass
            output = xor_net.forward(X_xor[i:i+1])
            
            # Compute loss
            loss = xor_net.compute_loss(y_xor[i:i+1], output)
            total_loss += loss
            
            # Backward pass
            output_delta, hidden_delta = xor_net.backward(X_xor[i:i+1], y_xor[i:i+1], output)
            
            # Update weights
            xor_net.update_weights(X_xor[i:i+1], output_delta, hidden_delta, learning_rate)
        
        avg_loss = total_loss / len(X_xor)
        xor_net.losses.append(avg_loss)
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d}, Average Loss: {avg_loss:.6f}")
    
    # Test final predictions
    print("\nFinal XOR Results:")
    print("Input\t\tTarget\tPrediction")
    for i in range(len(X_xor)):
        prediction = xor_net.forward(X_xor[i:i+1])
        print(f"{X_xor[i]}\t{y_xor[i][0]}\t{prediction[0][0]:.4f}")
    
    return xor_net

if __name__ == "__main__":
    print("Neural Network Comparison and Analysis")
    print("=" * 50)
    
    # Create comparison visualization
    create_comparison_visualization()
    
    # Demonstrate XOR problem
    xor_network = demonstrate_xor_problem()
    
    print("\nAnalysis complete! Check the generated visualizations.")