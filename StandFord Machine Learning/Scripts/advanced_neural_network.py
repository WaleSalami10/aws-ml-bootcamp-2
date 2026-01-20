#!/usr/bin/env python3
"""
Advanced Neural Network Implementation with NumPy

This script implements a sophisticated 4→6→4→1 neural network with:
- Multiple activation functions (ReLU, Sigmoid, Tanh, Leaky ReLU)
- Advanced weight initialization (Xavier/Glorot)
- Momentum and learning rate decay
- Complex dataset generation and training
- Comprehensive visualization and analysis

Building on your 2→2→1 foundation to create a deep learning system!
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from pathlib import Path

# Set random seed for reproducibility
np.random.seed(42)

class ActivationFunctions:
    """Collection of activation functions and their derivatives"""
    
    @staticmethod
    def sigmoid(x):
        """Your familiar sigmoid function - good for output layers"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of sigmoid"""
        s = ActivationFunctions.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def relu(x):
        """ReLU: Rectified Linear Unit - most popular for hidden layers"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """Derivative of ReLU"""
        return (x > 0).astype(float)
    
    @staticmethod
    def tanh(x):
        """Hyperbolic tangent - outputs between -1 and 1"""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        """Derivative of tanh"""
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """Leaky ReLU - fixes the 'dying ReLU' problem"""
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        """Derivative of Leaky ReLU"""
        return np.where(x > 0, 1, alpha)

class AdvancedNeuralNetwork:
    def __init__(self, layer_sizes=[4, 6, 4, 1], activations=['relu', 'relu', 'sigmoid']):
        """
        Initialize advanced neural network
        
        Args:
            layer_sizes: List of neurons in each layer [input, hidden1, hidden2, output]
            activations: List of activation functions for each layer transition
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)
        self.activations = activations
        
        print(f"🏗️ Building {' → '.join(map(str, layer_sizes))} neural network")
        print(f"📊 Activations: {' → '.join(activations)}")
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Advanced weight initialization (Xavier/Glorot initialization)
        for i in range(self.num_layers - 1):
            # Xavier initialization: better than random uniform
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            
            weight_matrix = np.random.uniform(-limit, limit, (fan_in, fan_out))
            bias_vector = np.zeros((1, fan_out))
            
            self.weights.append(weight_matrix)
            self.biases.append(bias_vector)
            
            print(f"Layer {i+1}: {fan_in} → {fan_out}, weights shape: {weight_matrix.shape}")
        
        print("✅ Advanced neural network initialized!")
    
    def get_activation_function(self, name):
        """Get activation function and its derivative"""
        functions = {
            'sigmoid': (ActivationFunctions.sigmoid, ActivationFunctions.sigmoid_derivative),
            'relu': (ActivationFunctions.relu, ActivationFunctions.relu_derivative),
            'tanh': (ActivationFunctions.tanh, ActivationFunctions.tanh_derivative),
            'leaky_relu': (ActivationFunctions.leaky_relu, ActivationFunctions.leaky_relu_derivative)
        }
        return functions[name]
    
    def forward_pass(self, inputs):
        """Advanced forward pass through all layers"""
        if inputs.ndim == 1:
            inputs = inputs.reshape(1, -1)
        
        # Store all intermediate values for backpropagation
        activations = [inputs]  # Store input as first activation
        z_values = []  # Store pre-activation values
        
        current_input = inputs
        
        # Forward pass through each layer
        for i in range(self.num_layers - 1):
            # Linear transformation: z = W*a + b
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            z_values.append(z)
            
            # Apply activation function
            activation_func, _ = self.get_activation_function(self.activations[i])
            a = activation_func(z)
            activations.append(a)
            
            current_input = a
        
        return {
            'activations': activations,  # All layer outputs (including input)
            'z_values': z_values,        # All pre-activation values
            'final_output': activations[-1]
        }
    
    def calculate_loss(self, predictions, targets):
        """Calculate Mean Squared Error loss"""
        if targets.ndim == 1:
            targets = targets.reshape(-1, 1)
        if predictions.ndim == 1:
            predictions = predictions.reshape(-1, 1)
        
        mse = np.mean((predictions - targets) ** 2)
        return mse

def advanced_backpropagation_with_momentum(network, forward_result, targets, 
                                         learning_rate, weight_momentum, 
                                         bias_momentum, momentum_factor):
    """
    Backpropagation with momentum for faster and more stable training
    
    Momentum helps by:
    - Accelerating in consistent directions
    - Dampening oscillations
    - Helping escape local minima
    """
    if targets.ndim == 1:
        targets = targets.reshape(-1, 1)
    
    activations = forward_result['activations']
    z_values = forward_result['z_values']
    
    # Calculate output error
    output_error = activations[-1] - targets
    
    # Initialize lists to store gradients
    weight_gradients = []
    bias_gradients = []
    
    # Backpropagate through each layer
    current_error = output_error
    
    for i in reversed(range(network.num_layers - 1)):
        # Get activation derivative
        _, activation_derivative = network.get_activation_function(network.activations[i])
        
        # Calculate delta
        delta = current_error * activation_derivative(z_values[i])
        
        # Calculate gradients
        weight_grad = np.dot(activations[i].T, delta) / activations[i].shape[0]  # Average over batch
        bias_grad = np.mean(delta, axis=0, keepdims=True)
        
        weight_gradients.append(weight_grad)
        bias_gradients.append(bias_grad)
        
        # Calculate error for previous layer
        if i > 0:
            current_error = np.dot(delta, network.weights[i].T)
    
    # Reverse gradients
    weight_gradients.reverse()
    bias_gradients.reverse()
    
    # Update weights and biases with momentum
    for i in range(network.num_layers - 1):
        # Update momentum terms
        weight_momentum[i] = momentum_factor * weight_momentum[i] + learning_rate * weight_gradients[i]
        bias_momentum[i] = momentum_factor * bias_momentum[i] + learning_rate * bias_gradients[i]
        
        # Update parameters
        network.weights[i] -= weight_momentum[i]
        network.biases[i] -= bias_momentum[i]
    
    # Return loss
    return np.mean(output_error ** 2)

def generate_complex_dataset(n_samples=1000, noise_level=0.1, random_state=42):
    """
    Generate a complex 4D dataset with non-linear patterns
    
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
    
    return X, y_normalized.reshape(-1, 1)

def train_advanced_network(network, X_train, y_train, X_test, y_test, 
                          epochs=2000, initial_lr=0.01, momentum=0.9, 
                          lr_decay=0.95, decay_every=200):
    """
    Advanced training with momentum and learning rate decay
    """
    print(f"🚀 Starting advanced training for {epochs} epochs")
    print(f"📊 Initial learning rate: {initial_lr}")
    print(f"⚡ Momentum: {momentum}")
    print(f"📉 Learning rate decay: {lr_decay} every {decay_every} epochs")
    
    # Initialize momentum terms
    weight_momentum = [np.zeros_like(w) for w in network.weights]
    bias_momentum = [np.zeros_like(b) for b in network.biases]
    
    current_lr = initial_lr
    
    # Training history
    train_losses = []
    test_losses = []
    learning_rates = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Forward pass on training data
        train_result = network.forward_pass(X_train)
        train_loss = network.calculate_loss(train_result['final_output'], y_train)
        
        # Advanced backpropagation with momentum
        train_loss = advanced_backpropagation_with_momentum(
            network, train_result, y_train, current_lr, 
            weight_momentum, bias_momentum, momentum
        )
        
        # Evaluate on test set (validation)
        test_result = network.forward_pass(X_test)
        test_loss = network.calculate_loss(test_result['final_output'], y_test)
        
        # Store history
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        learning_rates.append(current_lr)
        
        # Learning rate decay
        if (epoch + 1) % decay_every == 0:
            current_lr *= lr_decay
        
        # Print progress
        if epoch % 200 == 0 or epoch == epochs - 1:
            elapsed = time.time() - start_time
            print(f"Epoch {epoch:4d}: Train Loss = {train_loss:.6f}, "
                  f"Test Loss = {test_loss:.6f}, LR = {current_lr:.6f}, "
                  f"Time = {elapsed:.1f}s")
    
    total_time = time.time() - start_time
    print(f"\n✅ Training completed in {total_time:.1f} seconds")
    print(f"📈 Final train loss: {train_losses[-1]:.6f}")
    print(f"📊 Final test loss: {test_losses[-1]:.6f}")
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'learning_rates': learning_rates
    }

def plot_training_results(history, network, X_train, y_train, X_test, y_test):
    """Create comprehensive training visualizations"""
    
    plt.figure(figsize=(20, 12))
    
    # Plot 1: Loss curves
    plt.subplot(2, 4, 1)
    plt.plot(history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    plt.plot(history['test_losses'], 'r-', label='Test Loss', linewidth=2)
    plt.title('Training Progress', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Learning rate decay
    plt.subplot(2, 4, 2)
    plt.plot(history['learning_rates'], 'g-', linewidth=2)
    plt.title('Learning Rate Decay', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 3: Loss improvement over time
    plt.subplot(2, 4, 3)
    improvement = [(history['train_losses'][0] - loss) / history['train_losses'][0] * 100 
                   for loss in history['train_losses']]
    plt.plot(improvement, 'purple', linewidth=2)
    plt.title('Training Improvement', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Improvement (%)')
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Overfitting analysis
    plt.subplot(2, 4, 4)
    gap = np.array(history['test_losses']) - np.array(history['train_losses'])
    plt.plot(gap, 'orange', linewidth=2)
    plt.title('Overfitting Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Test Loss - Train Loss')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Plot 5: Predictions vs Actual (Training)
    plt.subplot(2, 4, 5)
    train_pred = network.forward_pass(X_train)['final_output']
    plt.scatter(y_train, train_pred, alpha=0.5, s=10)
    plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--', linewidth=2)
    plt.title('Training: Predicted vs Actual', fontsize=14, fontweight='bold')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True, alpha=0.3)
    
    # Plot 6: Predictions vs Actual (Test)
    plt.subplot(2, 4, 6)
    test_pred = network.forward_pass(X_test)['final_output']
    plt.scatter(y_test, test_pred, alpha=0.5, s=10, color='red')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
    plt.title('Test: Predicted vs Actual', fontsize=14, fontweight='bold')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.grid(True, alpha=0.3)
    
    # Plot 7: Residuals (Training)
    plt.subplot(2, 4, 7)
    train_residuals = y_train - train_pred
    plt.scatter(train_pred, train_residuals, alpha=0.5, s=10)
    plt.axhline(y=0, color='red', linestyle='--', linewidth=2)
    plt.title('Training Residuals', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.grid(True, alpha=0.3)
    
    # Plot 8: Error distribution
    plt.subplot(2, 4, 8)
    plt.hist(train_residuals, bins=30, alpha=0.7, color='blue', label='Train')
    test_residuals = y_test - test_pred
    plt.hist(test_residuals, bins=30, alpha=0.7, color='red', label='Test')
    plt.title('Error Distribution', fontsize=14, fontweight='bold')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    images_dir = Path("Neural Networks/images")
    images_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(images_dir / "advanced_neural_network_results.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print performance metrics
    train_mse = np.mean(train_residuals ** 2)
    test_mse = np.mean(test_residuals ** 2)
    train_r2 = 1 - train_mse / np.var(y_train)
    test_r2 = 1 - test_mse / np.var(y_test)
    
    print("\n📊 PERFORMANCE METRICS")
    print("=" * 50)
    print(f"Training MSE: {train_mse:.6f}")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Overfitting gap: {test_mse - train_mse:.6f}")
    
    if test_r2 > 0.8:
        print("\n🎉 Excellent performance! The network learned the complex patterns well.")
    elif test_r2 > 0.6:
        print("\n👍 Good performance! The network captured most of the patterns.")
    else:
        print("\n⚠️ The network might need more training or a different architecture.")

def main():
    """Main function to run the advanced neural network demo"""
    
    print("🚀 Advanced Neural Network with NumPy")
    print("=" * 50)
    print("Building on your 2→2→1 foundation to create a sophisticated 4→6→4→1 network!")
    
    # Generate complex dataset
    print("\n📊 Generating complex dataset...")
    X_train, y_train = generate_complex_dataset(n_samples=800)
    X_test, y_test = generate_complex_dataset(n_samples=200)
    
    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Output range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    
    # Create advanced network
    print("\n🏗️ Creating advanced neural network...")
    network = AdvancedNeuralNetwork(
        layer_sizes=[4, 6, 4, 1],
        activations=['relu', 'relu', 'sigmoid']
    )
    
    # Test forward pass
    test_input = np.array([0.5, 0.8, 0.3, 0.9])
    result = network.forward_pass(test_input)
    print(f"\n🧪 Test forward pass:")
    print(f"Input: {test_input}")
    print(f"Output: {result['final_output'][0][0]:.4f}")
    
    # Train the network
    print("\n🎯 Training the advanced network...")
    training_history = train_advanced_network(
        network=network,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        epochs=2000,
        initial_lr=0.01,
        momentum=0.9,
        lr_decay=0.95,
        decay_every=200
    )
    
    # Visualize results
    print("\n📈 Creating visualizations...")
    plot_training_results(training_history, network, X_train, y_train, X_test, y_test)
    
    # Test on edge cases
    print("\n🧪 Testing on edge cases...")
    edge_cases = {
        'All zeros': np.array([0, 0, 0, 0]),
        'All ones': np.array([1, 1, 1, 1]),
        'Mixed': np.array([1, -1, 0.5, -0.5]),
        'Large values': np.array([5, -5, 2, -2])
    }
    
    for name, inputs in edge_cases.items():
        result = network.forward_pass(inputs)
        output = result['final_output'][0][0]
        print(f"{name:15s}: {inputs} → {output:.6f}")
    
    print("\n🎉 Advanced neural network demo completed!")
    print("You've successfully built and trained a sophisticated 4→6→4→1 network!")
    print("This is a huge step up from your original 2→2→1 implementation.")

if __name__ == "__main__":
    main()