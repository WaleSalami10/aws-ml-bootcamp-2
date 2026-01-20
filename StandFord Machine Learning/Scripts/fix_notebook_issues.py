#!/usr/bin/env python3
"""
Script to demonstrate the corrected neural network architecture and fix permission issues.
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

def init_parameters(input_size, hidden_size, output_size):
    """Initialize network parameters with Xavier initialization."""
    weights_input_hidden = np.random.randn(input_size, hidden_size) * np.sqrt(1.0 / input_size)
    weights_hidden_output = np.random.randn(hidden_size, output_size) * np.sqrt(1.0 / hidden_size)
    return weights_input_hidden, weights_hidden_output

def create_corrected_architecture_diagram():
    """Create the corrected architecture diagram with 3 hidden neurons."""
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    
    # Corrected architecture: 2→3→1 (matching the code)
    ax.set_title('Corrected Network Architecture (2→3→1)', fontsize=16, fontweight='bold')
    
    # Draw network structure with CORRECT number of neurons
    layers = [2, 3, 1]  # This matches hidden_size = 3 in the code
    layer_names = ['Input', 'Hidden', 'Output']
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    positions = {}
    max_neurons = max(layers)
    
    for layer_idx, num_neurons in enumerate(layers):
        x = layer_idx * 3
        start_y = (max_neurons - num_neurons) / 2
        
        for neuron_idx in range(num_neurons):
            y = start_y + neuron_idx
            positions[(layer_idx, neuron_idx)] = (x, y)
            
            # Draw neuron
            circle = plt.Circle((x, y), 0.3, color=colors[layer_idx], ec='black')
            ax.add_patch(circle)
            
            # Add sigmoid symbol for hidden and output layers
            if layer_idx > 0:
                ax.text(x, y, 'σ', ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw connections
    for layer_idx in range(len(layers) - 1):
        for i in range(layers[layer_idx]):
            for j in range(layers[layer_idx + 1]):
                x1, y1 = positions[(layer_idx, i)]
                x2, y2 = positions[(layer_idx + 1, j)]
                ax.plot([x1 + 0.3, x2 - 0.3], [y1, y2], 'k-', alpha=0.3, linewidth=1)
    
    # Add layer labels
    for layer_idx, name in enumerate(layer_names):
        x = layer_idx * 3
        ax.text(x, max_neurons + 0.8, name, ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Add parameter count
    total_params = (2 * 3) + (3 * 1)  # weights only
    ax.text(3, -1.5, f'Total Parameters: {total_params} weights', ha='center', fontsize=12)
    ax.text(3, -2, 'Architecture: 2 inputs → 3 hidden (sigmoid) → 1 output (sigmoid)', ha='center', fontsize=10)
    
    ax.set_xlim(-1, 7)
    ax.set_ylim(-2.5, max_neurons + 1.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    
    # Save with error handling for permission issues
    try:
        os.makedirs('Neural Networks/images', exist_ok=True)
        plt.savefig('Neural Networks/images/corrected_architecture_diagram.png', dpi=300, bbox_inches='tight')
        print("✅ Corrected architecture diagram saved to Neural Networks/images/corrected_architecture_diagram.png")
    except PermissionError:
        # Fallback: save in current directory
        os.makedirs('images', exist_ok=True)
        plt.savefig('images/corrected_architecture_diagram.png', dpi=300, bbox_inches='tight')
        print("✅ Corrected architecture diagram saved to images/corrected_architecture_diagram.png")
    
    plt.show()

def demonstrate_corrected_network():
    """Demonstrate the corrected network with proper architecture."""
    
    print("🔧 Demonstrating Corrected Neural Network")
    print("=" * 50)
    
    # Network parameters (matching your code)
    input_size = 2
    hidden_size = 3  # This is what your code uses
    output_size = 1
    learning_rate = 0.1
    epochs = 1000
    
    print(f"Network Architecture: {input_size} → {hidden_size} → {output_size}")
    print(f"Total Parameters: {(input_size * hidden_size) + (hidden_size * output_size)} weights")
    
    # Initialize weights
    weights_input_hidden, weights_hidden_output = init_parameters(input_size, hidden_size, output_size)
    
    # Training data
    inputs = np.array([[0, 1]])
    target = np.array([[1]])
    
    # Training loop
    losses = []
    for epoch in range(epochs):
        # Forward pass
        hidden_input = np.dot(inputs, weights_input_hidden)
        hidden_output = sigmoid(hidden_input)
        output_input = np.dot(hidden_output, weights_hidden_output)
        output = sigmoid(output_input)
        
        # Compute loss
        loss = np.mean((target - output) ** 2)
        losses.append(loss)
        
        # Backward pass
        output_error = target - output
        output_delta = output_error * sigmoid_derivative(output)
        hidden_error = np.dot(output_delta, weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(hidden_output)
        
        # Update weights
        weights_hidden_output += learning_rate * np.dot(hidden_output.T, output_delta)
        weights_input_hidden += learning_rate * np.dot(inputs.T, hidden_delta)
        
        if epoch % 200 == 0:
            print(f"Epoch {epoch:4d}, Loss: {loss:.6f}, Output: {output[0][0]:.4f}")
    
    print(f"\n✅ Final Results:")
    print(f"   Target: {target[0][0]}")
    print(f"   Output: {output[0][0]:.6f}")
    print(f"   Final Loss: {losses[-1]:.6f}")
    
    return losses

def fix_permission_error_example():
    """Show how to fix the permission error."""
    
    print("\n🛠️  Permission Error Fix Examples")
    print("=" * 40)
    
    print("❌ Original problematic code:")
    print("   os.makedirs('../images', exist_ok=True)")
    print("   plt.savefig('../images/plot.png')")
    
    print("\n✅ Fixed code with error handling:")
    print("""
try:
    os.makedirs('../images', exist_ok=True)
    plt.savefig('../images/plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved to ../images/plot.png")
except PermissionError:
    # Fallback: save in current directory
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved to images/plot.png (current directory)")
""")

if __name__ == "__main__":
    print("🔧 Neural Network Corrections and Fixes")
    print("=" * 50)
    
    # Create corrected architecture diagram
    create_corrected_architecture_diagram()
    
    # Demonstrate corrected network
    demonstrate_corrected_network()
    
    # Show permission fix
    fix_permission_error_example()
    
    print("\n✅ All corrections completed!")
    print("   - Architecture diagram now shows 2→3→1 (matching your code)")
    print("   - XOR initialization uses 3 hidden neurons")
    print("   - Permission error handling added")
    print("   - All inconsistencies resolved")