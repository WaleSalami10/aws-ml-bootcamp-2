#!/usr/bin/env python3
"""
Sigmoid Model Visualization Script
Creates detailed analysis of the sigmoid neural network model.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def create_sigmoid_model_visualization():
    """Create comprehensive visualizations for the sigmoid neural network model."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Model Architecture Diagram
    ax1.set_title('Sigmoid Model Architecture\n(2 inputs → 3 hidden → 1 output)', fontsize=14, fontweight='bold')
    
    # Network structure from the sigmoid model
    layers = [2, 3, 1]  # input_size=2, hidden_size=3, output_size=1
    layer_names = ['Input\n(2 neurons)', 'Hidden\n(3 neurons)', 'Output\n(1 neuron)']
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    positions = {}
    max_neurons = max(layers)
    
    for layer_idx, num_neurons in enumerate(layers):
        x = layer_idx * 4
        start_y = (max_neurons - num_neurons) / 2
        
        for neuron_idx in range(num_neurons):
            y = start_y + neuron_idx
            positions[(layer_idx, neuron_idx)] = (x, y)
            
            # Draw neuron
            circle = plt.Circle((x, y), 0.3, color=colors[layer_idx], ec='black', linewidth=2)
            ax1.add_patch(circle)
            
            # Add sigmoid symbol inside neurons
            if layer_idx > 0:  # Hidden and output layers use sigmoid
                ax1.text(x, y, 'σ', ha='center', va='center', fontsize=10, fontweight='bold')
            
            # Labels
            if layer_idx == 0:
                ax1.text(x-0.8, y, f'x{neuron_idx+1}', ha='center', va='center', fontsize=11, fontweight='bold')
            elif layer_idx == len(layers) - 1:
                ax1.text(x+0.8, y, 'ŷ', ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Draw connections with weight indicators
    for layer_idx in range(len(layers) - 1):
        for i in range(layers[layer_idx]):
            for j in range(layers[layer_idx + 1]):
                x1, y1 = positions[(layer_idx, i)]
                x2, y2 = positions[(layer_idx + 1, j)]
                ax1.plot([x1 + 0.3, x2 - 0.3], [y1, y2], 'k-', alpha=0.6, linewidth=2)
    
    # Layer labels
    for layer_idx, name in enumerate(layer_names):
        x = layer_idx * 4
        ax1.text(x, max_neurons + 0.8, name, ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Add weight matrix labels
    ax1.text(2, -0.8, 'W₁ (2×3)', ha='center', va='center', fontsize=10, 
            bbox=dict(boxstyle="round", facecolor='yellow', alpha=0.7))
    ax1.text(6, -0.8, 'W₂ (3×1)', ha='center', va='center', fontsize=10,
            bbox=dict(boxstyle="round", facecolor='yellow', alpha=0.7))
    
    ax1.set_xlim(-1.5, 9.5)
    ax1.set_ylim(-1.5, max_neurons + 1.5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # 2. Sigmoid Function Properties
    ax2.set_title('Sigmoid Function: σ(x) = 1/(1 + e⁻ˣ)', fontsize=14, fontweight='bold')
    
    x = np.linspace(-10, 10, 1000)
    sigmoid_vals = 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    sigmoid_deriv = sigmoid_vals * (1 - sigmoid_vals)
    
    # Plot sigmoid function
    ax2.plot(x, sigmoid_vals, 'b-', linewidth=3, label='Sigmoid σ(x)')
    ax2.plot(x, sigmoid_deriv, 'r--', linewidth=2, label="Derivative σ'(x)")
    
    # Add key points
    ax2.plot(0, 0.5, 'go', markersize=8, label='σ(0) = 0.5')
    ax2.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    
    # Annotations
    ax2.annotate('Output range: (0, 1)', xy=(5, 0.9), xytext=(6, 0.8),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, color='blue')
    ax2.annotate('Max derivative: 0.25', xy=(0, 0.25), xytext=(3, 0.15),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlabel('Input (x)')
    ax2.set_ylabel('Output')
    ax2.set_xlim(-8, 8)
    ax2.set_ylim(-0.1, 1.1)
    
    # 3. Training Process Visualization
    ax3.set_title('Training Process: Forward & Backward Pass', fontsize=14, fontweight='bold')
    
    # Create flow diagram
    steps = [
        ('Input\n[0, 1]', 'lightblue'),
        ('Hidden\nσ(W₁·x)', 'lightgreen'),
        ('Output\nσ(W₂·h)', 'lightcoral'),
        ('Error\n(target - output)', 'yellow'),
        ('Backprop\nUpdate weights', 'orange')
    ]
    
    # Forward pass (top row)
    for i, (step, color) in enumerate(steps[:3]):
        x_pos = i * 2.5 + 1
        y_pos = 2
        
        rect = plt.Rectangle((x_pos - 0.6, y_pos - 0.3), 1.2, 0.6, 
                           facecolor=color, edgecolor='black', linewidth=1)
        ax3.add_patch(rect)
        ax3.text(x_pos, y_pos, step, ha='center', va='center', fontsize=9, fontweight='bold')
        
        if i < 2:
            ax3.arrow(x_pos + 0.6, y_pos, 1.3, 0, head_width=0.1, head_length=0.1, fc='blue', ec='blue')
    
    # Backward pass (bottom row)
    for i, (step, color) in enumerate(steps[3:]):
        x_pos = (2 - i) * 2.5 + 1
        y_pos = 0.5
        
        rect = plt.Rectangle((x_pos - 0.6, y_pos - 0.3), 1.2, 0.6, 
                           facecolor=color, edgecolor='black', linewidth=1)
        ax3.add_patch(rect)
        ax3.text(x_pos, y_pos, step, ha='center', va='center', fontsize=9, fontweight='bold')
        
        if i < 1:
            ax3.arrow(x_pos - 0.6, y_pos, -1.3, 0, head_width=0.1, head_length=0.1, fc='red', ec='red')
    
    # Connect forward to backward
    ax3.arrow(5.5, 1.7, 0, -0.9, head_width=0.1, head_length=0.1, fc='purple', ec='purple')
    
    # Labels
    ax3.text(3.5, 2.5, 'Forward Pass', ha='center', va='center', fontsize=11, 
            fontweight='bold', color='blue')
    ax3.text(3.5, 0, 'Backward Pass', ha='center', va='center', fontsize=11, 
            fontweight='bold', color='red')
    
    ax3.set_xlim(0, 7)
    ax3.set_ylim(-0.5, 3)
    ax3.axis('off')
    
    # 4. Model Performance Analysis
    ax4.set_title('Model Training: Input [0,1] → Target [1]', fontsize=14, fontweight='bold')
    
    # Simulate the training process from the original code
    np.random.seed(42)  # For reproducible results
    
    # Initialize parameters (matching the original model)
    input_size, hidden_size, output_size = 2, 3, 1
    weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
    weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))
    
    inputs = np.array([[0, 1]])
    target = np.array([[1]])
    learning_rate = 0.1
    
    # Track training progress
    epochs = []
    outputs = []
    errors = []
    
    for epoch in range(0, 1000, 50):  # Sample every 50 epochs
        # Forward pass
        hidden = 1 / (1 + np.exp(-np.dot(inputs, weights_input_hidden)))
        output = 1 / (1 + np.exp(-np.dot(hidden, weights_hidden_output)))
        
        # Calculate error
        error = np.mean((target - output) ** 2)
        
        epochs.append(epoch)
        outputs.append(output[0, 0])
        errors.append(error)
        
        # Backward pass (simplified for visualization)
        output_error = target - output
        output_delta = output_error * output * (1 - output)
        hidden_error = np.dot(output_delta, weights_hidden_output.T)
        hidden_delta = hidden_error * hidden * (1 - hidden)
        
        # Update weights
        weights_hidden_output += learning_rate * np.dot(hidden.T, output_delta)
        weights_input_hidden += learning_rate * np.dot(inputs.T, hidden_delta)
    
    # Plot training progress
    ax4_twin = ax4.twinx()
    
    line1 = ax4.plot(epochs, outputs, 'b-', linewidth=2, marker='o', markersize=4, label='Model Output')
    line2 = ax4_twin.plot(epochs, errors, 'r--', linewidth=2, marker='s', markersize=4, label='MSE Error')
    
    # Target line
    ax4.axhline(y=1, color='green', linestyle=':', linewidth=2, label='Target = 1')
    
    ax4.set_xlabel('Training Epoch')
    ax4.set_ylabel('Model Output', color='blue')
    ax4_twin.set_ylabel('Mean Squared Error', color='red')
    
    ax4.tick_params(axis='y', labelcolor='blue')
    ax4_twin.tick_params(axis='y', labelcolor='red')
    
    # Combine legends
    lines1, labels1 = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_twin.get_legend_handles_labels()
    ax4.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.1)
    
    # Add final result annotation
    final_output = outputs[-1]
    ax4.annotate(f'Final Output: {final_output:.3f}', 
                xy=(epochs[-1], final_output), xytext=(epochs[-5], 0.8),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, color='blue', fontweight='bold')
    
    plt.tight_layout()
    
    # Save the visualization
    output_dir = 'Neural Networks/images'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/sigmoid_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Sigmoid model analysis saved to {output_dir}/sigmoid_model_analysis.png")
    
    return final_output

if __name__ == "__main__":
    print("Creating sigmoid model visualization...")
    final_result = create_sigmoid_model_visualization()
    print(f"Training completed! Final output: {final_result:.6f}")