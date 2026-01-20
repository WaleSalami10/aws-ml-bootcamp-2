#!/usr/bin/env python3
"""
Neural Network Visualization Script
Creates detailed diagrams of neural network architecture and components.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Circle, Arrow
import os

# Create images directory if it doesn't exist
os.makedirs('images', exist_ok=True)

def create_neural_network_diagram():
    """Create a comprehensive neural network architecture diagram."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Basic Neural Network Architecture
    ax1.set_title('Multi-Layer Perceptron Architecture', fontsize=14, fontweight='bold')
    
    # Network structure: 3 inputs, 4 hidden, 2 outputs
    layers = [3, 4, 2]
    layer_names = ['Input\nLayer', 'Hidden\nLayer', 'Output\nLayer']
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    
    # Position neurons
    max_neurons = max(layers)
    positions = {}
    
    for layer_idx, num_neurons in enumerate(layers):
        x = layer_idx * 4
        start_y = (max_neurons - num_neurons) / 2
        
        for neuron_idx in range(num_neurons):
            y = start_y + neuron_idx
            positions[(layer_idx, neuron_idx)] = (x, y)
            
            # Draw neuron
            circle = Circle((x, y), 0.4, color=colors[layer_idx], ec='black', linewidth=2)
            ax1.add_patch(circle)
            
            # Add neuron labels
            if layer_idx == 0:
                ax1.text(x-1.2, y, f'x₁' if neuron_idx == 0 else f'x₂' if neuron_idx == 1 else f'x₃', 
                        ha='center', va='center', fontsize=12, fontweight='bold')
            elif layer_idx == len(layers) - 1:
                ax1.text(x+1.2, y, f'ŷ₁' if neuron_idx == 0 else f'ŷ₂', 
                        ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Draw connections with weights
    weight_examples = [0.8, -0.3, 0.5, 0.2]
    connection_count = 0
    
    for layer_idx in range(len(layers) - 1):
        for i in range(layers[layer_idx]):
            for j in range(layers[layer_idx + 1]):
                x1, y1 = positions[(layer_idx, i)]
                x2, y2 = positions[(layer_idx + 1, j)]
                
                # Draw connection
                line_width = abs(weight_examples[connection_count % len(weight_examples)]) * 3
                color = 'red' if weight_examples[connection_count % len(weight_examples)] < 0 else 'blue'
                ax1.plot([x1 + 0.4, x2 - 0.4], [y1, y2], color=color, 
                        linewidth=line_width, alpha=0.7)
                
                # Add weight label on some connections
                if connection_count < 4:
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    weight_val = weight_examples[connection_count]
                    ax1.text(mid_x, mid_y + 0.2, f'w={weight_val}', 
                            ha='center', va='center', fontsize=8, 
                            bbox=dict(boxstyle="round,pad=0.1", facecolor='white', alpha=0.8))
                
                connection_count += 1
    
    # Add layer labels
    for layer_idx, name in enumerate(layer_names):
        x = layer_idx * 4
        ax1.text(x, max_neurons + 1, name, ha='center', va='center', 
                fontsize=12, fontweight='bold')
    
    # Add bias terms
    for layer_idx in range(1, len(layers)):
        x = layer_idx * 4
        y = max_neurons + 0.5
        ax1.text(x, y - 0.3, '+b', ha='center', va='center', 
                fontsize=10, style='italic')
    
    ax1.set_xlim(-2, 10)
    ax1.set_ylim(-0.5, max_neurons + 1.5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Add legend for connections
    ax1.text(5, -0.3, 'Line thickness ∝ |weight|', fontsize=10, style='italic')
    ax1.text(5, -0.6, 'Blue: positive weights, Red: negative weights', fontsize=10, style='italic')
    
    # 2. Single Neuron Detail
    ax2.set_title('Single Neuron (Perceptron) Detail', fontsize=14, fontweight='bold')
    
    # Draw inputs
    inputs = ['x₁', 'x₂', 'x₃']
    weights = ['w₁', 'w₂', 'w₃']
    
    for i, (inp, weight) in enumerate(zip(inputs, weights)):
        y_pos = 2 - i * 0.8
        
        # Input
        ax2.text(0, y_pos, inp, ha='center', va='center', fontsize=12, 
                bbox=dict(boxstyle="circle", facecolor='lightblue'))
        
        # Weight
        ax2.text(1.5, y_pos, weight, ha='center', va='center', fontsize=10,
                bbox=dict(boxstyle="round", facecolor='yellow'))
        
        # Arrow to summation
        ax2.arrow(0.3, y_pos, 0.9, 0, head_width=0.1, head_length=0.1, 
                 fc='black', ec='black')
    
    # Summation symbol
    ax2.text(3, 1, '∑', ha='center', va='center', fontsize=24, fontweight='bold',
            bbox=dict(boxstyle="circle", facecolor='lightgreen'))
    
    # Bias
    ax2.text(3, 0.2, '+b', ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="round", facecolor='orange'))
    
    # Activation function
    ax2.text(4.5, 1, 'σ(z)', ha='center', va='center', fontsize=14, fontweight='bold',
            bbox=dict(boxstyle="round", facecolor='lightcoral'))
    
    # Output
    ax2.text(6, 1, 'a', ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle="circle", facecolor='lightgray'))
    
    # Arrows
    ax2.arrow(3.3, 1, 0.9, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax2.arrow(4.8, 1, 0.9, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    
    # Mathematical formula
    ax2.text(3, -0.8, 'z = w₁x₁ + w₂x₂ + w₃x₃ + b', ha='center', va='center', 
            fontsize=12, bbox=dict(boxstyle="round", facecolor='lightyellow'))
    ax2.text(3, -1.3, 'a = σ(z)', ha='center', va='center', 
            fontsize=12, bbox=dict(boxstyle="round", facecolor='lightyellow'))
    
    ax2.set_xlim(-0.5, 6.5)
    ax2.set_ylim(-2, 3)
    ax2.axis('off')
    
    # 3. Activation Functions
    ax3.set_title('Common Activation Functions', fontsize=14, fontweight='bold')
    
    z = np.linspace(-5, 5, 1000)
    
    # Sigmoid
    sigmoid = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    ax3.plot(z, sigmoid, 'b-', linewidth=2, label='Sigmoid: σ(z) = 1/(1+e⁻ᶻ)')
    
    # ReLU
    relu = np.maximum(0, z)
    ax3.plot(z, relu, 'r-', linewidth=2, label='ReLU: max(0, z)')
    
    # Tanh
    tanh = np.tanh(z)
    ax3.plot(z, tanh, 'g-', linewidth=2, label='Tanh: (eᶻ - e⁻ᶻ)/(eᶻ + e⁻ᶻ)')
    
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=10)
    ax3.set_xlabel('Input (z)', fontsize=12)
    ax3.set_ylabel('Output', fontsize=12)
    ax3.set_xlim(-5, 5)
    ax3.set_ylim(-1.5, 5)
    
    # 4. Forward Propagation Flow
    ax4.set_title('Forward Propagation Process', fontsize=14, fontweight='bold')
    
    # Create flow diagram
    steps = [
        'Input\nData\n(X)',
        'Layer 1\nz⁽¹⁾ = W⁽¹⁾X + b⁽¹⁾\na⁽¹⁾ = σ(z⁽¹⁾)',
        'Layer 2\nz⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾\na⁽²⁾ = σ(z⁽²⁾)',
        'Output\nPrediction\n(ŷ)'
    ]
    
    colors_flow = ['lightblue', 'lightgreen', 'lightgreen', 'lightcoral']
    
    for i, (step, color) in enumerate(zip(steps, colors_flow)):
        x_pos = i * 2.5
        
        # Draw box
        box = FancyBboxPatch((x_pos - 0.8, 0.5), 1.6, 1.5, 
                            boxstyle="round,pad=0.1", 
                            facecolor=color, edgecolor='black', linewidth=2)
        ax4.add_patch(box)
        
        # Add text
        ax4.text(x_pos, 1.25, step, ha='center', va='center', 
                fontsize=10, fontweight='bold')
        
        # Add arrow (except for last step)
        if i < len(steps) - 1:
            ax4.arrow(x_pos + 0.8, 1.25, 0.9, 0, head_width=0.15, 
                     head_length=0.2, fc='black', ec='black')
    
    # Add mathematical notation
    ax4.text(3.75, -0.5, 'Matrix Operations at Each Layer:', 
            ha='center', va='center', fontsize=12, fontweight='bold')
    ax4.text(3.75, -1, 'Linear Transformation → Activation Function', 
            ha='center', va='center', fontsize=11, style='italic')
    
    ax4.set_xlim(-1, 8.5)
    ax4.set_ylim(-1.5, 2.5)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('images/neural_network_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_backpropagation_diagram():
    """Create a diagram showing backpropagation process."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Backpropagation Flow
    ax1.set_title('Backpropagation: Error Flow', fontsize=14, fontweight='bold')
    
    # Network layers (simplified)
    layers = ['Input', 'Hidden', 'Output']
    x_positions = [1, 3, 5]
    
    # Draw layers
    for i, (layer, x_pos) in enumerate(zip(layers, x_positions)):
        if i == 0:
            color = 'lightblue'
        elif i == len(layers) - 1:
            color = 'lightcoral'
        else:
            color = 'lightgreen'
            
        circle = Circle((x_pos, 2), 0.5, color=color, ec='black', linewidth=2)
        ax1.add_patch(circle)
        ax1.text(x_pos, 2, layer, ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Forward arrows (blue)
    for i in range(len(x_positions) - 1):
        ax1.arrow(x_positions[i] + 0.5, 2.2, 0.8, 0, head_width=0.1, 
                 head_length=0.1, fc='blue', ec='blue', linewidth=2)
        ax1.text(x_positions[i] + 0.9, 2.5, 'Forward', ha='center', va='center', 
                fontsize=9, color='blue', fontweight='bold')
    
    # Backward arrows (red)
    for i in range(len(x_positions) - 1, 0, -1):
        ax1.arrow(x_positions[i] - 0.5, 1.8, -0.8, 0, head_width=0.1, 
                 head_length=0.1, fc='red', ec='red', linewidth=2)
        ax1.text(x_positions[i] - 0.9, 1.5, 'Backward', ha='center', va='center', 
                fontsize=9, color='red', fontweight='bold')
    
    # Loss function
    ax1.text(5, 0.8, 'Loss Function\nJ = L(ŷ, y)', ha='center', va='center', 
            fontsize=12, bbox=dict(boxstyle="round", facecolor='yellow'))
    
    # Gradient computation
    ax1.text(3, 0.5, 'Compute Gradients:\n∂J/∂W, ∂J/∂b', ha='center', va='center', 
            fontsize=11, bbox=dict(boxstyle="round", facecolor='orange'))
    
    ax1.set_xlim(0, 6)
    ax1.set_ylim(0, 3.5)
    ax1.axis('off')
    
    # 2. Chain Rule Visualization
    ax2.set_title('Chain Rule in Backpropagation', fontsize=14, fontweight='bold')
    
    # Create chain rule diagram
    chain_elements = [
        ('∂J/∂a⁽ˡ⁾', 'Loss w.r.t.\nActivation'),
        ('∂a⁽ˡ⁾/∂z⁽ˡ⁾', 'Activation\nDerivative'),
        ('∂z⁽ˡ⁾/∂W⁽ˡ⁾', 'Linear\nDerivative')
    ]
    
    y_pos = 2.5
    for i, (formula, description) in enumerate(chain_elements):
        x_pos = i * 2 + 1
        
        # Formula box
        ax2.text(x_pos, y_pos, formula, ha='center', va='center', 
                fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round", facecolor='lightblue'))
        
        # Description
        ax2.text(x_pos, y_pos - 0.8, description, ha='center', va='center', 
                fontsize=10, style='italic')
        
        # Multiplication symbol
        if i < len(chain_elements) - 1:
            ax2.text(x_pos + 1, y_pos, '×', ha='center', va='center', 
                    fontsize=16, fontweight='bold')
    
    # Final result
    ax2.text(3, 1, '∂J/∂W⁽ˡ⁾ = ∂J/∂a⁽ˡ⁾ × ∂a⁽ˡ⁾/∂z⁽ˡ⁾ × ∂z⁽ˡ⁾/∂W⁽ˡ⁾', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round", facecolor='yellow'))
    
    # Update rule
    ax2.text(3, 0.3, 'Weight Update: W⁽ˡ⁾ := W⁽ˡ⁾ - α × ∂J/∂W⁽ˡ⁾', 
            ha='center', va='center', fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round", facecolor='lightgreen'))
    
    ax2.set_xlim(0, 6)
    ax2.set_ylim(0, 3.5)
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('images/backpropagation_process.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

def create_learning_process_diagram():
    """Create a diagram showing the learning process."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Training Loop
    ax1.set_title('Neural Network Training Loop', fontsize=14, fontweight='bold')
    
    # Create circular flow diagram
    steps = [
        ('Initialize\nWeights', (2, 3)),
        ('Forward\nPropagation', (4, 3)),
        ('Compute\nLoss', (4, 1)),
        ('Backward\nPropagation', (2, 1)),
        ('Update\nWeights', (1, 2))
    ]
    
    colors = ['lightblue', 'lightgreen', 'yellow', 'orange', 'lightcoral']
    
    for i, ((step, pos), color) in enumerate(zip(steps, colors)):
        x, y = pos
        circle = Circle((x, y), 0.4, color=color, ec='black', linewidth=2)
        ax1.add_patch(circle)
        ax1.text(x, y, step, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Add arrows between steps
    arrows = [
        ((2.4, 3), (3.6, 3)),      # Initialize → Forward
        ((4, 2.6), (4, 1.4)),      # Forward → Loss
        ((3.6, 1), (2.4, 1)),      # Loss → Backward
        ((1.8, 1.3), (1.2, 1.7)),  # Backward → Update
        ((1.2, 2.3), (1.8, 2.7))   # Update → Initialize (loop)
    ]
    
    for (start, end) in arrows:
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        ax1.arrow(start[0], start[1], dx, dy, head_width=0.08, 
                 head_length=0.08, fc='black', ec='black')
    
    # Add epoch counter
    ax1.text(2.5, 0.2, 'Repeat for Multiple Epochs', ha='center', va='center', 
            fontsize=12, fontweight='bold', style='italic')
    
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 4)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # 2. Cost Function Behavior
    ax2.set_title('Cost Function During Training', fontsize=14, fontweight='bold')
    
    # Simulate cost function decrease
    epochs = np.arange(0, 1000, 10)
    cost = 2 * np.exp(-epochs/200) + 0.1 + 0.05 * np.random.randn(len(epochs))
    
    ax2.plot(epochs, cost, 'b-', linewidth=2, label='Training Cost')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Cost', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Add annotations
    ax2.annotate('Initial high cost', xy=(0, cost[0]), xytext=(200, cost[0] + 0.3),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=10, color='red')
    ax2.annotate('Converged cost', xy=(epochs[-1], cost[-1]), 
                xytext=(epochs[-1] - 200, cost[-1] + 0.3),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')
    
    # 3. Weight Evolution
    ax3.set_title('Weight Evolution During Training', fontsize=14, fontweight='bold')
    
    # Simulate weight changes
    epochs_w = np.arange(0, 500, 5)
    weight1 = 0.1 + 0.8 * (1 - np.exp(-epochs_w/100)) + 0.02 * np.random.randn(len(epochs_w))
    weight2 = -0.2 - 0.6 * (1 - np.exp(-epochs_w/150)) + 0.02 * np.random.randn(len(epochs_w))
    
    ax3.plot(epochs_w, weight1, 'b-', linewidth=2, label='Weight 1')
    ax3.plot(epochs_w, weight2, 'r-', linewidth=2, label='Weight 2')
    ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Weight Value', fontsize=12)
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Learning Rate Effects
    ax4.set_title('Effect of Learning Rate', fontsize=14, fontweight='bold')
    
    epochs_lr = np.arange(0, 200, 2)
    
    # Different learning rates
    cost_low_lr = 1.5 * np.exp(-epochs_lr/100) + 0.1
    cost_good_lr = 1.5 * np.exp(-epochs_lr/30) + 0.1
    cost_high_lr = 1.5 * np.exp(-epochs_lr/10) * (1 + 0.3 * np.sin(epochs_lr/5)) + 0.1
    
    ax4.plot(epochs_lr, cost_low_lr, 'g-', linewidth=2, label='Low LR (α=0.001)')
    ax4.plot(epochs_lr, cost_good_lr, 'b-', linewidth=2, label='Good LR (α=0.01)')
    ax4.plot(epochs_lr, cost_high_lr, 'r-', linewidth=2, label='High LR (α=0.1)')
    
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('Cost', fontsize=12)
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Add annotations
    ax4.text(150, 0.8, 'Too slow', color='green', fontsize=10, fontweight='bold')
    ax4.text(100, 0.3, 'Just right', color='blue', fontsize=10, fontweight='bold')
    ax4.text(50, 1.2, 'Oscillating', color='red', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('images/neural_network_learning_process.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig

if __name__ == "__main__":
    print("Creating neural network visualizations...")
    
    # Create comprehensive neural network diagram
    print("1. Creating comprehensive neural network diagram...")
    create_neural_network_diagram()
    
    # Create backpropagation diagram
    print("2. Creating backpropagation process diagram...")
    create_backpropagation_diagram()
    
    # Create learning process diagram
    print("3. Creating learning process diagram...")
    create_learning_process_diagram()
    
    print("\nAll visualizations saved to 'images/' directory!")
    print("Files created:")
    print("- images/neural_network_comprehensive.png")
    print("- images/backpropagation_process.png") 
    print("- images/neural_network_learning_process.png")