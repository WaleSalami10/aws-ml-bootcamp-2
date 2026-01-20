#!/usr/bin/env python3
"""
Derivative Demonstration for Neural Networks

This script visually demonstrates why derivatives are essential for neural network training.
It shows the relationship between activation functions, their derivatives, and learning efficiency.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_derivative(x):
    """Derivative of sigmoid"""
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    """ReLU activation function"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU"""
    return (x > 0).astype(float)

def tanh_derivative(x):
    """Derivative of tanh"""
    return 1 - np.tanh(x) ** 2

def demonstrate_chain_rule():
    """Demonstrate how derivatives are used in the chain rule"""
    
    print("🔗 CHAIN RULE DEMONSTRATION")
    print("=" * 50)
    
    # Simple network: input → weight → activation → output
    x = 2.0          # Input
    w = 0.5          # Weight
    target = 1.0     # Target output
    learning_rate = 0.1
    
    print(f"Network setup:")
    print(f"  Input (x): {x}")
    print(f"  Weight (w): {w}")
    print(f"  Target: {target}")
    print(f"  Learning rate: {learning_rate}")
    
    # Forward pass
    z = w * x  # Linear combination
    a = sigmoid(z)  # Activation
    error = (a - target) ** 2  # Loss
    
    print(f"\nForward pass:")
    print(f"  z = w × x = {w} × {x} = {z}")
    print(f"  a = sigmoid({z}) = {a:.4f}")
    print(f"  error = (a - target)² = ({a:.4f} - {target})² = {error:.4f}")
    
    # Backward pass (chain rule)
    d_error_d_a = 2 * (a - target)  # ∂error/∂a
    d_a_d_z = sigmoid_derivative(z)  # ∂a/∂z (THIS IS THE KEY!)
    d_z_d_w = x  # ∂z/∂w
    
    # Chain rule: ∂error/∂w = ∂error/∂a × ∂a/∂z × ∂z/∂w
    d_error_d_w = d_error_d_a * d_a_d_z * d_z_d_w
    
    print(f"\nBackward pass (chain rule):")
    print(f"  ∂error/∂a = 2 × (a - target) = 2 × ({a:.4f} - {target}) = {d_error_d_a:.4f}")
    print(f"  ∂a/∂z = sigmoid_derivative({z}) = {d_a_d_z:.4f}  ← ACTIVATION DERIVATIVE!")
    print(f"  ∂z/∂w = x = {d_z_d_w}")
    print(f"  ∂error/∂w = {d_error_d_a:.4f} × {d_a_d_z:.4f} × {d_z_d_w} = {d_error_d_w:.4f}")
    
    # Weight update
    w_new = w - learning_rate * d_error_d_w
    print(f"\nWeight update:")
    print(f"  w_new = w - learning_rate × ∂error/∂w")
    print(f"  w_new = {w} - {learning_rate} × {d_error_d_w:.4f} = {w_new:.4f}")
    
    # Show improvement
    z_new = w_new * x
    a_new = sigmoid(z_new)
    error_new = (a_new - target) ** 2
    
    print(f"\nAfter weight update:")
    print(f"  New output: {a_new:.4f} (closer to target {target})")
    print(f"  New error: {error_new:.4f} (reduced from {error:.4f})")
    print(f"  Improvement: {((error - error_new) / error * 100):.1f}%")
    
    print(f"\n💡 The derivative {d_a_d_z:.4f} determined how much the weight should change!")

def plot_activation_functions_and_derivatives():
    """Plot activation functions and their derivatives"""
    
    x = np.linspace(-6, 6, 1000)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Sigmoid
    axes[0, 0].plot(x, sigmoid(x), 'b-', linewidth=3, label='Sigmoid')
    axes[0, 0].set_title('Sigmoid Function', fontsize=14, fontweight='bold')
    axes[0, 0].set_ylabel('Output')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[0, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    axes[1, 0].plot(x, sigmoid_derivative(x), 'b--', linewidth=3, label='Sigmoid Derivative')
    axes[1, 0].set_title('Sigmoid Derivative', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Input (x)')
    axes[1, 0].set_ylabel('Derivative')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    axes[1, 0].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 0].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Add annotations for sigmoid
    axes[1, 0].annotate('Maximum derivative\nat x=0', xy=(0, 0.25), xytext=(2, 0.2),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=10, color='red')
    axes[1, 0].annotate('Vanishing gradients\nfor large |x|', xy=(4, 0.02), xytext=(3, 0.15),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=10, color='red')
    
    # ReLU
    axes[0, 1].plot(x, relu(x), 'g-', linewidth=3, label='ReLU')
    axes[0, 1].set_title('ReLU Function', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    axes[0, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    axes[1, 1].plot(x, relu_derivative(x), 'g--', linewidth=3, label='ReLU Derivative')
    axes[1, 1].set_title('ReLU Derivative', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Input (x)')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 1].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 1].set_ylim(-0.1, 1.1)
    
    # Add annotations for ReLU
    axes[1, 1].annotate('Constant gradient = 1\nfor x > 0', xy=(3, 1), xytext=(2, 0.7),
                       arrowprops=dict(arrowstyle='->', color='green'),
                       fontsize=10, color='green')
    axes[1, 1].annotate('No gradient\nfor x < 0', xy=(-3, 0), xytext=(-2, 0.3),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=10, color='red')
    
    # Tanh
    axes[0, 2].plot(x, np.tanh(x), 'r-', linewidth=3, label='Tanh')
    axes[0, 2].set_title('Tanh Function', fontsize=14, fontweight='bold')
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend()
    axes[0, 2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[0, 2].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    axes[1, 2].plot(x, tanh_derivative(x), 'r--', linewidth=3, label='Tanh Derivative')
    axes[1, 2].set_title('Tanh Derivative', fontsize=14, fontweight='bold')
    axes[1, 2].set_xlabel('Input (x)')
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend()
    axes[1, 2].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1, 2].axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Add annotations for tanh
    axes[1, 2].annotate('Maximum derivative\nat x=0', xy=(0, 1), xytext=(2, 0.8),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=10, color='red')
    
    plt.tight_layout()
    
    # Save the plot
    images_dir = Path("Neural Networks/images")
    images_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(images_dir / "activation_functions_and_derivatives.png", dpi=300, bbox_inches='tight')
    plt.show()

def demonstrate_vanishing_gradients():
    """Demonstrate the vanishing gradient problem"""
    
    print("\n🚨 VANISHING GRADIENT DEMONSTRATION")
    print("=" * 50)
    
    # Test different input magnitudes
    inputs = [-5, -2, -1, 0, 1, 2, 5]
    
    print(f"{'Input':<8} {'Sigmoid':<10} {'Sigmoid Deriv':<15} {'ReLU':<8} {'ReLU Deriv':<12}")
    print("-" * 60)
    
    for x in inputs:
        sig_val = sigmoid(x)
        sig_deriv = sigmoid_derivative(x)
        relu_val = relu(x)
        relu_deriv = relu_derivative(x)
        
        print(f"{x:<8} {sig_val:<10.4f} {sig_deriv:<15.6f} {relu_val:<8.1f} {relu_deriv:<12.0f}")
    
    print(f"\n💡 Key Observations:")
    print(f"   • Sigmoid derivative becomes tiny for large |x| (vanishing gradients)")
    print(f"   • ReLU derivative is always 1 for positive x (no vanishing)")
    print(f"   • This is why ReLU revolutionized deep learning!")

def simulate_learning_with_different_activations():
    """Simulate learning speed with different activation functions"""
    
    print("\n📈 LEARNING SPEED COMPARISON")
    print("=" * 50)
    
    # Simple learning scenario
    x = 3.0  # Large input (where sigmoid struggles)
    target = 1.0
    learning_rate = 0.1
    initial_weight = 0.1
    
    # Sigmoid learning
    w_sigmoid = initial_weight
    sigmoid_losses = []
    
    # ReLU learning  
    w_relu = initial_weight
    relu_losses = []
    
    print(f"Learning scenario:")
    print(f"  Input: {x}")
    print(f"  Target: {target}")
    print(f"  Initial weight: {initial_weight}")
    print(f"  Learning rate: {learning_rate}")
    
    print(f"\n{'Step':<6} {'Sigmoid Loss':<15} {'ReLU Loss':<12} {'Sigmoid Weight':<16} {'ReLU Weight':<12}")
    print("-" * 70)
    
    for step in range(10):
        # Sigmoid network
        z_sig = w_sigmoid * x
        a_sig = sigmoid(z_sig)
        loss_sig = (a_sig - target) ** 2
        grad_sig = 2 * (a_sig - target) * sigmoid_derivative(z_sig) * x
        w_sigmoid -= learning_rate * grad_sig
        sigmoid_losses.append(loss_sig)
        
        # ReLU network
        z_relu = w_relu * x
        a_relu = relu(z_relu)
        loss_relu = (a_relu - target) ** 2 if a_relu != target else 0
        if z_relu > 0:
            grad_relu = 2 * (a_relu - target) * 1 * x  # ReLU derivative = 1
        else:
            grad_relu = 0
        w_relu -= learning_rate * grad_relu
        relu_losses.append(loss_relu)
        
        if step % 2 == 0:  # Print every other step
            print(f"{step:<6} {loss_sig:<15.6f} {loss_relu:<12.6f} {w_sigmoid:<16.4f} {w_relu:<12.4f}")
    
    print(f"\nFinal Results:")
    print(f"  Sigmoid final loss: {sigmoid_losses[-1]:.6f}")
    print(f"  ReLU final loss: {relu_losses[-1]:.6f}")
    print(f"  ReLU learned {relu_losses[-1]/sigmoid_losses[-1]:.1f}x better!")

def plot_gradient_flow_in_deep_network():
    """Visualize how gradients flow through a deep network"""
    
    print("\n🏗️ DEEP NETWORK GRADIENT FLOW")
    print("=" * 50)
    
    # Simulate a 5-layer network
    layers = 5
    
    # Different scenarios
    sigmoid_gradients = []
    relu_gradients = []
    mixed_gradients = []
    
    # Starting gradient (from output layer)
    initial_grad = 1.0
    
    # Sigmoid network (all sigmoid activations)
    grad = initial_grad
    sigmoid_gradients.append(grad)
    for layer in range(layers - 1):
        # Assume activation input around 2 (where sigmoid derivative is small)
        grad *= sigmoid_derivative(2.0)
        sigmoid_gradients.append(grad)
    
    # ReLU network (all ReLU activations)
    grad = initial_grad
    relu_gradients.append(grad)
    for layer in range(layers - 1):
        # ReLU derivative is 1 for positive inputs
        grad *= 1.0
        relu_gradients.append(grad)
    
    # Mixed network (ReLU hidden + sigmoid output)
    grad = initial_grad
    mixed_gradients.append(grad)
    for layer in range(layers - 1):
        if layer == layers - 2:  # Last layer (output)
            grad *= sigmoid_derivative(1.0)  # Reasonable sigmoid derivative
        else:  # Hidden layers
            grad *= 1.0  # ReLU derivative
        mixed_gradients.append(grad)
    
    # Plot the results
    plt.figure(figsize=(12, 8))
    
    layer_names = ['Output', 'Layer 4', 'Layer 3', 'Layer 2', 'Input']
    x_pos = range(len(layer_names))
    
    plt.semilogy(x_pos, sigmoid_gradients, 'b-o', linewidth=3, markersize=8, label='All Sigmoid')
    plt.semilogy(x_pos, relu_gradients, 'g-s', linewidth=3, markersize=8, label='All ReLU')
    plt.semilogy(x_pos, mixed_gradients, 'r-^', linewidth=3, markersize=8, label='ReLU + Sigmoid')
    
    plt.xlabel('Network Layer (Output → Input)', fontsize=12)
    plt.ylabel('Gradient Magnitude (log scale)', fontsize=12)
    plt.title('Gradient Flow Through Deep Network', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, layer_names)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add annotations
    plt.annotate('Vanishing gradients!\nLearning becomes impossible', 
                xy=(4, sigmoid_gradients[-1]), xytext=(3, sigmoid_gradients[-1] * 100),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=10, color='blue')
    
    plt.annotate('Strong gradients\nEffective learning', 
                xy=(4, relu_gradients[-1]), xytext=(2.5, relu_gradients[-1] * 0.1),
                arrowprops=dict(arrowstyle='->', color='green'),
                fontsize=10, color='green')
    
    plt.tight_layout()
    
    # Save the plot
    images_dir = Path("Neural Networks/images")
    images_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(images_dir / "gradient_flow_deep_network.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print numerical results
    print(f"Gradient at input layer:")
    print(f"  All Sigmoid: {sigmoid_gradients[-1]:.8f} (vanished!)")
    print(f"  All ReLU: {relu_gradients[-1]:.1f} (strong)")
    print(f"  Mixed: {mixed_gradients[-1]:.4f} (reasonable)")
    
    print(f"\nGradient ratio (ReLU vs Sigmoid): {relu_gradients[-1]/sigmoid_gradients[-1]:.0f}x stronger!")

def main():
    """Main demonstration function"""
    
    print("🎓 WHY DERIVATIVES MATTER IN NEURAL NETWORKS")
    print("=" * 60)
    print("This demonstration shows why activation function derivatives are essential for learning.")
    
    # 1. Chain rule demonstration
    demonstrate_chain_rule()
    
    # 2. Plot activation functions and derivatives
    print(f"\n📊 Creating activation function plots...")
    plot_activation_functions_and_derivatives()
    
    # 3. Vanishing gradient demonstration
    demonstrate_vanishing_gradients()
    
    # 4. Learning speed comparison
    simulate_learning_with_different_activations()
    
    # 5. Deep network gradient flow
    print(f"\n📊 Creating deep network gradient flow plot...")
    plot_gradient_flow_in_deep_network()
    
    print(f"\n🎉 SUMMARY")
    print("=" * 30)
    print("Derivatives are essential because they:")
    print("  1. Enable the chain rule for backpropagation")
    print("  2. Determine learning speed and effectiveness")
    print("  3. Affect gradient flow in deep networks")
    print("  4. Influence choice of activation functions")
    print("\nWithout derivatives, neural networks couldn't learn efficiently!")
    print("\n📁 Plots saved to 'Neural Networks/images/' folder")

if __name__ == "__main__":
    main()