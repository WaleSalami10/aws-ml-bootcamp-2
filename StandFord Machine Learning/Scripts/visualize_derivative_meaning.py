#!/usr/bin/env python3
"""
Visualize What ∂a/∂z = 0.1966 Actually Means

This script creates visual demonstrations to show exactly what the activation derivative represents
and why the specific value 0.1966 is important for neural network learning.
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

def demonstrate_derivative_calculation():
    """Show the exact calculation that gives us 0.1966"""
    
    print("🧮 EXACT CALCULATION OF ∂a/∂z = 0.1966")
    print("=" * 50)
    
    z = 1.0
    print(f"Given: z = {z}")
    
    # Step 1: Calculate sigmoid
    exp_neg_z = np.exp(-z)
    sigmoid_z = 1 / (1 + exp_neg_z)
    
    print(f"\nStep 1: Calculate sigmoid(z)")
    print(f"  e^(-z) = e^(-{z}) = {exp_neg_z:.6f}")
    print(f"  sigmoid(z) = 1 / (1 + e^(-z))")
    print(f"             = 1 / (1 + {exp_neg_z:.6f})")
    print(f"             = 1 / {1 + exp_neg_z:.6f}")
    print(f"             = {sigmoid_z:.6f}")
    
    # Step 2: Calculate derivative
    one_minus_sigmoid = 1 - sigmoid_z
    derivative = sigmoid_z * one_minus_sigmoid
    
    print(f"\nStep 2: Calculate derivative")
    print(f"  ∂a/∂z = sigmoid(z) × (1 - sigmoid(z))")
    print(f"        = {sigmoid_z:.6f} × (1 - {sigmoid_z:.6f})")
    print(f"        = {sigmoid_z:.6f} × {one_minus_sigmoid:.6f}")
    print(f"        = {derivative:.6f}")
    
    print(f"\n✅ Result: ∂a/∂z = {derivative:.4f}")
    print(f"   This is our activation derivative!")

def visualize_derivative_as_slope():
    """Show the derivative as the slope of the sigmoid curve"""
    
    print("\n📈 DERIVATIVE AS SLOPE VISUALIZATION")
    print("=" * 40)
    
    # Create sigmoid curve
    x = np.linspace(-4, 4, 1000)
    y = sigmoid(x)
    
    # Our specific point
    z_point = 1.0
    a_point = sigmoid(z_point)
    slope = sigmoid_derivative(z_point)
    
    print(f"At z = {z_point}:")
    print(f"  sigmoid({z_point}) = {a_point:.4f}")
    print(f"  derivative = {slope:.4f} (this is the slope!)")
    
    # Create the plot
    plt.figure(figsize=(15, 10))
    
    # Main sigmoid curve
    plt.subplot(2, 2, 1)
    plt.plot(x, y, 'b-', linewidth=3, label='Sigmoid Function')
    
    # Mark our point
    plt.plot(z_point, a_point, 'ro', markersize=10, label=f'Point ({z_point}, {a_point:.3f})')
    
    # Draw tangent line
    tangent_x = np.linspace(z_point - 1, z_point + 1, 100)
    tangent_y = a_point + slope * (tangent_x - z_point)
    plt.plot(tangent_x, tangent_y, 'r--', linewidth=2, label=f'Tangent (slope = {slope:.3f})')
    
    plt.title('Sigmoid Function with Tangent Line', fontsize=14, fontweight='bold')
    plt.xlabel('z (input to activation)')
    plt.ylabel('a (output from activation)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    
    # Zoomed view
    plt.subplot(2, 2, 2)
    x_zoom = np.linspace(z_point - 0.5, z_point + 0.5, 200)
    y_zoom = sigmoid(x_zoom)
    plt.plot(x_zoom, y_zoom, 'b-', linewidth=3, label='Sigmoid (zoomed)')
    plt.plot(z_point, a_point, 'ro', markersize=10)
    
    # Tangent line (zoomed)
    tangent_x_zoom = np.linspace(z_point - 0.3, z_point + 0.3, 100)
    tangent_y_zoom = a_point + slope * (tangent_x_zoom - z_point)
    plt.plot(tangent_x_zoom, tangent_y_zoom, 'r--', linewidth=2, label=f'Slope = {slope:.3f}')
    
    plt.title('Zoomed View: Derivative as Slope', fontsize=14, fontweight='bold')
    plt.xlabel('z')
    plt.ylabel('a')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Show what the slope means
    plt.subplot(2, 2, 3)
    # Demonstrate small changes
    delta_z = 0.1
    z1 = z_point
    z2 = z_point + delta_z
    a1 = sigmoid(z1)
    a2 = sigmoid(z2)
    delta_a = a2 - a1
    
    plt.plot([z1, z2], [a1, a2], 'go-', linewidth=3, markersize=8, label='Actual change')
    plt.plot([z1, z2], [a1, a1 + slope * delta_z], 'ro--', linewidth=2, markersize=6, 
             label=f'Predicted change (using derivative)')
    
    # Add annotations
    plt.annotate(f'Δz = {delta_z}', xy=(z1 + delta_z/2, a1 - 0.02), ha='center', fontsize=10)
    plt.annotate(f'Δa = {delta_a:.4f}', xy=(z2 + 0.05, (a1 + a2)/2), ha='left', fontsize=10)
    plt.annotate(f'Predicted Δa = {slope * delta_z:.4f}', xy=(z2 + 0.05, a1 + slope * delta_z), 
                ha='left', fontsize=10, color='red')
    
    plt.title('What the Derivative Predicts', fontsize=14, fontweight='bold')
    plt.xlabel('z')
    plt.ylabel('a')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xlim(z1 - 0.05, z2 + 0.2)
    plt.ylim(a1 - 0.05, a2 + 0.05)
    
    # Derivative function
    plt.subplot(2, 2, 4)
    plt.plot(x, sigmoid_derivative(x), 'g-', linewidth=3, label='Sigmoid Derivative')
    plt.plot(z_point, slope, 'ro', markersize=10, label=f'Our point: {slope:.3f}')
    plt.axhline(y=slope, color='r', linestyle='--', alpha=0.5)
    plt.axvline(x=z_point, color='r', linestyle='--', alpha=0.5)
    
    plt.title('Derivative Function', fontsize=14, fontweight='bold')
    plt.xlabel('z')
    plt.ylabel('∂a/∂z')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    images_dir = Path("Neural Networks/images")
    images_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(images_dir / "derivative_as_slope.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n💡 Key Insights:")
    print(f"   • The derivative {slope:.4f} is the slope of the sigmoid curve at z = {z_point}")
    print(f"   • It predicts how much the output changes for small input changes")
    print(f"   • For Δz = {delta_z}, actual Δa = {delta_a:.4f}, predicted = {slope * delta_z:.4f}")
    print(f"   • The prediction is very accurate for small changes!")

def compare_derivative_values():
    """Compare derivative values at different points"""
    
    print("\n📊 DERIVATIVE VALUES AT DIFFERENT POINTS")
    print("=" * 50)
    
    test_points = [-3, -1, 0, 1, 2, 3]
    
    print(f"{'z':<6} {'sigmoid(z)':<12} {'derivative':<12} {'Learning Speed':<15}")
    print("-" * 50)
    
    for z in test_points:
        sig_val = sigmoid(z)
        deriv_val = sigmoid_derivative(z)
        
        if deriv_val > 0.2:
            speed = "Fast"
        elif deriv_val > 0.1:
            speed = "Good"
        elif deriv_val > 0.05:
            speed = "Slow"
        else:
            speed = "Very Slow"
        
        print(f"{z:<6} {sig_val:<12.4f} {deriv_val:<12.4f} {speed:<15}")
    
    # Visualize the comparison
    plt.figure(figsize=(12, 8))
    
    x = np.linspace(-4, 4, 1000)
    
    # Sigmoid function
    plt.subplot(2, 1, 1)
    plt.plot(x, sigmoid(x), 'b-', linewidth=3, label='Sigmoid Function')
    
    # Mark test points
    for z in test_points:
        plt.plot(z, sigmoid(z), 'ro', markersize=8)
        plt.annotate(f'z={z}', xy=(z, sigmoid(z)), xytext=(z, sigmoid(z) + 0.1), 
                    ha='center', fontsize=10)
    
    plt.title('Sigmoid Function with Test Points', fontsize=14, fontweight='bold')
    plt.xlabel('z')
    plt.ylabel('sigmoid(z)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Derivative function
    plt.subplot(2, 1, 2)
    plt.plot(x, sigmoid_derivative(x), 'g-', linewidth=3, label='Sigmoid Derivative')
    
    # Mark test points and color-code by learning speed
    colors = ['red', 'orange', 'green', 'green', 'orange', 'red']
    speeds = ['Very Slow', 'Slow', 'Fast', 'Good', 'Slow', 'Very Slow']
    
    for z, color, speed in zip(test_points, colors, speeds):
        plt.plot(z, sigmoid_derivative(z), 'o', color=color, markersize=10)
        plt.annotate(f'{sigmoid_derivative(z):.3f}\\n({speed})', 
                    xy=(z, sigmoid_derivative(z)), xytext=(z, sigmoid_derivative(z) + 0.03), 
                    ha='center', fontsize=9, color=color)
    
    # Highlight our specific point
    our_z = 1.0
    our_deriv = sigmoid_derivative(our_z)
    plt.plot(our_z, our_deriv, 'bo', markersize=15, markerfacecolor='none', markeredgewidth=3)
    plt.annotate(f'Our point: {our_deriv:.4f}\\n(Good for learning!)', 
                xy=(our_z, our_deriv), xytext=(our_z + 0.5, our_deriv + 0.05), 
                ha='left', fontsize=12, color='blue', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='blue'))
    
    plt.title('Derivative Values and Learning Speed', fontsize=14, fontweight='bold')
    plt.xlabel('z')
    plt.ylabel('∂a/∂z')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(images_dir / "derivative_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n🎯 Our derivative 0.1966 at z=1.0 is in the 'Good' range!")
    print(f"   • Not too small (avoids vanishing gradients)")
    print(f"   • Not too large (avoids instability)")
    print(f"   • Just right for effective learning!")

def demonstrate_learning_impact():
    """Show how the derivative value affects learning"""
    
    print("\n🎓 HOW DERIVATIVE VALUE AFFECTS LEARNING")
    print("=" * 45)
    
    # Simulate learning at different z values
    scenarios = [
        ("Small derivative (z=3)", 3.0),
        ("Good derivative (z=1)", 1.0),
        ("Maximum derivative (z=0)", 0.0)
    ]
    
    learning_rate = 0.1
    target = 1.0
    initial_weight = 0.2
    input_val = 2.0
    
    plt.figure(figsize=(15, 5))
    
    for i, (name, z_bias) in enumerate(scenarios):
        plt.subplot(1, 3, i+1)
        
        # Simulate learning
        weights = [initial_weight]
        losses = []
        
        w = initial_weight
        for step in range(20):
            # Forward pass
            z = w * input_val + z_bias  # Add bias to control where we are on sigmoid
            a = sigmoid(z)
            loss = (a - target) ** 2
            losses.append(loss)
            
            # Backward pass
            derivative = sigmoid_derivative(z)
            gradient = 2 * (a - target) * derivative * input_val
            w -= learning_rate * gradient
            weights.append(w)
        
        # Plot learning curve
        plt.plot(losses, linewidth=3, label=f'{name}')
        plt.title(f'{name}\\nDerivative ≈ {sigmoid_derivative(z_bias):.3f}', 
                 fontsize=12, fontweight='bold')
        plt.xlabel('Training Step')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
        
        # Add final loss annotation
        plt.annotate(f'Final loss: {losses[-1]:.4f}', 
                    xy=(len(losses)-1, losses[-1]), xytext=(len(losses)-5, losses[-1]*2),
                    arrowprops=dict(arrowstyle='->', color='red'),
                    fontsize=10, color='red')
    
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(images_dir / "learning_speed_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"💡 Key Observations:")
    print(f"   • Small derivatives → slow learning (high final loss)")
    print(f"   • Good derivatives → effective learning (low final loss)")
    print(f"   • Maximum derivatives → fastest learning (lowest final loss)")
    print(f"   • Our 0.1966 derivative gives good learning performance!")

def main():
    """Main demonstration function"""
    
    print("🔍 UNDERSTANDING ∂a/∂z = 0.1966")
    print("=" * 40)
    print("Let's explore exactly what this activation derivative means!")
    
    # 1. Show exact calculation
    demonstrate_derivative_calculation()
    
    # 2. Visualize as slope
    visualize_derivative_as_slope()
    
    # 3. Compare different values
    compare_derivative_values()
    
    # 4. Show learning impact
    demonstrate_learning_impact()
    
    print(f"\n🎉 SUMMARY")
    print("=" * 20)
    print("∂a/∂z = 0.1966 means:")
    print("  1. The sigmoid curve has a slope of 0.1966 at z=1.0")
    print("  2. Small changes in z cause proportional changes in a")
    print("  3. This derivative enables effective neural network learning")
    print("  4. It's in the 'sweet spot' - not too small, not too large")
    print("  5. This is why the network can learn efficiently!")
    
    print(f"\n📁 Visualizations saved to 'Neural Networks/images/' folder")

if __name__ == "__main__":
    main()