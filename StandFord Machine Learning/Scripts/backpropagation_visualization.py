"""
Backpropagation Visualization Script
This script creates a clear diagram showing how backpropagation works
"""

import matplotlib.pyplot as plt
import numpy as np

def visualize_backpropagation():
    """Create a diagram showing how backpropagation works"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # TOP DIAGRAM: Forward Pass
    ax1.set_title('FORWARD PASS: Data flows forward →', fontsize=16, fontweight='bold', color='blue')
    
    # Draw network structure
    # Input layer
    ax1.scatter([1, 1], [3, 1], s=300, c='lightblue', edgecolors='black', linewidth=2)
    ax1.text(0.5, 3, 'Input 1\n[0.5]', ha='center', va='center', fontsize=10, fontweight='bold')
    ax1.text(0.5, 1, 'Input 2\n[0.8]', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Hidden layer
    ax1.scatter([4, 4], [3, 1], s=300, c='lightgreen', edgecolors='black', linewidth=2)
    ax1.text(4, 3.5, 'Hidden 1\n[0.73]', ha='center', va='center', fontsize=10, fontweight='bold')
    ax1.text(4, 0.5, 'Hidden 2\n[0.68]', ha='center', va='center', fontsize=10, fontweight='bold')
    
    # Output layer
    ax1.scatter([7], [2], s=300, c='lightcoral', edgecolors='black', linewidth=2)
    ax1.text(7, 2.5, 'Output\n[0.65]', ha='center', va='center', fontsize=10, fontweight='bold')
    ax1.text(8.5, 2, 'Target: 1.0\nError: -0.35', ha='left', va='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    # Forward arrows
    ax1.annotate('', xy=(3.7, 3), xytext=(1.3, 3), arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
    ax1.annotate('', xy=(3.7, 1), xytext=(1.3, 1), arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
    ax1.annotate('', xy=(3.7, 3), xytext=(1.3, 1), arrowprops=dict(arrowstyle='->', lw=2, color='blue', alpha=0.5))
    ax1.annotate('', xy=(3.7, 1), xytext=(1.3, 3), arrowprops=dict(arrowstyle='->', lw=2, color='blue', alpha=0.5))
    ax1.annotate('', xy=(6.7, 2), xytext=(4.3, 3), arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
    ax1.annotate('', xy=(6.7, 2), xytext=(4.3, 1), arrowprops=dict(arrowstyle='->', lw=3, color='blue'))
    
    # Weight labels
    ax1.text(2.5, 3.5, 'W₁₁', ha='center', va='center', fontsize=12, fontweight='bold', color='blue')
    ax1.text(2.5, 0.5, 'W₂₂', ha='center', va='center', fontsize=12, fontweight='bold', color='blue')
    ax1.text(5.5, 2.5, 'W₃', ha='center', va='center', fontsize=12, fontweight='bold', color='blue')
    
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 4)
    ax1.axis('off')
    
    # BOTTOM DIAGRAM: Backward Pass
    ax2.set_title('BACKWARD PASS: Error flows backward ← (Learning happens here!)', fontsize=16, fontweight='bold', color='red')
    
    # Draw same network structure
    # Input layer
    ax2.scatter([1, 1], [3, 1], s=300, c='lightblue', edgecolors='black', linewidth=2)
    ax2.text(0.5, 3, 'Input 1\n[0.5]', ha='center', va='center', fontsize=10)
    ax2.text(0.5, 1, 'Input 2\n[0.8]', ha='center', va='center', fontsize=10)
    
    # Hidden layer
    ax2.scatter([4, 4], [3, 1], s=300, c='lightgreen', edgecolors='black', linewidth=2)
    ax2.text(4, 3.5, 'Hidden 1\nδ₁ = -0.05', ha='center', va='center', fontsize=9, fontweight='bold')
    ax2.text(4, 0.5, 'Hidden 2\nδ₂ = -0.04', ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Output layer
    ax2.scatter([7], [2], s=300, c='lightcoral', edgecolors='black', linewidth=2)
    ax2.text(7, 2.5, 'Output\nδ = -0.08', ha='center', va='center', fontsize=10, fontweight='bold')
    ax2.text(8.5, 2, 'Error: -0.35\n(Too low!)', ha='left', va='center', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    # Backward arrows (error propagation)
    ax2.annotate('', xy=(4.3, 3), xytext=(6.7, 2), arrowprops=dict(arrowstyle='->', lw=4, color='red'))
    ax2.annotate('', xy=(4.3, 1), xytext=(6.7, 2), arrowprops=dict(arrowstyle='->', lw=4, color='red'))
    ax2.annotate('', xy=(1.3, 3), xytext=(3.7, 3), arrowprops=dict(arrowstyle='->', lw=3, color='red', alpha=0.7))
    ax2.annotate('', xy=(1.3, 1), xytext=(3.7, 1), arrowprops=dict(arrowstyle='->', lw=3, color='red', alpha=0.7))
    ax2.annotate('', xy=(1.3, 3), xytext=(3.7, 1), arrowprops=dict(arrowstyle='->', lw=2, color='red', alpha=0.5))
    ax2.annotate('', xy=(1.3, 1), xytext=(3.7, 3), arrowprops=dict(arrowstyle='->', lw=2, color='red', alpha=0.5))
    
    # Weight update labels
    ax2.text(2.5, 3.5, 'ΔW₁₁\n+0.02', ha='center', va='center', fontsize=10, fontweight='bold', 
             color='red', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.text(2.5, 0.5, 'ΔW₂₂\n+0.03', ha='center', va='center', fontsize=10, fontweight='bold', 
             color='red', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax2.text(5.5, 2.5, 'ΔW₃\n+0.06', ha='center', va='center', fontsize=10, fontweight='bold', 
             color='red', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlim(0, 10)
    ax2.set_ylim(0, 4)
    ax2.axis('off')
    
    # Add explanatory text
    fig.text(0.02, 0.5, 'STEP 1: Calculate Error\nSTEP 2: Propagate Error Backward\nSTEP 3: Calculate Weight Updates\nSTEP 4: Update Weights', 
             fontsize=12, va='center', ha='left', 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('StandFord Machine Learning/Neural Networks/images/backpropagation_diagram.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n🎯 KEY CONCEPTS:")
    print("• δ (delta) = how much each neuron contributed to the error")
    print("• ΔW (delta W) = how much to change each weight")
    print("• Error flows BACKWARD through the network")
    print("• Each weight gets updated based on its contribution to the error")
    print("• This process repeats thousands of times until the network learns!")

def explain_backpropagation_steps():
    """Explain backpropagation in simple steps"""
    print("\n" + "="*60)
    print("🔍 BACKPROPAGATION STEP-BY-STEP EXPLANATION")
    print("="*60)
    
    print("\n📊 EXAMPLE: Network predicts 0.65, but target is 1.0")
    print("\n🔴 STEP 1: Calculate Output Error")
    print("   • Error = Predicted - Target = 0.65 - 1.0 = -0.35")
    print("   • The network's output is too LOW by 0.35")
    
    print("\n🔴 STEP 2: Calculate Output Delta (δ)")
    print("   • δ_output = Error × sigmoid_derivative(output_input)")
    print("   • δ_output = -0.35 × 0.23 = -0.08")
    print("   • This tells us HOW MUCH the output neuron should change")
    
    print("\n🔴 STEP 3: Propagate Error to Hidden Layer")
    print("   • Each hidden neuron gets blamed based on its weight to output")
    print("   • Hidden1_error = δ_output × weight_hidden1_to_output")
    print("   • Hidden2_error = δ_output × weight_hidden2_to_output")
    print("   • This spreads the blame backward!")
    
    print("\n🔴 STEP 4: Calculate Weight Updates")
    print("   • For each weight: ΔW = learning_rate × input × δ")
    print("   • If input was HIGH and δ is NEGATIVE → decrease weight")
    print("   • If input was LOW and δ is NEGATIVE → increase weight less")
    print("   • This makes the network output HIGHER next time!")
    
    print("\n🔴 STEP 5: Update All Weights")
    print("   • W_new = W_old - learning_rate × ΔW")
    print("   • Every weight in the network gets adjusted")
    print("   • The network is now slightly better!")
    
    print("\n🎯 THE MAGIC:")
    print("   • This happens AUTOMATICALLY for ANY error")
    print("   • The network learns from EVERY mistake")
    print("   • After thousands of examples, it becomes very good!")
    print("\n" + "="*60)

if __name__ == "__main__":
    print("🧠 BACKPROPAGATION VISUALIZATION")
    print("="*50)
    
    # Create the images directory if it doesn't exist
    import os
    os.makedirs('StandFord Machine Learning/Neural Networks/images', exist_ok=True)
    
    # Show the visualization
    visualize_backpropagation()
    
    # Show the step-by-step explanation
    explain_backpropagation_steps()