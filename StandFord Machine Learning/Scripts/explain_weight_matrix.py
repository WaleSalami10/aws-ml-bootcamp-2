#!/usr/bin/env python3
"""
Explain the weight matrix structure in detail
"""

import numpy as np
import matplotlib.pyplot as plt

print("🧠 UNDERSTANDING THE WEIGHT MATRIX")
print("=" * 60)

# The weight matrix from the notebook
weights_input_to_hidden = np.array([[0.5, 0.3],   # From input1 to [hidden1, hidden2]
                                   [0.2, 0.8]])   # From input2 to [hidden1, hidden2]

print("📊 THE WEIGHT MATRIX:")
print(weights_input_to_hidden)
print()

print("🔍 BREAKING IT DOWN:")
print("=" * 60)

print("\n📍 Matrix Structure:")
print("   Rows = Input neurons (2 inputs)")
print("   Columns = Hidden neurons (2 hidden)")
print("   Shape: (2, 2) = 2 inputs × 2 hidden neurons")

print("\n📍 What Each Number Means:")
print("   [0.5, 0.3] ← Row 1: Weights FROM Input1")
print("   [0.2, 0.8] ← Row 2: Weights FROM Input2")
print("    ↑    ↑")
print("    |    └─ TO Hidden2")
print("    └────── TO Hidden1")

print("\n🔗 CONNECTION MAPPING:")
print("=" * 60)

connections = [
    ("Input1 → Hidden1", 0, 0, 0.5),
    ("Input1 → Hidden2", 0, 1, 0.3),
    ("Input2 → Hidden1", 1, 0, 0.2),
    ("Input2 → Hidden2", 1, 1, 0.8)
]

for connection, row, col, weight in connections:
    print(f"{connection:18} = weights[{row}][{col}] = {weight}")

print("\n🎯 HOW IT WORKS IN PRACTICE:")
print("=" * 60)

# Example calculation
test_input = np.array([1.0, 0.5])
print(f"Example input: {test_input}")
print()

print("Step 1: Calculate what each HIDDEN neuron receives")
print("-" * 50)

# Manual calculation for clarity
hidden1_input = test_input[0] * weights_input_to_hidden[0,0] + test_input[1] * weights_input_to_hidden[1,0]
hidden2_input = test_input[0] * weights_input_to_hidden[0,1] + test_input[1] * weights_input_to_hidden[1,1]

print(f"Hidden1 receives:")
print(f"  Input1×Weight + Input2×Weight")
print(f"  {test_input[0]}×{weights_input_to_hidden[0,0]} + {test_input[1]}×{weights_input_to_hidden[1,0]} = {hidden1_input}")
print()

print(f"Hidden2 receives:")
print(f"  Input1×Weight + Input2×Weight") 
print(f"  {test_input[0]}×{weights_input_to_hidden[0,1]} + {test_input[1]}×{weights_input_to_hidden[1,1]} = {hidden2_input}")
print()

# Show matrix multiplication does the same thing
result_matrix = np.dot(test_input, weights_input_to_hidden)
print(f"Matrix multiplication gives same result: {result_matrix}")

print("\n🎨 VISUAL REPRESENTATION:")
print("=" * 60)

print("""
Network Structure:

Input Layer    Weights    Hidden Layer
                         
   [1.0] ──0.5──→ [Hidden1] = 1.0×0.5 + 0.5×0.2 = 0.6
      │    0.3  ╱     │
      │      ╱╲       │
      │    ╱    ╲     │
      │  ╱  0.2   ╲   │
      │╱           ╲  │
   [0.5] ──0.8──→ [Hidden2] = 1.0×0.3 + 0.5×0.8 = 0.7
                         
""")

print("🔢 MATRIX MATH EXPLANATION:")
print("=" * 60)

print("\nMatrix multiplication: input × weights = hidden_input")
print()
print("  [1.0, 0.5] × [[0.5, 0.3],  = [0.6, 0.7]")
print("                [0.2, 0.8]]")
print()
print("How it works:")
print("  Result[0] = 1.0×0.5 + 0.5×0.2 = 0.6  (Hidden1)")
print("  Result[1] = 1.0×0.3 + 0.5×0.8 = 0.7  (Hidden2)")

print("\n💡 KEY INSIGHTS:")
print("=" * 60)
print("1. Each ROW represents weights FROM one input")
print("2. Each COLUMN represents weights TO one hidden neuron")
print("3. Matrix multiplication does all calculations at once")
print("4. Bigger weights = stronger influence")
print("5. The matrix shape must be: (inputs, hidden_neurons)")

def visualize_weight_matrix():
    """Create a visual representation of the weight matrix"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Left plot: Network diagram with weights
    ax1.set_title('Network with Weight Values', fontsize=14, fontweight='bold')
    
    # Input neurons
    input_pos = [(0, 2), (0, 1)]
    hidden_pos = [(3, 2), (3, 1)]
    
    # Draw neurons
    for i, pos in enumerate(input_pos):
        circle = plt.Circle(pos, 0.2, color='lightblue', ec='black', linewidth=2)
        ax1.add_patch(circle)
        ax1.text(pos[0]-0.5, pos[1], f'Input{i+1}', ha='right', va='center', fontsize=10)
    
    for i, pos in enumerate(hidden_pos):
        circle = plt.Circle(pos, 0.2, color='lightgreen', ec='black', linewidth=2)
        ax1.add_patch(circle)
        ax1.text(pos[0]+0.5, pos[1], f'Hidden{i+1}', ha='left', va='center', fontsize=10)
    
    # Draw connections with weights
    weights = [[0.5, 0.3], [0.2, 0.8]]
    colors = ['red', 'blue']
    
    for i, i_pos in enumerate(input_pos):
        for j, h_pos in enumerate(hidden_pos):
            weight = weights[i][j]
            # Line thickness based on weight
            linewidth = weight * 5
            ax1.plot([i_pos[0]+0.2, h_pos[0]-0.2], [i_pos[1], h_pos[1]], 
                    color=colors[i], linewidth=linewidth, alpha=0.7)
            
            # Weight labels
            mid_x = (i_pos[0] + h_pos[0]) / 2
            mid_y = (i_pos[1] + h_pos[1]) / 2
            ax1.text(mid_x, mid_y+0.1, f'{weight}', ha='center', va='center', 
                    fontsize=10, fontweight='bold', 
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
    
    ax1.set_xlim(-1, 4)
    ax1.set_ylim(0.5, 2.5)
    ax1.set_aspect('equal')
    ax1.axis('off')
    
    # Right plot: Matrix visualization
    ax2.set_title('Weight Matrix Structure', fontsize=14, fontweight='bold')
    
    # Create matrix visualization
    matrix_data = np.array([[0.5, 0.3], [0.2, 0.8]])
    im = ax2.imshow(matrix_data, cmap='Blues', aspect='equal')
    
    # Add text annotations
    for i in range(2):
        for j in range(2):
            text = ax2.text(j, i, f'{matrix_data[i, j]}', 
                          ha="center", va="center", color="black", fontsize=16, fontweight='bold')
    
    ax2.set_xticks([0, 1])
    ax2.set_yticks([0, 1])
    ax2.set_xticklabels(['To Hidden1', 'To Hidden2'])
    ax2.set_yticklabels(['From Input1', 'From Input2'])
    ax2.set_xlabel('Hidden Neurons (Columns)', fontsize=12)
    ax2.set_ylabel('Input Neurons (Rows)', fontsize=12)
    
    # Add colorbar
    plt.colorbar(im, ax=ax2, shrink=0.6)
    
    plt.tight_layout()
    plt.show()

print("\n📊 Creating visualization...")
visualize_weight_matrix()

print("\n🎓 SUMMARY:")
print("The weight matrix is like a 'connection strength table'")
print("- Rows: Which input neuron the weight comes FROM")
print("- Columns: Which hidden neuron the weight goes TO") 
print("- Values: How strong each connection is")
print("- Matrix multiplication: Efficient way to calculate all connections at once!")