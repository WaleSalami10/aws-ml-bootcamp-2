#!/usr/bin/env python3
"""
Explain the difference between test values and linspace values
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

print("🎯 UNDERSTANDING SIGMOID INPUTS")
print("=" * 50)

# Method 1: Test specific values (for understanding)
print("\n📋 METHOD 1: Test Specific Values")
print("Purpose: Understand how sigmoid works with exact numbers")
print("-" * 30)

test_values = [-5, -2, 0, 2, 5]
print("Input → Sigmoid Output")
for val in test_values:
    result = sigmoid(val)
    print(f"{val:2d} → {result:.3f}")

print("\n💡 What this shows:")
print("- Negative numbers → close to 0")
print("- Zero → exactly 0.5") 
print("- Positive numbers → close to 1")

# Method 2: Linspace values (for plotting)
print("\n📊 METHOD 2: Linspace Values") 
print("Purpose: Create smooth curve for visualization")
print("-" * 30)

x_plot = np.linspace(-10, 10, 100)  # 100 points from -10 to 10
y_plot = sigmoid(x_plot)

print(f"Created {len(x_plot)} points for plotting")
print(f"First few x values: {x_plot[:5]}")
print(f"Last few x values: {x_plot[-5:]}")
print(f"First few y values: {y_plot[:5]}")

print("\n💡 What this shows:")
print("- Smooth S-shaped curve")
print("- Gradual transition from 0 to 1")
print("- No sudden jumps")

# Create visualization showing both
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Test values only
ax1.scatter(test_values, [sigmoid(x) for x in test_values], 
           color='red', s=100, zorder=5)
for i, val in enumerate(test_values):
    ax1.annotate(f'({val}, {sigmoid(val):.3f})', 
                (val, sigmoid(val)), 
                xytext=(5, 5), textcoords='offset points',
                fontsize=10)

ax1.set_title('Method 1: Specific Test Values\n(Understanding)', fontweight='bold')
ax1.set_xlabel('Input')
ax1.set_ylabel('Sigmoid Output')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(-6, 6)
ax1.set_ylim(-0.1, 1.1)

# Plot 2: Smooth curve with linspace
ax2.plot(x_plot, y_plot, 'b-', linewidth=2, label='Sigmoid curve')
ax2.scatter(test_values, [sigmoid(x) for x in test_values], 
           color='red', s=100, zorder=5, label='Test points')

ax2.set_title('Method 2: Smooth Curve with Linspace\n(Visualization)', fontweight='bold')
ax2.set_xlabel('Input')
ax2.set_ylabel('Sigmoid Output')
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

print("\n🎓 SUMMARY:")
print("- Test values: Learn exact behavior")
print("- Linspace values: See the full picture")
print("- Both together: Complete understanding!")