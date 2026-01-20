#!/usr/bin/env python3
"""
Print ALL sigmoid values for the full linspace (100 points)
"""

import numpy as np

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

print("🔢 ALL 100 SIGMOID VALUES FROM LINSPACE(-10, 10, 100)")
print("=" * 70)

# Create the exact same linspace as in the notebook
x_values = np.linspace(-10, 10, 100)
y_values = sigmoid(x_values)

print("Index | Input (x) | Sigmoid Output (y)")
print("-" * 40)

for i, (x, y) in enumerate(zip(x_values, y_values)):
    print(f"{i:3d}   | {x:8.3f} | {y:.8f}")

print("\n" + "=" * 70)
print("📊 SUMMARY STATISTICS:")
print("=" * 70)

print(f"Total points: {len(x_values)}")
print(f"Input range: {x_values[0]:.1f} to {x_values[-1]:.1f}")
print(f"Output range: {y_values[0]:.8f} to {y_values[-1]:.8f}")
print(f"Step size: {x_values[1] - x_values[0]:.6f}")

# Find where sigmoid is closest to key values
closest_to_quarter = np.argmin(np.abs(y_values - 0.25))
closest_to_half = np.argmin(np.abs(y_values - 0.5))
closest_to_three_quarter = np.argmin(np.abs(y_values - 0.75))

print(f"\n🎯 Key Points:")
print(f"Closest to 0.25: x={x_values[closest_to_quarter]:.3f}, y={y_values[closest_to_quarter]:.6f}")
print(f"Closest to 0.50: x={x_values[closest_to_half]:.3f}, y={y_values[closest_to_half]:.6f}")
print(f"Closest to 0.75: x={x_values[closest_to_three_quarter]:.3f}, y={y_values[closest_to_three_quarter]:.6f}")

print(f"\n💡 This is the exact data used to create the smooth sigmoid curve!")