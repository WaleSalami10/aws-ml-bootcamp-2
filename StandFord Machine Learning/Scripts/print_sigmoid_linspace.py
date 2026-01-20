#!/usr/bin/env python3
"""
Print out sigmoid values for linspace inputs to see the pattern
"""

import numpy as np

def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))

print("🔢 SIGMOID VALUES FOR LINSPACE INPUTS")
print("=" * 60)

# Create linspace values (fewer points for readability)
x_values = np.linspace(-10, 10, 21)  # 21 points from -10 to 10
y_values = sigmoid(x_values)

print("Input (x) → Sigmoid Output (y)")
print("-" * 30)

for i, (x, y) in enumerate(zip(x_values, y_values)):
    print(f"{x:6.1f} → {y:.6f}")

print("\n" + "=" * 60)
print("🎯 PATTERN ANALYSIS:")
print("=" * 60)

# Show the pattern more clearly
print("\n📊 Key Observations:")
print(f"At x = -10: sigmoid = {sigmoid(-10):.6f} (almost 0)")
print(f"At x = -5:  sigmoid = {sigmoid(-5):.6f} (very small)")
print(f"At x = 0:   sigmoid = {sigmoid(0):.6f} (exactly 0.5)")
print(f"At x = 5:   sigmoid = {sigmoid(5):.6f} (very large)")
print(f"At x = 10:  sigmoid = {sigmoid(10):.6f} (almost 1)")

print("\n📈 The S-Curve Pattern:")
print("- Very negative inputs → outputs near 0")
print("- Around zero → outputs near 0.5") 
print("- Very positive inputs → outputs near 1")
print("- Smooth transition (no jumps)")

# Show more detailed breakdown
print("\n" + "=" * 60)
print("🔍 DETAILED BREAKDOWN BY REGIONS:")
print("=" * 60)

regions = [
    ("Very Negative", -10, -3),
    ("Negative", -3, -1), 
    ("Around Zero", -1, 1),
    ("Positive", 1, 3),
    ("Very Positive", 3, 10)
]

for region_name, start, end in regions:
    print(f"\n📍 {region_name} Region ({start} to {end}):")
    region_x = np.linspace(start, end, 5)
    region_y = sigmoid(region_x)
    
    for x, y in zip(region_x, region_y):
        print(f"   {x:6.1f} → {y:.6f}")

# Show the steepest change area
print("\n" + "=" * 60)
print("⚡ STEEPEST CHANGE AREA (around zero):")
print("=" * 60)

steep_x = np.linspace(-2, 2, 9)
steep_y = sigmoid(steep_x)

print("This is where sigmoid changes most rapidly:")
for x, y in zip(steep_x, steep_y):
    change_rate = "📈 FAST" if abs(x) < 1 else "📊 slower"
    print(f"   {x:6.1f} → {y:.6f} {change_rate}")

print("\n💡 Why this matters for neural networks:")
print("- Around zero: Network can learn quickly (big changes)")
print("- Far from zero: Network learns slowly (small changes)")
print("- This is why weight initialization matters!")