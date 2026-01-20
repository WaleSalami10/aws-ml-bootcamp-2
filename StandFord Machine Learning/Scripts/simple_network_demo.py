#!/usr/bin/env python3
"""
Simple Neural Network Demo - Beginner Friendly
Shows how data flows through layers step by step
"""

import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    """Simple sigmoid function - converts any number to 0-1 range"""
    return 1 / (1 + np.exp(-x))

class BeginnerNeuralNetwork:
    """A very simple neural network for learning purposes"""
    
    def __init__(self):
        print("🧠 Creating Simple Neural Network")
        print("Architecture: 2 inputs → 2 hidden → 1 output")
        
        # Simple weights (we'll set these manually for clarity)
        self.w_input_hidden = np.array([[0.5, 0.3],    # From input1 to [hidden1, hidden2]
                                       [0.2, 0.8]])    # From input2 to [hidden1, hidden2]
        
        self.w_hidden_output = np.array([[0.6],         # From hidden1 to output
                                        [0.4]])         # From hidden2 to output
        
        print("✅ Network created!")
    
    def forward_pass_explained(self, inputs):
        """Show step-by-step how data flows through the network"""
        print(f"\n🔄 FORWARD PASS with input {inputs}")
        print("=" * 50)
        
        # Step 1: Input Layer (just pass through)
        print(f"📥 INPUT LAYER: {inputs}")
        print("   (No processing - just pass the values forward)")
        
        # Step 2: Input → Hidden Layer
        print(f"\n🔗 INPUT → HIDDEN LAYER:")
        hidden_input = np.dot(inputs, self.w_input_hidden)
        print(f"   Multiply inputs by weights and sum:")
        print(f"   Hidden1 = {inputs[0]}×{self.w_input_hidden[0,0]} + {inputs[1]}×{self.w_input_hidden[1,0]} = {hidden_input[0]:.3f}")
        print(f"   Hidden2 = {inputs[0]}×{self.w_input_hidden[0,1]} + {inputs[1]}×{self.w_input_hidden[1,1]} = {hidden_input[1]:.3f}")
        
        # Apply sigmoid activation
        hidden_output = sigmoid(hidden_input)
        print(f"\n🎯 HIDDEN LAYER ACTIVATION (apply sigmoid):")
        print(f"   Hidden1 output = sigmoid({hidden_input[0]:.3f}) = {hidden_output[0]:.3f}")
        print(f"   Hidden2 output = sigmoid({hidden_input[1]:.3f}) = {hidden_output[1]:.3f}")
        
        # Step 3: Hidden → Output Layer
        print(f"\n🔗 HIDDEN → OUTPUT LAYER:")
        print(f"   The hidden outputs become inputs to output layer:")
        output_input = np.dot(hidden_output, self.w_hidden_output)
        print(f"   Output = {hidden_output[0]:.3f}×{self.w_hidden_output[0,0]} + {hidden_output[1]:.3f}×{self.w_hidden_output[1,0]} = {output_input[0]:.3f}")
        
        # Final activation
        final_output = sigmoid(output_input)
        print(f"\n🎯 OUTPUT LAYER ACTIVATION:")
        print(f"   Final output = sigmoid({output_input[0]:.3f}) = {final_output[0]:.3f}")
        
        print(f"\n🏁 FINAL RESULT: {final_output[0]:.4f}")
        print("=" * 50)
        
        return hidden_output, final_output

def demonstrate_key_concept():
    """Show the key concept: layer outputs become next layer inputs"""
    print("\n" + "="*60)
    print("🎓 KEY CONCEPT: Layer Outputs → Next Layer Inputs")
    print("="*60)
    
    network = BeginnerNeuralNetwork()
    
    # Test different inputs
    test_cases = [
        [1.0, 0.0],
        [0.0, 1.0], 
        [0.5, 0.5],
        [1.0, 1.0]
    ]
    
    for i, test_input in enumerate(test_cases):
        print(f"\n📋 Example {i+1}: Input = {test_input}")
        print("-" * 30)
        
        # Show the flow clearly
        inputs = np.array(test_input)
        
        # Input → Hidden
        hidden_raw = np.dot(inputs, network.w_input_hidden)
        hidden_activated = sigmoid(hidden_raw)
        
        # Hidden → Output  
        output_raw = np.dot(hidden_activated, network.w_hidden_output)
        output_final = sigmoid(output_raw)
        
        print(f"Layer 1 (Input):  {inputs} → (pass through)")
        print(f"Layer 2 (Hidden): {hidden_activated} ← (after processing)")
        print(f"Layer 3 (Output): {output_final} ← (after processing)")
        print(f"")
        print(f"Notice: Hidden layer output {hidden_activated}")
        print(f"        becomes input to output layer!")

def create_simple_visualization():
    """Create a simple network diagram"""
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Network positions
    input_pos = [(0, 2), (0, 1)]
    hidden_pos = [(3, 2), (3, 1)]  
    output_pos = [(6, 1.5)]
    
    # Draw neurons
    for pos in input_pos:
        circle = plt.Circle(pos, 0.3, color='lightblue', ec='black', linewidth=2)
        ax.add_patch(circle)
    
    for pos in hidden_pos:
        circle = plt.Circle(pos, 0.3, color='lightgreen', ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], 'σ', ha='center', va='center', fontsize=14, fontweight='bold')
    
    for pos in output_pos:
        circle = plt.Circle(pos, 0.3, color='lightcoral', ec='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(pos[0], pos[1], 'σ', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Draw connections
    for i_pos in input_pos:
        for h_pos in hidden_pos:
            ax.plot([i_pos[0]+0.3, h_pos[0]-0.3], [i_pos[1], h_pos[1]], 'k-', alpha=0.4, linewidth=2)
    
    for h_pos in hidden_pos:
        for o_pos in output_pos:
            ax.plot([h_pos[0]+0.3, o_pos[0]-0.3], [h_pos[1], o_pos[1]], 'k-', alpha=0.4, linewidth=2)
    
    # Labels
    ax.text(-0.5, 2, 'Input 1', ha='right', va='center', fontsize=12)
    ax.text(-0.5, 1, 'Input 2', ha='right', va='center', fontsize=12)
    ax.text(3.5, 2, 'Hidden 1', ha='left', va='center', fontsize=12)
    ax.text(3.5, 1, 'Hidden 2', ha='left', va='center', fontsize=12)
    ax.text(6.5, 1.5, 'Output', ha='left', va='center', fontsize=12)
    
    # Layer titles
    ax.text(0, 0.2, 'INPUT\nLAYER', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(3, 0.2, 'HIDDEN\nLAYER', ha='center', va='center', fontsize=14, fontweight='bold')
    ax.text(6, 0.2, 'OUTPUT\nLAYER', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Flow arrows
    ax.annotate('Data flows →', xy=(1.5, 2.7), xytext=(1.5, 2.7), 
                fontsize=12, ha='center', color='red', fontweight='bold')
    ax.annotate('', xy=(2.5, 2.5), xytext=(0.5, 2.5), 
                arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    ax.annotate('', xy=(5.5, 2.5), xytext=(3.5, 2.5), 
                arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    
    ax.set_xlim(-1, 7)
    ax.set_ylim(-0.5, 3)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Simple Neural Network: 2→2→1 Architecture', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    print("🎓 SIMPLE NEURAL NETWORK TUTORIAL")
    print("="*50)
    
    # Create and demonstrate the network
    network = BeginnerNeuralNetwork()
    
    # Show detailed forward pass
    test_input = np.array([1.0, 0.5])
    hidden, output = network.forward_pass_explained(test_input)
    
    # Demonstrate key concept
    demonstrate_key_concept()
    
    # Create visualization
    print("\n📊 Creating network visualization...")
    create_simple_visualization()
    
    print("\n🎯 KEY TAKEAWAYS:")
    print("1. Data flows: Input → Hidden → Output")
    print("2. Each layer's output becomes the next layer's input")
    print("3. Sigmoid function converts numbers to 0-1 range")
    print("4. Weights control how much influence each connection has")
    print("\nNext: Learn how to train the network to make better predictions!")