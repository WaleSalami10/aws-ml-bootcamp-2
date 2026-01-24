import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

def draw_neural_network(layer_dims, figsize=(16, 10), show_batch_norm=True, show_dropout=True):
    """
    Draw a neural network diagram for the FiveLayerNN architecture.

    Args:
        layer_dims: List of layer sizes [n_x, n1, n2, n3, n4, n5]
        figsize: Figure size tuple
        show_batch_norm: Whether to show batch norm annotations
        show_dropout: Whether to show dropout annotations
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(-0.5, 6.5)
    ax.set_ylim(-1.5, 1.5)
    ax.axis('off')
    ax.set_aspect('equal')

    # Layer positions (x-coordinates)
    layer_positions = [0, 1, 2, 3, 4, 5, 6]

    # Colors for different layer types
    colors = {
        'input': '#3498db',      # Blue
        'hidden': '#2ecc71',     # Green
        'output': '#e74c3c',     # Red
        'connection': '#95a5a6'  # Gray
    }

    # Layer labels
    layer_names = ['Input\n(X)', 'Layer 1\nReLU', 'Layer 2\nReLU',
                   'Layer 3\nReLU', 'Layer 4\nReLU', 'Output\nSigmoid']

    # Maximum neurons to display per layer (for visualization)
    max_display = 6

    # Store neuron positions for connections
    neuron_positions = []

    for layer_idx, (x_pos, n_neurons) in enumerate(zip(layer_positions, layer_dims)):
        # Determine how many neurons to display
        n_display = min(n_neurons, max_display)
        show_dots = n_neurons > max_display

        # Calculate vertical positions
        if show_dots:
            # Leave space for "..." in the middle
            top_neurons = n_display // 2
            bottom_neurons = n_display - top_neurons
            y_positions_top = np.linspace(0.8, 0.4, top_neurons)
            y_positions_bottom = np.linspace(-0.4, -0.8, bottom_neurons)
            y_positions = list(y_positions_top) + list(y_positions_bottom)
        else:
            if n_neurons == 1:
                y_positions = [0]
            else:
                y_positions = np.linspace(0.8, -0.8, n_display)

        layer_neuron_pos = []

        # Choose color based on layer type
        if layer_idx == 0:
            color = colors['input']
        elif layer_idx == len(layer_dims) - 1:
            color = colors['output']
        else:
            color = colors['hidden']

        # Draw neurons
        for i, y_pos in enumerate(y_positions):
            circle = plt.Circle((x_pos, y_pos), 0.08, color=color, ec='black', linewidth=1.5, zorder=3)
            ax.add_patch(circle)
            layer_neuron_pos.append((x_pos, y_pos))

        # Add "..." for truncated layers
        if show_dots:
            ax.text(x_pos, 0, '...', fontsize=16, ha='center', va='center', fontweight='bold')

        neuron_positions.append(layer_neuron_pos)

        # Add layer labels
        ax.text(x_pos, -1.2, layer_names[layer_idx], fontsize=10, ha='center', va='top', fontweight='bold')

        # Add neuron count
        ax.text(x_pos, 1.1, f'n={n_neurons}', fontsize=9, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray'))

    # Draw connections between layers
    for layer_idx in range(len(layer_dims) - 1):
        for start_pos in neuron_positions[layer_idx]:
            for end_pos in neuron_positions[layer_idx + 1]:
                # Draw lighter connections
                ax.plot([start_pos[0], end_pos[0]], [start_pos[1], end_pos[1]],
                       color=colors['connection'], linewidth=0.3, alpha=0.5, zorder=1)

    # Add annotations for operations between layers
    annotations = []

    for layer_idx in range(1, len(layer_dims)):
        x_mid = layer_positions[layer_idx] - 0.5

        if layer_idx < len(layer_dims) - 1:  # Hidden layers
            ops = ['W·A + b']
            if show_batch_norm:
                ops.append('BatchNorm')
            ops.append('ReLU')
            if show_dropout:
                ops.append('Dropout')
        else:  # Output layer
            ops = ['W·A + b', 'Sigmoid']

        annotation_text = '\n'.join(ops)
        annotations.append((x_mid, 1.35, annotation_text))

    for x, y, text in annotations:
        ax.text(x, y, text, fontsize=7, ha='center', va='bottom',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='lightyellow', edgecolor='orange', alpha=0.8))

    # Add title
    ax.set_title('5-Layer Neural Network Architecture\n' +
                 f'Layer Dimensions: {layer_dims}', fontsize=14, fontweight='bold', pad=20)

    # Add legend
    legend_elements = [
        mpatches.Patch(color=colors['input'], label='Input Layer'),
        mpatches.Patch(color=colors['hidden'], label='Hidden Layers (ReLU)'),
        mpatches.Patch(color=colors['output'], label='Output Layer (Sigmoid)'),
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

    # Add formula annotations at the bottom
    formulas = [
        "Forward: Z[l] = W[l]·A[l-1] + b[l], A[l] = g(Z[l])",
        "Backward: dZ[l] = dA[l] * g'(Z[l]), dW[l] = (1/m)·dZ[l]·A[l-1]ᵀ",
        "Update: W[l] = W[l] - α·dW[l] (or Adam/Momentum/RMSprop)"
    ]
    for i, formula in enumerate(formulas):
        ax.text(3, -1.45 - i*0.12, formula, fontsize=8, ha='center', va='top',
               family='monospace', style='italic')

    plt.tight_layout()
    return fig, ax


def draw_detailed_network(layer_dims, figsize=(18, 12)):
    """
    Draw a more detailed neural network diagram showing all components.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Create a more detailed view with boxes for each layer
    n_layers = len(layer_dims)
    box_width = 1.2
    box_spacing = 2.0

    colors = {
        'input': '#3498db',
        'linear': '#f39c12',
        'batchnorm': '#9b59b6',
        'activation': '#2ecc71',
        'dropout': '#e74c3c',
        'output': '#1abc9c'
    }

    x_positions = []
    current_x = 0

    # Input layer
    rect = mpatches.FancyBboxPatch((current_x, 0.2), box_width, 0.6,
                                    boxstyle="round,pad=0.02",
                                    facecolor=colors['input'], edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    ax.text(current_x + box_width/2, 0.5, f'Input\n(n={layer_dims[0]})',
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')
    x_positions.append(current_x + box_width/2)
    current_x += box_spacing

    # Hidden layers (1-4)
    for l in range(1, 5):
        layer_x = current_x

        # Linear transformation
        rect = mpatches.FancyBboxPatch((layer_x, 0.6), box_width, 0.3,
                                        boxstyle="round,pad=0.02",
                                        facecolor=colors['linear'], edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(layer_x + box_width/2, 0.75, f'Linear\nW{l}·A + b{l}',
                ha='center', va='center', fontsize=8, fontweight='bold')

        # Batch Normalization
        rect = mpatches.FancyBboxPatch((layer_x, 0.25), box_width, 0.3,
                                        boxstyle="round,pad=0.02",
                                        facecolor=colors['batchnorm'], edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(layer_x + box_width/2, 0.4, 'BatchNorm\n(optional)',
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')

        # ReLU Activation
        rect = mpatches.FancyBboxPatch((layer_x, -0.1), box_width, 0.3,
                                        boxstyle="round,pad=0.02",
                                        facecolor=colors['activation'], edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(layer_x + box_width/2, 0.05, 'ReLU\nmax(0, Z)',
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')

        # Dropout
        rect = mpatches.FancyBboxPatch((layer_x, -0.45), box_width, 0.3,
                                        boxstyle="round,pad=0.02",
                                        facecolor=colors['dropout'], edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        ax.text(layer_x + box_width/2, -0.3, 'Dropout\n(optional)',
                ha='center', va='center', fontsize=8, fontweight='bold', color='white')

        # Layer label
        ax.text(layer_x + box_width/2, 1.0, f'Layer {l}\n(n={layer_dims[l]})',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

        x_positions.append(layer_x + box_width/2)
        current_x += box_spacing

    # Output layer (Layer 5)
    layer_x = current_x

    # Linear transformation
    rect = mpatches.FancyBboxPatch((layer_x, 0.35), box_width, 0.3,
                                    boxstyle="round,pad=0.02",
                                    facecolor=colors['linear'], edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(layer_x + box_width/2, 0.5, f'Linear\nW5·A + b5',
            ha='center', va='center', fontsize=8, fontweight='bold')

    # Sigmoid
    rect = mpatches.FancyBboxPatch((layer_x, 0), box_width, 0.3,
                                    boxstyle="round,pad=0.02",
                                    facecolor=colors['output'], edgecolor='black', linewidth=1)
    ax.add_patch(rect)
    ax.text(layer_x + box_width/2, 0.15, 'Sigmoid\n1/(1+e⁻ᶻ)',
            ha='center', va='center', fontsize=8, fontweight='bold', color='white')

    ax.text(layer_x + box_width/2, 0.75, f'Output (Layer 5)\n(n={layer_dims[5]})',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

    x_positions.append(layer_x + box_width/2)

    # Draw arrows between layers
    for i in range(len(x_positions) - 1):
        ax.annotate('', xy=(x_positions[i+1] - box_width/2 - 0.1, 0.5 if i == 0 else 0.2),
                    xytext=(x_positions[i] + box_width/2 + 0.1, 0.5 if i == 0 else 0.2),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Add legend
    legend_elements = [
        mpatches.Patch(color=colors['input'], label='Input'),
        mpatches.Patch(color=colors['linear'], label='Linear Transform (W·A + b)'),
        mpatches.Patch(color=colors['batchnorm'], label='Batch Normalization'),
        mpatches.Patch(color=colors['activation'], label='ReLU Activation'),
        mpatches.Patch(color=colors['dropout'], label='Dropout'),
        mpatches.Patch(color=colors['output'], label='Sigmoid Output'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)

    # Set axis limits and title
    ax.set_xlim(-0.5, current_x + box_width + 0.5)
    ax.set_ylim(-0.8, 1.4)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('5-Layer Neural Network - Detailed Architecture\n' +
                 f'Layer Dimensions: {layer_dims}', fontsize=14, fontweight='bold')

    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # Default architecture from five_layer_nn.py
    layer_dims = [2, 20, 15, 10, 5, 1]

    # Generate simple network diagram
    fig1, ax1 = draw_neural_network(layer_dims)
    fig1.savefig('network_diagram.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved: network_diagram.png")

    # Generate detailed network diagram
    fig2, ax2 = draw_detailed_network(layer_dims)
    fig2.savefig('network_diagram_detailed.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved: network_diagram_detailed.png")

    plt.show()
