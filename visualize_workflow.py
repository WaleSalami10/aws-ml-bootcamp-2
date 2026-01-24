import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def draw_training_workflow(figsize=(18, 14)):
    """
    Draw the complete training workflow for the 5-Layer Neural Network.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 14)
    ax.axis('off')

    # Colors
    colors = {
        'input': '#3498db',
        'forward': '#2ecc71',
        'loss': '#e74c3c',
        'backward': '#9b59b6',
        'update': '#f39c12',
        'output': '#1abc9c',
        'decision': '#e67e22',
        'process': '#34495e',
        'optional': '#95a5a6'
    }

    def draw_box(x, y, width, height, text, color, fontsize=9, textcolor='white'):
        rect = FancyBboxPatch((x, y), width, height,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, text,
                ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=textcolor, wrap=True)

    def draw_arrow(start, end, color='black', style='->', connectionstyle='arc3,rad=0'):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle=style, lw=2, color=color,
                                  connectionstyle=connectionstyle))

    # Title
    ax.text(9, 13.5, '5-Layer Neural Network Training Workflow',
            fontsize=16, fontweight='bold', ha='center')

    # ==================== INITIALIZATION SECTION ====================
    ax.text(2.5, 12.8, 'INITIALIZATION', fontsize=11, fontweight='bold',
            ha='center', color=colors['process'])

    draw_box(0.5, 11.5, 4, 1, 'Initialize Parameters\n(He/Xavier/Random)', colors['process'])
    draw_box(5, 11.5, 4, 1, 'Initialize Optimizer\n(Adam/Momentum/RMSprop)', colors['process'])

    draw_arrow((4.5, 12), (5, 12))

    # ==================== TRAINING LOOP ====================
    ax.text(9, 10.8, 'TRAINING LOOP (for each epoch)', fontsize=11, fontweight='bold',
            ha='center', color=colors['decision'])

    # Learning rate decay
    draw_box(0.5, 9.5, 3.5, 0.9, 'Learning Rate Decay\n(optional)', colors['optional'], textcolor='black')

    # Mini-batch creation
    draw_box(4.5, 9.5, 4, 0.9, 'Create Mini-Batches\n(shuffle & partition)', colors['input'])

    draw_arrow((4, 9.95), (4.5, 9.95))

    # ==================== FORWARD PROPAGATION ====================
    ax.text(9, 8.7, 'FORWARD PROPAGATION', fontsize=11, fontweight='bold',
            ha='center', color=colors['forward'])

    # Forward prop steps
    y_forward = 7.3
    box_h = 1.2

    draw_box(0.3, y_forward, 3.2, box_h,
             'Layer 1-4:\nZ = W·A + b', colors['forward'])

    draw_box(3.8, y_forward, 3.2, box_h,
             'BatchNorm\n(optional)\nZ = γ·Z_norm + β', colors['optional'], textcolor='black')

    draw_box(7.3, y_forward, 3.2, box_h,
             'ReLU\nActivation\nA = max(0, Z)', colors['forward'])

    draw_box(10.8, y_forward, 3.2, box_h,
             'Dropout\n(optional)\nA = A * mask / p', colors['optional'], textcolor='black')

    draw_box(14.3, y_forward, 3.2, box_h,
             'Layer 5:\nZ = W·A + b\nA = σ(Z)', colors['output'])

    # Forward arrows
    draw_arrow((3.5, y_forward + box_h/2), (3.8, y_forward + box_h/2))
    draw_arrow((7, y_forward + box_h/2), (7.3, y_forward + box_h/2))
    draw_arrow((10.5, y_forward + box_h/2), (10.8, y_forward + box_h/2))
    draw_arrow((14, y_forward + box_h/2), (14.3, y_forward + box_h/2))

    # ==================== LOSS COMPUTATION ====================
    ax.text(9, 6.3, 'LOSS COMPUTATION', fontsize=11, fontweight='bold',
            ha='center', color=colors['loss'])

    draw_box(5, 5, 4, 1.1,
             'Cross-Entropy Loss\nL = -1/m Σ[y·log(ŷ) + (1-y)·log(1-ŷ)]', colors['loss'])

    draw_box(10, 5, 4, 1.1,
             'L2 Regularization\n+ λ/2m Σ||W||²', colors['optional'], textcolor='black')

    draw_arrow((15.9, y_forward), (15.9, 6.1), connectionstyle='arc3,rad=0')
    draw_arrow((15.9, 6.1), (14, 5.55), connectionstyle='arc3,rad=0')
    draw_arrow((9, 5.55), (10, 5.55))

    # ==================== BACKWARD PROPAGATION ====================
    ax.text(9, 4.1, 'BACKWARD PROPAGATION', fontsize=11, fontweight='bold',
            ha='center', color=colors['backward'])

    y_back = 2.5
    box_h = 1.4

    draw_box(0.3, y_back, 3.5, box_h,
             'Output Layer:\ndZ⁵ = A⁵ - Y\ndW⁵ = 1/m·dZ⁵·A⁴ᵀ', colors['backward'])

    draw_box(4.1, y_back, 3.5, box_h,
             'Hidden Layers:\ndA = Wᵀ·dZ\ndZ = dA * g\'(Z)', colors['backward'])

    draw_box(7.9, y_back, 3.5, box_h,
             'BatchNorm Backward\n(if enabled)\ndZ = BN_backward(dZ)', colors['optional'], textcolor='black')

    draw_box(11.7, y_back, 3.5, box_h,
             'Compute Gradients:\ndW = 1/m·dZ·Aᵀ + λ/m·W\ndb = 1/m·Σ dZ', colors['backward'])

    # Backward arrows (reversed)
    draw_arrow((4.1, y_back + box_h/2), (3.8, y_back + box_h/2))
    draw_arrow((7.9, y_back + box_h/2), (7.6, y_back + box_h/2))
    draw_arrow((11.7, y_back + box_h/2), (11.4, y_back + box_h/2))

    # Arrow from loss to backward
    draw_arrow((7, 5), (7, 3.9), connectionstyle='arc3,rad=0')

    # ==================== UPDATE PARAMETERS ====================
    ax.text(9, 1.6, 'UPDATE PARAMETERS', fontsize=11, fontweight='bold',
            ha='center', color=colors['update'])

    draw_box(1, 0.3, 3.8, 1.1,
             'GD: W = W - α·dW', colors['update'])

    draw_box(5.1, 0.3, 3.8, 1.1,
             'Momentum:\nv = β·v + (1-β)·dW\nW = W - α·v', colors['update'])

    draw_box(9.2, 0.3, 3.8, 1.1,
             'RMSprop:\ns = β₂·s + (1-β₂)·dW²\nW = W - α·dW/√(s+ε)', colors['update'])

    draw_box(13.3, 0.3, 4.2, 1.1,
             'Adam:\nv,s + bias correction\nW = W - α·v̂/√(ŝ+ε)', colors['update'])

    # Arrow from gradients to update
    draw_arrow((13.45, y_back), (9, 1.4), connectionstyle='arc3,rad=-0.2')

    # Loop back arrow
    ax.annotate('', xy=(0.5, 9.95), xytext=(0.3, 1.4),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=colors['decision'],
                              connectionstyle='arc3,rad=-0.3'))
    ax.text(0.1, 5.5, 'Next\nEpoch', fontsize=9, fontweight='bold',
            ha='center', color=colors['decision'], rotation=90)

    # Mini-batch loop arrow
    draw_arrow((8.5, 9.5), (8.5, 8.7), color=colors['input'])

    plt.tight_layout()
    return fig, ax


def draw_inference_workflow(figsize=(16, 6)):
    """
    Draw the inference (prediction) workflow.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis('off')

    colors = {
        'input': '#3498db',
        'forward': '#2ecc71',
        'output': '#1abc9c',
        'decision': '#e74c3c'
    }

    def draw_box(x, y, width, height, text, color, fontsize=9):
        rect = FancyBboxPatch((x, y), width, height,
                              boxstyle="round,pad=0.02,rounding_size=0.1",
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, text,
                ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color='white', wrap=True)

    def draw_arrow(start, end):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Title
    ax.text(8, 5.5, 'Inference (Prediction) Workflow', fontsize=14, fontweight='bold', ha='center')

    # Workflow boxes
    y = 2.5
    h = 1.5

    draw_box(0.5, y, 2.5, h, 'Input\nX', colors['input'])
    draw_box(3.5, y, 2.5, h, 'Layer 1-4\nZ = W·A + b\nA = ReLU(Z)', colors['forward'])
    draw_box(6.5, y, 2.5, h, 'BatchNorm\n(use running\nmean/var)', colors['forward'])
    draw_box(9.5, y, 2.5, h, 'Layer 5\nZ = W·A + b\nA = σ(Z)', colors['forward'])
    draw_box(12.5, y, 2.5, h, 'Threshold\nŷ = A > 0.5', colors['decision'])

    # Arrows
    draw_arrow((3, y + h/2), (3.5, y + h/2))
    draw_arrow((6, y + h/2), (6.5, y + h/2))
    draw_arrow((9, y + h/2), (9.5, y + h/2))
    draw_arrow((12, y + h/2), (12.5, y + h/2))

    # Notes
    ax.text(8, 1.5, 'Note: No dropout during inference (training=False)',
            fontsize=10, ha='center', style='italic')
    ax.text(8, 1.0, 'BatchNorm uses running statistics instead of batch statistics',
            fontsize=10, ha='center', style='italic')

    plt.tight_layout()
    return fig, ax


def draw_complete_pipeline(figsize=(18, 20)):
    """
    Draw the complete ML pipeline including data preprocessing, training, and evaluation.
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 20)
    ax.axis('off')

    colors = {
        'data': '#3498db',
        'preprocess': '#9b59b6',
        'model': '#2ecc71',
        'train': '#f39c12',
        'eval': '#e74c3c',
        'output': '#1abc9c',
        'arrow': '#34495e'
    }

    def draw_box(x, y, width, height, text, color, fontsize=9, textcolor='white'):
        rect = FancyBboxPatch((x, y), width, height,
                              boxstyle="round,pad=0.02,rounding_size=0.15",
                              facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, text,
                ha='center', va='center', fontsize=fontsize,
                fontweight='bold', color=textcolor, wrap=True)

    def draw_arrow(start, end, color='black'):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', lw=2, color=color))

    # Title
    ax.text(9, 19.5, 'Complete Neural Network Pipeline', fontsize=16, fontweight='bold', ha='center')

    # ==================== PHASE 1: DATA ====================
    ax.text(9, 18.8, '1. DATA PREPARATION', fontsize=12, fontweight='bold', ha='center', color=colors['data'])

    draw_box(1, 17.5, 3, 1, 'Load Data\n(X, Y)', colors['data'])
    draw_box(5, 17.5, 3, 1, 'Train/Test\nSplit', colors['data'])
    draw_box(9, 17.5, 4, 1, 'Normalization\n(Z-score/MinMax/L2)', colors['preprocess'])
    draw_box(14, 17.5, 3, 1, 'Normalized\nX_train, X_test', colors['data'])

    draw_arrow((4, 18), (5, 18))
    draw_arrow((8, 18), (9, 18))
    draw_arrow((13, 18), (14, 18))

    # ==================== PHASE 2: MODEL SETUP ====================
    ax.text(9, 16.5, '2. MODEL INITIALIZATION', fontsize=12, fontweight='bold', ha='center', color=colors['model'])

    draw_box(1, 15, 3.5, 1, 'Define Architecture\nlayer_dims=[n_x,...,n_5]', colors['model'])
    draw_box(5, 15, 3.5, 1, 'Weight Init\n(He/Xavier)', colors['model'])
    draw_box(9, 15, 3.5, 1, 'Optimizer Setup\n(Adam/Momentum)', colors['model'])
    draw_box(13, 15, 4, 1, 'Hyperparameters\nλ, keep_prob, α', colors['model'])

    draw_arrow((4.5, 15.5), (5, 15.5))
    draw_arrow((8.5, 15.5), (9, 15.5))
    draw_arrow((12.5, 15.5), (13, 15.5))

    # ==================== PHASE 3: TRAINING ====================
    ax.text(9, 13.8, '3. TRAINING LOOP', fontsize=12, fontweight='bold', ha='center', color=colors['train'])

    # Epoch loop
    draw_box(0.5, 11.5, 16, 2, '', colors['train'], textcolor='black')
    ax.text(8.5, 13.2, 'For each epoch:', fontsize=10, fontweight='bold', ha='center')

    draw_box(1, 11.8, 2.5, 1, 'LR Decay', colors['train'])
    draw_box(4, 11.8, 3, 1, 'Mini-batch\nCreation', colors['train'])
    draw_box(7.5, 11.8, 2.5, 1, 'Forward\nProp', colors['train'])
    draw_box(10.5, 11.8, 2.5, 1, 'Compute\nLoss', colors['train'])
    draw_box(13.5, 11.8, 2.5, 1, 'Backward\nProp', colors['train'])

    draw_arrow((3.5, 12.3), (4, 12.3))
    draw_arrow((7, 12.3), (7.5, 12.3))
    draw_arrow((10, 12.3), (10.5, 12.3))
    draw_arrow((13, 12.3), (13.5, 12.3))

    # Update parameters
    draw_box(6, 9.8, 6, 1.2, 'Update Parameters\nW = W - α·(dW with optimizer)', colors['train'])
    draw_arrow((9, 11.5), (9, 11))

    # ==================== PHASE 4: FORWARD PROPAGATION DETAIL ====================
    ax.text(9, 8.8, '4. FORWARD PROPAGATION (Detail)', fontsize=12, fontweight='bold', ha='center', color=colors['model'])

    y_fp = 7
    draw_box(0.5, y_fp, 2.8, 1.3, 'Linear\nZ = W·A + b', colors['model'])
    draw_box(3.8, y_fp, 2.8, 1.3, 'BatchNorm\nZ = γ·Ẑ + β', colors['preprocess'])
    draw_box(7.1, y_fp, 2.8, 1.3, 'Activation\nA = ReLU(Z)', colors['model'])
    draw_box(10.4, y_fp, 2.8, 1.3, 'Dropout\nA = A·D/p', colors['preprocess'])
    draw_box(13.7, y_fp, 3.3, 1.3, 'Output\nA⁵ = σ(Z⁵)', colors['output'])

    draw_arrow((3.3, y_fp + 0.65), (3.8, y_fp + 0.65))
    draw_arrow((6.6, y_fp + 0.65), (7.1, y_fp + 0.65))
    draw_arrow((9.9, y_fp + 0.65), (10.4, y_fp + 0.65))
    draw_arrow((13.2, y_fp + 0.65), (13.7, y_fp + 0.65))

    ax.text(9, 6.5, '(Repeat for Layers 1-4, then Output Layer)', fontsize=9, ha='center', style='italic')

    # ==================== PHASE 5: BACKWARD PROPAGATION DETAIL ====================
    ax.text(9, 5.8, '5. BACKWARD PROPAGATION (Detail)', fontsize=12, fontweight='bold', ha='center', color=colors['eval'])

    y_bp = 4
    draw_box(0.5, y_bp, 3, 1.3, 'Output Grad\ndZ⁵ = A⁵ - Y', colors['eval'])
    draw_box(4, y_bp, 3.2, 1.3, 'Activation Grad\ndZ = dA·g\'(Z)', colors['eval'])
    draw_box(7.7, y_bp, 3.2, 1.3, 'BatchNorm Grad\ndZ = BN_back(dZ)', colors['preprocess'])
    draw_box(11.4, y_bp, 3.2, 1.3, 'Weight Grad\ndW = 1/m·dZ·Aᵀ', colors['eval'])
    draw_box(15.1, y_bp, 2.4, 1.3, 'Propagate\ndA = Wᵀ·dZ', colors['eval'])

    draw_arrow((4, y_bp + 0.65), (3.5, y_bp + 0.65))
    draw_arrow((7.7, y_bp + 0.65), (7.2, y_bp + 0.65))
    draw_arrow((11.4, y_bp + 0.65), (10.9, y_bp + 0.65))
    draw_arrow((15.1, y_bp + 0.65), (14.6, y_bp + 0.65))

    # ==================== PHASE 6: EVALUATION ====================
    ax.text(9, 3, '6. EVALUATION & PREDICTION', fontsize=12, fontweight='bold', ha='center', color=colors['output'])

    draw_box(1.5, 1.5, 3.5, 1, 'Forward Pass\n(training=False)', colors['output'])
    draw_box(5.5, 1.5, 3.5, 1, 'Threshold\nŷ = (A⁵ > 0.5)', colors['output'])
    draw_box(9.5, 1.5, 3.5, 1, 'Accuracy\nacc = mean(ŷ == Y)', colors['output'])
    draw_box(13.5, 1.5, 3.5, 1, 'Predictions\nŷ_test', colors['output'])

    draw_arrow((5, 2), (5.5, 2))
    draw_arrow((9, 2), (9.5, 2))
    draw_arrow((13, 2), (13.5, 2))

    # Legend
    legend_y = 0.3
    legend_items = [
        ('Data/Input', colors['data']),
        ('Preprocessing', colors['preprocess']),
        ('Model/Forward', colors['model']),
        ('Training', colors['train']),
        ('Backward/Eval', colors['eval']),
        ('Output', colors['output'])
    ]
    for i, (label, color) in enumerate(legend_items):
        x = 1 + i * 2.8
        rect = FancyBboxPatch((x, legend_y), 0.4, 0.4, boxstyle="round,pad=0.02",
                              facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(x + 0.5, legend_y + 0.2, label, fontsize=8, va='center')

    plt.tight_layout()
    return fig, ax


if __name__ == "__main__":
    # Generate training workflow diagram
    fig1, ax1 = draw_training_workflow()
    fig1.savefig('workflow_training.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved: workflow_training.png")

    # Generate inference workflow diagram
    fig2, ax2 = draw_inference_workflow()
    fig2.savefig('workflow_inference.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved: workflow_inference.png")

    # Generate complete pipeline diagram
    fig3, ax3 = draw_complete_pipeline()
    fig3.savefig('workflow_complete_pipeline.png', dpi=150, bbox_inches='tight', facecolor='white')
    print("Saved: workflow_complete_pipeline.png")

    plt.show()
