# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AWS Machine Learning Bootcamp repository - a 12-week journey learning ML on AWS. The focus is implementing core ML algorithms **from scratch using only NumPy** (no sklearn, tensorflow for core implementations), with educational clarity prioritized over performance.

## Common Commands

```bash
# Install dependencies
pip install -r "StandFord Machine Learning/requirements.txt"

# Run the cat classifier (requires datasets/train_catvnoncat.h5 and test_catvnoncat.h5)
python cat_classifier.py

# Run data generation scripts
python Scripts/{dataset_name}_data.py

# Run visualization scripts
python Scripts/{algorithm_name}_plot.py

# Start documentation site
cd docs-site && python3 -m http.server 8080
# Or use: ./docs-site/deploy.sh
```

## Architecture

### Core Neural Network Implementation

**[five_layer_nn.py](five_layer_nn.py)** - The main `FiveLayerNN` class implementing a 5-layer neural network from scratch:
- Configurable layer dimensions via `layer_dims` list `[n_x, n1, n2, n3, n4, n5]`
- Weight initialization: `zeros`, `random`, `xavier`, `he`
- Optimizers: `gd`, `momentum`, `rmsprop`, `adam`
- Regularization: L2 (`lambd`) and Dropout (`keep_prob`)
- Batch normalization support (`use_batch_norm`)
- Learning rate decay (`decay_rate`, `time_interval`)
- Static normalization methods: `normalize_minmax`, `normalize_zscore`, `normalize_mean`, `normalize_l2`

**[cat_classifier.py](cat_classifier.py)** - Uses `FiveLayerNN` for cat vs non-cat binary classification:
- Loads data from HDF5 files in `datasets/` folder
- Provides preprocessing, evaluation metrics, and visualization functions
- Demonstrates the full training pipeline with the neural network class

### Implementation Philosophy

- **Educational First**: Prioritize code clarity and mathematical documentation over optimization
- **From-Scratch Rule**: Implement algorithms using only NumPy for core logic
- **Mathematical Rigor**: Include LaTeX equations in markdown documentation
- **Visualization Required**: Every algorithm must generate plots (saved to `images/` folders)

### Code Patterns

Functions should follow this structure:
```python
def algorithm_name(X, y, learning_rate=0.01, num_iterations=1000):
    # 1. Initialize parameters
    # 2. Training loop with cost computation
    # 3. Return trained parameters and cost history
```

Training loops should print progress every 100 iterations.

Use mathematical variable names: `theta`, `learning_rate`, `num_iterations`, `lambd` (not `lambda`).

### Directory Structure

- `StandFord Machine Learning/` - Main learning content organized by topic
  - `Supervised Learning/` - Linear and Logistic Regression
  - `Neural Networks/` - Neural network implementations and notes
  - `Unsupervised Learning/` - Clustering algorithms
  - `Scripts/` - Data generation and visualization scripts
- `Data/` - Generated CSV datasets for training/testing
- `docs-site/` - Static documentation site (HTML/CSS/JS with Markdown rendering via Marked.js)
- `Scripts/` - Training scripts at project root

### File Naming Conventions

- Data generation: `{dataset_name}_data.py`
- Plotting: `{algorithm_name}_plot.py`
- Notebooks: `{Algorithm_name}.ipynb`
- Theory docs: `{Algorithm_Name}.md` (title case)
