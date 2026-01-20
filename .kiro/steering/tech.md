---
inclusion: always
---

# Technology Stack & Development Guidelines

## Core Technology Stack
- **Python 3.x**: Primary language - use only NumPy for core ML algorithm implementations
- **NumPy**: Mathematical operations, array manipulations, matrix operations
- **Pandas**: Data loading, manipulation, and CSV handling
- **Matplotlib**: All visualizations and plotting (save as PNG in `images/` folders)
- **Jupyter Notebooks**: Interactive development and educational content

## AWS Services Integration
- **SageMaker**: Model training, deployment, and managed ML workflows
- **AWS Glue**: Data preparation and ETL processes
- **Comprehend**: NLP and text analysis services
- **Rekognition**: Computer vision and image analysis
- **Bedrock**: Generative AI and foundation model access

## Implementation Philosophy
- **Educational First**: Code clarity over performance optimization
- **From-Scratch Rule**: Implement core algorithms using only NumPy (no sklearn, tensorflow, etc.)
- **Mathematical Rigor**: Include LaTeX equations in markdown documentation
- **Visualization Required**: Every algorithm must generate plots saved to `images/` folder
- **Progressive Learning**: Start with univariate, progress to multivariate implementations

## Code Architecture Patterns

### Function Structure (Required)
```python
def algorithm_name(X, y, learning_rate=0.01, num_iterations=1000):
    # 1. Initialize parameters
    # 2. Training loop with cost computation
    # 3. Return trained parameters and cost history
```

### Training Loop Pattern
```python
for i in range(num_iterations):
    # Forward pass
    # Cost computation  
    # Gradient computation
    # Parameter updates
    if i % 100 == 0:  # Progress printing every 100 iterations
        print(f"Cost after iteration {i}: {cost}")
```

## Code Style Requirements
- **Variable Names**: Use mathematical terminology (`theta`, `learning_rate`, `num_iterations`)
- **Error Handling**: Implement epsilon clipping for numerical stability
- **Function Separation**: Separate concerns (cost functions, gradients, predictions)
- **Documentation**: Include docstrings with mathematical formulas
- **Progress Tracking**: Print training progress every 100 iterations

## File Generation Rules
- **Data Scripts**: Generate synthetic datasets, save to `Data/` folder as CSV
- **Plotting Scripts**: Create visualizations, save PNG files to appropriate `images/` folder
- **Notebooks**: Interactive implementations with markdown explanations
- **Documentation**: Theory files with LaTeX math formatting

## Development Workflow
```bash
# Install dependencies
pip install -r requirements.txt

# Generate data first
python Scripts/{dataset_name}_data.py

# Create visualizations
python Scripts/{algorithm_name}_plot.py

# Run Jupyter for interactive development
jupyter notebook
```

## Quality Standards
- All implementations must work with generated datasets in `Data/` folder
- Every algorithm requires cost function visualization
- Mathematical formulas must be documented in markdown
- Code must be executable without external ML libraries (except NumPy/Pandas/Matplotlib)