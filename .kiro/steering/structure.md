# Project Structure

## Directory Organization

```
StandFord Machine Learning/
├── Data/                           # Datasets for training and testing
│   ├── house_prices.csv           # Generated housing data
│   └── tumor_data.csv             # Medical classification data
├── Scripts/                       # Standalone Python scripts
│   ├── *_data.py                  # Data generation scripts
│   ├── *_plot.py                  # Visualization scripts
│   └── predict_*.py               # Prediction and analysis scripts
├── Supervised Learning/           # Supervised learning algorithms
│   ├── Linear Regression/
│   │   ├── Notebooks/             # Jupyter notebooks for hands-on learning
│   │   └── Notes/                 # Markdown documentation with theory
│   ├── Logistic Regression/
│   │   ├── Notebooks/             # Implementation notebooks
│   │   └── Notes/                 # Theory and documentation
│   └── images/                    # Generated plots and visualizations
├── Unsupervised Learning/         # Unsupervised learning algorithms
│   └── images/                    # Visualization outputs
├── Re-enforcement Learning/       # Reinforcement learning (future)
└── requirements.txt               # Python dependencies
```

## File Naming Conventions

### Scripts
- Data generation: `{dataset_name}_data.py`
- Plotting utilities: `{algorithm_name}_plot.py`
- Prediction scripts: `predict_{domain}.py`

### Notebooks
- Algorithm implementations: `{Algorithm_name}.ipynb`
- Exercises: `{algorithm_name}_exercise.ipynb`
- Multi-variable versions: `{algorithm_name}_multiple_variable.ipynb`

### Documentation
- Theory files: `{Algorithm_Name}.md` (title case)
- Implementation notes: `{algorithm_name}.md` (lowercase)

### Images
- Generated plots: `{descriptive_name}.png`
- Store in `images/` folder within each learning category

## Content Organization Patterns

### Algorithm Folders
Each algorithm should have:
- `Notebooks/`: Interactive Jupyter implementations
- `Notes/`: Theoretical documentation with math formulas
- Supporting scripts for visualization and data generation

### Documentation Structure
1. **Overview**: Brief algorithm description
2. **Mathematical Concepts**: LaTeX-formatted equations
3. **Implementation Steps**: Code walkthrough
4. **Evaluation Metrics**: Performance measurement
5. **Visualizations**: Charts and plots explanation

### Code Organization
- Separate data generation from model implementation
- Keep visualization code in dedicated plotting scripts
- Use descriptive function names that match mathematical concepts
- Group related functions together (cost calculation, gradient computation, parameter updates)