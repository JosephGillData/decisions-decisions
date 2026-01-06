# Decision Tree from Scratch

A pure Python implementation of decision trees, random forests, and gradient boosting **without sklearn**. Built to demonstrate deep understanding of how these algorithms work internally.

## Why Build This?

Most ML courses teach you to call `sklearn.DecisionTreeClassifier()` and move on. But understanding *how* decision trees work is crucial for:

- **Debugging models**: Why is my tree overfitting? What features is it actually using?
- **Explaining predictions**: How do I explain this model's decision to a stakeholder?
- **Building intuition**: Understanding trees makes gradient boosting (XGBoost, LightGBM) much clearer

This project implements everything from first principles:
- **Impurity measures** (Gini, variance) - how we measure split quality
- **Greedy splitting** - how we choose features and thresholds
- **Recursive tree building** - how the tree grows node by node
- **Ensemble methods** - how Random Forest and Gradient Boosting improve on single trees

## Algorithms Implemented

| Algorithm | Description | Key Insight |
|-----------|-------------|-------------|
| **Decision Tree** | Recursively splits data to minimize impurity | Greedy local optimization at each node |
| **Random Forest** | Ensemble of trees on bootstrap samples | Averaging reduces variance |
| **Gradient Boosting** | Sequential trees that correct previous errors | Following the gradient in function space |

## Project Structure

```
decisions-decisions/
├── src/                    # Core implementations
│   ├── decision_tree.py    # Decision tree algorithm
│   ├── node.py             # Tree node data structure
│   ├── random_forest.py    # Random forest ensemble
│   ├── gradient_boosting.py # Gradient boosting ensemble
│   ├── decision_tree_graph.py  # Tree visualization
│   └── data_utils.py       # Dataset loading utilities
├── notebooks/
│   └── demo_decision_tree.ipynb  # Interactive demo
├── tests/                  # Unit tests
├── docs/                   # Algorithm documentation
└── requirements.txt
```

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/yourusername/decisions-decisions.git
cd decisions-decisions

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Demo

```bash
jupyter notebook notebooks/demo_decision_tree.ipynb
```

Or use directly in Python:

```python
from src.decision_tree import DecisionTree
from src.data_utils import load_insurance_data

# Load data (downloads from Kaggle automatically)
X_train, X_test, y_train, y_test = load_insurance_data()

# Train a decision tree
tree = DecisionTree(
    criterion='variance',
    max_depth=5,
    min_samples_split=10
)
model = tree.train(X_train, y_train)

# Make predictions
predictions = tree.predict(model, X_test)
```

### 3. Visualize the Tree

```python
from src.decision_tree_graph import DecisionTreeVisualizer

viz = DecisionTreeVisualizer()
viz.visualize(model, feature_dtypes=X_train.dtypes)
```

## How It Works

### Decision Tree Algorithm

The core algorithm works by recursively finding the best way to split the data:

1. **At each node**, evaluate every feature and every possible threshold
2. **Score each split** using weighted impurity (Gini for classification, variance for regression)
3. **Choose the split** that minimizes impurity
4. **Recurse** on left and right children until stopping conditions are met

```
                    [smoker=yes?]
                    /            \
                 Yes              No
                  |                |
            [bmi <= 30?]      [age <= 43?]
            /        \        /         \
         ...        ...     ...        ...
```

### Stopping Conditions

The tree stops growing when:
- `max_depth` is reached (prevents overfitting)
- Node has fewer than `min_samples_split` samples
- Split would create leaves smaller than `min_samples_leaf`
- No valid split improves impurity

### Random Forest

Random Forest builds diversity through:
1. **Bootstrap sampling**: Each tree trains on a random sample (with replacement)
2. **Averaging**: Final prediction is the mean of all tree predictions

```python
rf = RandomForest(n_estimators=100, max_depth=10)
model = rf.train(X_train, y_train)
predictions = rf.predict(model, X_test)
```

### Gradient Boosting

Gradient Boosting builds trees sequentially:
1. Start with `F_0 = mean(y)`
2. For each round: fit a tree to residuals `y - F_{t-1}(x)`
3. Update: `F_t = F_{t-1} + learning_rate * tree_t`

```python
gb = GradientBoosting(n_estimators=100, learning_rate=0.1, max_depth=3)
model = gb.train(X_train, y_train)
predictions = gb.predict(model, X_test)
```

## Datasets

The project uses two Kaggle datasets (downloaded automatically):

1. **Insurance** (Regression): Predict medical insurance charges
2. **Mobile Phones** (Classification): Predict phone price range

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib (for visualization)
- NetworkX (for tree visualization)
- Graphviz (system installation for tree layout)

## Documentation

See `docs/tree-based-algorithms.md` for detailed mathematical explanations of:
- Variance and variance reduction
- Gini impurity
- Gradient descent in function space
- XGBoost's second-order approximation

## Running Tests

```bash
pytest tests/
```

## License

MIT License - Use freely for learning and projects.

## About

This is an educational project built to demonstrate understanding of ML fundamentals.
The code prioritizes **clarity over performance** - the goal is to learn, not to compete with sklearn.

Built by Joseph as a portfolio project demonstrating ML engineering skills.
