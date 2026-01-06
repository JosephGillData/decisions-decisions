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

## Experiments

This project includes two complete experiments demonstrating model evaluation:

### 1. Insurance Dataset (Regression)

Predicts medical insurance charges based on patient demographics (age, BMI, smoker status, etc.).

**Metrics used**: MAE, RMSE, R²

### 2. Mobile Phones Dataset (Classification)

Predicts phone price category (0-3) based on hardware specifications (RAM, battery, screen size, etc.).

**Metrics used**: Accuracy, Precision, Recall, F1 (macro-averaged)

### Train vs Test Comparison (Overfitting Analysis)

Both experiments compute metrics on **training AND test sets** to assess overfitting:

| What to look for | Interpretation |
|-----------------|----------------|
| Train >> Test | Model is overfitting (memorizing training data) |
| Train ≈ Test | Good generalization |
| Both low | Model is underfitting (too simple) |

**Why this matters:**

A model that performs well on training data but poorly on test data has **high variance** - it's learned noise specific to the training set rather than true patterns.

- **Decision Trees**: High variance, prone to overfitting (especially deep trees)
- **Random Forests**: Reduce variance through averaging diverse trees
- **Gradient Boosting**: Can overfit with too many trees, but learning rate provides regularization

### Running the Experiments

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Run all experiments
python experiments/run_experiments.py
```

**Outputs saved to `outputs/`:**
- `insurance_metrics.csv` - Regression metrics comparison
- `mobile_phones_metrics.csv` - Classification metrics comparison
- `mobile_phones_confusion_matrix.csv` - Confusion matrix for classification

## Project Structure

```
decisions-decisions/
├── src/                        # Core implementations
│   ├── decision_tree.py        # Decision tree algorithm
│   ├── node.py                 # Tree node data structure
│   ├── random_forest.py        # Random forest ensemble
│   ├── gradient_boosting.py    # Gradient boosting ensemble
│   ├── decision_tree_graph.py  # Tree visualization
│   ├── metrics.py              # Evaluation metrics (MAE, RMSE, R², F1, etc.)
│   └── data_utils.py           # Dataset loading utilities
├── experiments/
│   └── run_experiments.py      # Train/test evaluation script
├── notebooks/
│   └── demo_decision_tree.ipynb
├── outputs/                    # Experiment results (CSV files)
├── tests/                      # Unit tests
├── docs/                       # Algorithm documentation
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

### 2. Run the Experiments

```bash
python experiments/run_experiments.py
```

This trains Decision Tree, Random Forest, and Gradient Boosting on both datasets and outputs metrics to `outputs/`.

### 3. Run the Interactive Demo

```bash
jupyter notebook notebooks/demo_decision_tree.ipynb
```

### 4. Use Directly in Python

```python
from src.decision_tree import DecisionTree
from src.data_utils import load_insurance_data
from src.metrics import compute_regression_metrics

# Load data (downloads from Kaggle automatically)
X_train, X_test, y_train, y_test = load_insurance_data()

# Train a decision tree
tree = DecisionTree(
    criterion='variance',  # 'gini' for classification
    max_depth=5,
    min_samples_split=10
)
model = tree.train(X_train, y_train)

# Make predictions
train_pred = tree.predict(model, X_train)
test_pred = tree.predict(model, X_test)

# Evaluate
train_metrics = compute_regression_metrics(y_train, train_pred)
test_metrics = compute_regression_metrics(y_test, test_pred)

print(f"Train R²: {train_metrics['r2']:.4f}")
print(f"Test R²: {test_metrics['r2']:.4f}")
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

### Stopping Conditions (Regularization)

The tree stops growing when:
- `max_depth` is reached (primary regularization parameter)
- Node has fewer than `min_samples_split` samples
- Split would create leaves smaller than `min_samples_leaf`
- No valid split improves impurity

### Random Forest

Random Forest reduces overfitting through:
1. **Bootstrap sampling**: Each tree trains on a random sample (with replacement)
2. **Aggregation**: Final prediction is the mean (regression) or mode (classification)

```python
rf = RandomForest(
    n_estimators=100,
    criterion='variance',  # or 'gini' for classification
    max_depth=10
)
model = rf.train(X_train, y_train)
predictions = rf.predict(model, X_test)
```

### Gradient Boosting

Gradient Boosting builds trees sequentially, each correcting the errors of the previous ensemble:

1. Start with `F_0 = mean(y)`
2. For each round: fit a tree to residuals `y - F_{t-1}(x)`
3. Update: `F_t = F_{t-1} + learning_rate * tree_t`

```python
gb = GradientBoosting(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3  # Shallow trees work well for boosting
)
model = gb.train(X_train, y_train)
predictions = gb.predict(model, X_test)
```

## Datasets

The project uses two Kaggle datasets (downloaded automatically via `kagglehub`):

1. **Insurance** (Regression)
   - Target: `charges` (medical insurance cost)
   - Features: age, sex, bmi, children, smoker, region

2. **Mobile Phones** (Classification)
   - Target: `price_range` (0-3, four price categories)
   - Features: battery_power, clock_speed, ram, etc.

## Evaluation Metrics

### Regression (Insurance)

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MAE** | mean(\|y - ŷ\|) | Average prediction error in original units |
| **RMSE** | √mean((y - ŷ)²) | Penalizes large errors more heavily |
| **R²** | 1 - SS_res/SS_tot | Proportion of variance explained (0-1) |

### Classification (Mobile Phones)

| Metric | Description |
|--------|-------------|
| **Accuracy** | % of correct predictions |
| **Precision** | Of predicted positives, how many are correct |
| **Recall** | Of actual positives, how many did we find |
| **F1** | Harmonic mean of precision and recall |

## Requirements

- Python 3.8+
- NumPy
- Pandas
- Matplotlib
- NetworkX
- kagglehub (for downloading datasets)
- pytest (for tests)

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

## About

This is an **educational / portfolio project** demonstrating understanding of ML fundamentals.
The code prioritizes **clarity over performance** - the goal is to learn, not to compete with sklearn.

**Key learning outcomes:**
- How decision trees recursively partition data
- Why impurity measures (Gini, variance) guide splitting decisions
- How ensemble methods reduce overfitting
- How to evaluate models properly (train vs test comparison)

Built by Joseph as a portfolio project demonstrating ML engineering skills.
