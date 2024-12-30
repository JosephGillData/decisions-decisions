# Decisions Decisions

The decision tree, random forest and XgBoost algorithms are all related, each one building upon the complexity of the other.

For each algorithm, we will:
- Understand the theory
- Understand the mathematics
- Build the algorithm from scratch and
  - Apply to a regression problem
  - Apply to a categorical problem

Each of the algorithms will be applied to the same datasets so we can compare the performance of the models.
 
## Decision Trees

### Theory

Decision trees are the foundation of all tree based algorithms, such as random forest and XgBoost. Although decision trees themselves aren't used in ML applications, understanding how they work is key to understanding more performant algorithms.

A machine learning problem is set up so that, at least in the training dataset, we have a set of features (X) and a label ($y_{\text{true}}$). Let's assume this label is categorical. Decision trees work by analysing X at each stage of the decision tree (node) to identify the optimal way of splitting a subset of the data into two subsets that most closely align the predicted labels ($y_{\text{pred}}$) with $y_{\text{true}}$. This process is repeated until some regularisation parameters kick in.

### Extra Points

- Decision trees are a white box model, meaning they are easy to interpret.
- Finding the optimal tree is extremly computationally expensive. Therefore, decision trees are greedy algorithms, meaning they find the optimal solution at each stage of the process, even though this doesn't guanarantee a global optimum.
- Decision trees are a nonparametric model, meaning that the number of parameters of the model (degrees of freedom) aren't determined prior to training. It also doesn't make any assumptions about the structure of the data. In comparison, the linear regression model assumes that the data is linear and the number of parameters. This makes regularisation very important to ensure the model doesn't overfit the training data.

### Hyperparameters

-  Criterion: The function to measure the quality of a split.
- Max depth: The maximum depth of the tree.
- Min samples split: The minimum number of samples required to split an internal node.
- Min samples leaf: The minimum number of samples required to be at a leaf node.

### References
#### Writing

- https://medium.com/@enozeren/building-a-decision-tree-from-scratch-324b9a5ed836
- https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
- Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow (Aurélien Géron) (Z-Library).pdf
- https://www.youtube.com/watch?v=ZVR2Way4nwQ&list=PLM8wYQRetTxAl5FpMIJCcJbfZjSB0IeC_


#### Code

- https://github.com/enesozeren/machine_learning_from_scratch/blob/main/decision_trees/decision_tree.py
- https://github.com/Suji04/ML_from_Scratch/blob/master/decision%20tree%20classification.ipynb

## TODO

- Mathematics section
- Implement for categorical problem