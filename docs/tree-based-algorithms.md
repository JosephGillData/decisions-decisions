# Tree Based Algorithms

Render -> Ctlr + Shift + V

The decision tree, random forest and XGBoost algorithms are all related, each one building upon the complexity of the other.

For each algorithm, we will:

- Understand the theory
- Understand the mathematics
- Build the algorithm from scratch and apply to a regression problem

Each of the algorithms will be applied to the same datasets so we can compare the performance of the models.

# Decision Trees

## Theory

Decision trees are the foundation of all tree based algorithms, such as random forest and XGBoost. Although decision trees themselves aren't used in ML applications, understanding how they work is key to understanding more performant algorithms.

A machine learning problem is set up so that, in the training dataset, we have a set of features $(X)$ and a label $(y_{\text{true}})$. Decision trees work by analysing $X$ at each stage of the decision tree (node) to identify the optimal method of splitting a subset of the data into two subsets that most closely aligns the predicted labels $(y_{\text{pred}})$ with $(y_{\text{true}})$. It does so by reducing the impurity or variance in classification and regression problems respectively. This process is repeated until regularisation parameters kick in.

## Extra Points

- Decision trees are a white box model, meaning they are easy to interpret.
- Finding the optimal tree is very computationally expensive. Therefore, decision trees are greedy algorithms, meaning they find the optimal solution at each stage of the process, even though this doesn't guarantee a global optimum.
- Decision trees are a nonparametric model, meaning that the number of parameters of the model (degrees of freedom) aren't determined prior to training. In comparison, linear regression models assumes the number of parameters. Therefore regularisation is required to ensure the model doesn't overfit the training data. [1]

## Key Hyperparameters

- Criterion: The function to measure the quality of a split (Gini, entropy or variance).
- Max depth: The maximum depth of the tree.
- Min samples split: The minimum number of samples required to split an internal node.
- Min samples leaf: The minimum number of samples required to be at a leaf node.

## Key Equations

### Variance

Variance is a measure of how spread out the values in a continuous dataset are around the mean. It disproportionately punishes outliers by taking the square. In a decision tree regression problem, the variance (or weighted sum of the variance in the child nodes) is the loss function that we want to minimise. 

$$
\text{Var} = \frac{1}{n}\sum_{i=1}^{n} (y_i - \bar{y})^2
$$

Where:

- $y_i$: The target value of sample $i$.
- $\overline{y}$: The mean of the target values.
- $n$: The number of samples.
- $(y_i - \bar{y})$: The distance from a sample to the mean.

### Variance Reduction

The variance reduction is the difference in variance between the parent node and the child nodes.

$$
\begin{aligned} \Delta \text{Var} &= \text{Var}_{\text{parent}} - \text{Var}_{\text{split}} \\ &= \text{Var}_{\text{parent}} - \sum_{j=1}^{m} \left(\frac{n_j}{n_{\text{parent}}} \cdot \text{Var}_j \right) \end{aligned}
$$

- $\text{Var}_{\text{split}}$ is the weighted sum of the variance of the child nodes.
- $n_{parent}$ is the number of samples in the parent node.
- $n_j$ is the number of samples in child node $j$.
- $\text{Var}_j$ is the variance of child node $j$.

# Random Forest

## Theory

A random forest model is a collection of decision trees that are trained indendently on a sample of the data. To create predictions, the mean of the predictions from all the decision trees is taken. The random forest model is less prone to overfitting than a single decision tree, as each tree is fit to a random sample of the data.

## Extra Points

- The random forest model is an Ensemble technique / ensemble method, as it's predictions are a combination of agroup of predictors. This approach to ML follows the wisfdom of the crowd, as opposed to a single best (overfit) model.
- Random forest models are a collection of decision trees trained on random subsets of the training set. This sampling is usually performed with replacement and is called bagging (short for bootstrap aggregating).
- Each individual predictor has a higher bias than if it were trained on the original training set, but aggregation reduces both bias and variance. Generally, the net result is that the ensemble has a similar bias but a lower variance than a single predictor trained on the original training set.
- Random forest algorithms can introduce extra randomness between trees by randomly restricting the number of features in the training data for each tree. Trading a higher bias for a lower varianc, generally yielding a better model.

## Key Equations

## Predictions

The prediction of a random forest model is equal to the average of the predictions from the individual trees.

$$
\hat{y}(x) = \frac{1}{T}\sum_{t=1}^{T}f_t(x)
$$

- $T$ is the number of decision trees.
- $f_t(x)$ is the prediction of tree $t$.
- $\hat{y}(x)$ is the random forest prediction.

# Gradient Boosting

Gradient boosting is a simplified version of extreme gradient boosting (XGBoost).

## The Loss Function

In decision trees (and other ML models), the core idea is to improve a model by minimising a loss function. The loss function quantifies the difference between the model's predictions $(\hat{y})$ and the target variable $(y)$, with the goal of minimizing this difference to improve model accuracy. Therefore, we look to find a function/model $F(x)$ that minimises the equation $L(F(x), y)$.

## Gradient Descent

In mathematics, it's useful to find the minimum of a function. This is often difficult to do analytically, as the function must be double differentiable, therefore it's often done computationally via gradient descent methods.

In gradient descent, the goal is to find a set of parameters $x$ that minimize a function $F(x)$. This can be done iteratively,

$$
x_{n+1} = x_n - \gamma \Delta F(x_n)
$$

where:
- $x_n$ is the current set of parameters at iteration $n$
- $\gamma$ is the learning rate (step size factor)
- $\Delta F(x)$ is the gradient of the function with respect to the parameters

The $\gamma \Delta F(x)$ changes $x$ in such a way that $F(x)$ decreases in value, approaching a minimum.
 
## Applied Gradient Descent (Gradient Boosting)

Let's assume that there is a machine learning model that has an optimal set of parameters $F_*(x) = y$. By defition, this $F_*(x)$ must have minised a loss function $L(F(x), y)$ 

In standard gradient descent, we iteravitely update the parameters of a model by moving in the negative direction of the gradient of a function. In the context of gradient boosting, our goal is not to directly update the parameter vector of a model, but to improve our model $F(x)$ so that it's predictions, $\hat{y}$ can approach $y$ - and the loss function is minimised.

We update the model $F(x)$ by adding a tree (weak learner):

$$
F_{n+1}(x) = F_n(x) + \gamma h_n(x)
$$

Where the new tree $h(x)$:
- Predicts the residuals of model $F_n(x)$
- The loss function of $h(x)$ is the negative gradient of the loss function

$$
h_n(x) = - \frac{\partial L(y, F_n(x))}{\partial F_n(x)}
$$

By training a new decision tree $h_n(x)$ to predict the negative of this gradient, we are effectively determining the optimal "correction" in function space that would decrease the loss most effectively. This is analogous to the gradient descent update rule in parameter space but applied to the space of functions. Over successive iterations, this process moved $F_n(x)$ closer to the optimal model $F_*(x)$ that minimises $L(F(x), y)$

## Applied Example

Let's say we are training a gradient boosting model with a square error loss function. Often multiplied by $\frac{1}{2}$ for ease of calculation.

$$
L(\hat{y}, y) = \frac{1}{2}(\hat{y} - y)^2
$$

The negative differential of this loss function is

$$
\Delta L(\hat{y}, y) = y - \hat{y}
$$

### First Model

The first model (or base model) usually predicts the mean of $y$.

$$
F_0(x) = \bar{y} = \frac{1}{N} \sum_{i=1}^{N}y_i
$$

### Each iteration

At each iteration, a new tree is trained on the residuals of the previous model:

1. The residuals $r_i$ are calculated,

$$
\begin{aligned}
r_i^{(n)} &= y_i - F_n(x) \\
&= y_i - \hat{y}_i
\end{aligned}
$$

2. A new tree is trained to predict these residuals,

$$
h_{n}(x) \approx -r_i^{n}
$$

3. The model is updated

$$
F_{n+1}(x) = F_n(x) + \gamma h_n(x)
$$


### Predictions

Once all $T$ trees have been trained and added to the model, the final prediction for any new input $x$ is obtained by summing the contributions of all trees along with the initial prediction:

$$
\hat{y} = F_T(x) = F_0(x) + \gamma \sum_{n=1}^T h_n(x)
$$

# Extreme Gradient Boosting

The XGBoost model builds upon the gradient boosting model to enhance optimisation and regularisation. 

## Second-Order Taylor Expansion

Instead of just using the first derivative (gradient) of the loss, Newton’s method approximates the loss using a second-order Taylor expansion around the current prediction $F_n(x)$:
$$
L(y, F_n(x) + f(x)) \approx L(y, F_n(x)) + f(x) \cdot g + \frac{1}{2}f(x)^2 \cdot h
$$

Where:
* $g = \frac{\partial L(y, F_n(x))}{\partial F_n(x)}$ is the gradient
* $h = \frac{\partial^2 L(y, F_n(x))}{\partial F_n(x)^2}$ is the gradient

## Optimal Leaf Weight Calculation

When building a tree, XGBoost paritions the data into leaves. For a given leaf $j$ containing samples $I_j$, we assume that te tree outputs a constant value $w_j$ for that leaf. The total approximate loss for that leaf becomes:

$$
\sum_{i \epsilon I_j}(g_iw_j + \frac{1}{2}h_iw_j^2)
$$

To minimise this with respect to $w_j$ (the prediction for the leaf - the thing we have control over), we set the derivative to zero:

$$
\frac{\partial}{\partial w_j}
\sum_{i \epsilon I_j}(g_iw_j + \frac{1}{2}h_iw_j^2) = \sum_{i \epsilon I_j}(g_i + h_iw_j) = 0
$$

Solving for $w_j$ gives:

$$
w_j^* = - \frac{\sum_{i \epsilon I_j } g_i}{\sum_{i \epsilon I_j} h_i + \lambda}
$$

where $\lambda$ is an additional regularisation parameter to prevent overfitting.

## Points of Clarification

The loss function specified by the user isn't the actual loss function used in training the decision trees in the XGBoost models. Although this remains the objective of the model overall. The local behaviour of the loss function is approximated using its gradient and Hessian to determine the updates of each tree.

The term "leaf weight" in XGBoost refers to the constant value assigned to all samples that fall into a given leaf of a tree. In each boosting iteration, instead of predicting a complex function within a leaf, XGBoost uses a single number—the leaf weight—as the prediction for all samples in that leaf.

So, while the leaf weight is essentially the predicted value for samples in that leaf, it’s called a "weight" because it’s derived from an optimization process (using the gradient and Hessian of the loss function) to determine the best constant update for that group of samples. In the overall model, the final prediction $\hat{y}$ is the sum of the base prediction and the contributions (leaf weights) from all the trees:

$$
\hat{y} = F_T(x) = F_0(x) + \gamma \sum_{n=1}^T w_{j(t)}
$$

where $w_{j(t)}$ is the leaf weight for the leaf


## References

**Writing**

1. Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow (Aurélien Géron) (Z-Library).pdf
2. [https://www.youtube.com/watch?v=u4IxOk2ijSs](https://www.youtube.com/watch?v=u4IxOk2ijSs)

**Code**

- https://github.com/enesozeren/machine_learning_from_scratch/blob/main/decision_trees/decision_tree.py
- https://github.com/Suji04/ML_from_Scratch/blob/master/decision%20tree%20classification.ipynb

## TODO

- Implement for categorical problem
- Implement for continuous problem