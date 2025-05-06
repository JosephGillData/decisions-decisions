import numpy as np
import pandas as pd
import logging
from python_scripts.decision_tree import DecisionTree

class GradientBoosting:

    def __init__(
            self,
            # gradient boosting parameters
            n_estimators=3,
            learning_rate=0.01,
            # decision tree parameters
            problem_type='regression',
            criterion='squared_error',
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2
            ):

        # Initialise the parameters
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.problem_type = problem_type
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def _update_criterion_with_negative_differential(self):

        # Update the loss function (criterion) with the negative differential 
        if self.criterion == 'squared_error':
            self.criterion = 'negative_squared_error_diff'

    def train(self, X_train, y_train):

        X_y_train = pd.concat([X_train, y_train], axis=1)
        target_column = y_train.columns[0]

        self._update_criterion_with_negative_differential()

        # Initial prediction is the mean of the target column
        y_0 = y_train[target_column].mean()
        # Initialize the cumulative prediction for all samples
        strong_learner_preds = np.full(len(y_train), y_0)
        # Store models
        gb_model = [y_0]

        # Iterate over the number of boosting rounds
        for tree_num in range(1, self.n_estimators+1):
            print('tree_num', tree_num)
            
            # Calculate residuals based on the current cumulative model prediction
            residuals = y_train[target_column] - strong_learner_preds
            residuals = pd.DataFrame({target_column: residuals})

            # Initialise decision tree
            decision_tree = DecisionTree(
                problem_type=self.problem_type,
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                )
            dt_model = decision_tree.train(X_train, residuals)
            # Store the gradient boosting model
            gb_model.append(dt_model)

            # Predict on training data with the new tree
            weak_learner_preds = decision_tree.predict(dt_model, X_train)

            # Update the cumulative prediction: add the new tree's predictions scaled by the learning rate
            strong_learner_preds += self.learning_rate * weak_learner_preds

        return gb_model

    def predict(self, model, X_test):

        # Initialise decision tree as a class that is reusable for each decision tree prediction
        decision_tree = DecisionTree(
            problem_type=self.problem_type,
            criterion=self.criterion,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            )

        # Set prediction as the initial prediction (of F_0)
        strong_learner_preds = np.full(X_test.shape[0], model[0])

        # Iterate over the number of boosting rounds (models)
        for tree_num in range(1, self.n_estimators+1):

            # Get the model and predict on X_test
            dt_model = model[tree_num]
            weak_learner_preds = decision_tree.predict(dt_model, X_test)
            strong_learner_preds += (self.learning_rate * weak_learner_preds)

        return strong_learner_preds

