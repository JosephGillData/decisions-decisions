import numpy as np
import pandas as pd
import logging
from python_scripts.decision_tree import DecisionTree

class RandomForest:

    def __init__(
            self,
            # random forest parameters
            n_estimators=100,
            # decision tree parameters
            problem_type='regression',
            criterion='variance',
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            ):

        # Initialise the parameters
        self.n_estimators = n_estimators
        self.problem_type = problem_type
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf

    def train(self, X_train, y_train):

        X_y_train = pd.concat([X_train, y_train], axis=1)
        target_column = y_train.columns[0]

        dt_models = []

        for tree_num in range(1, self.n_estimators+1):

            print(tree_num)

            X_y_sample = X_y_train.sample(frac=1, replace=True)
            y_sample = pd.DataFrame(X_y_sample[target_column])
            X_sample = X_y_sample.drop(columns=[target_column])

            decision_tree = DecisionTree(
                problem_type=self.problem_type,
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                )
            
            dt_model = decision_tree.train(X_sample, y_sample)
            dt_models.append(dt_model)

        return dt_models

    def predict(self, model, X_test):

        dt_preds = []

        for dt_model in model:

            decision_tree = DecisionTree(
                problem_type='regression',
                criterion='variance',
                max_depth=3,
                min_samples_split=10,
                min_samples_leaf=2,
                )

            dt_pred = decision_tree.predict(dt_model, X_test)
            dt_preds.append(dt_pred)

        dt_preds = np.array(dt_preds)
        rf_preds = dt_preds.mean(axis=0)

        return rf_preds


                    

    
