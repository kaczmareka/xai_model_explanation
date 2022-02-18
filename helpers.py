# Imports
import numpy as np

from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline


# Helpers for the hyperparameter search
def _adjust_key_names(dict_hyperparameters, add: bool = False):
    """
    Adjusts names of dictionary keys so that they are compatible with the pipeline and sklearn model class.
    Important: does not create a new dictionary, modifies the exist one!

    :param add: If add is True, then "clf__" is added to the beginning of key names (necessary for pipeline).
                If add is False, then "clf__" is removed from the beginning of key names.
    :return: None
    """
    if add:
        dict_keys = ["clf__" + key for key in list(dict_hyperparameters.keys())]
        for dict_key in dict_keys:
            dict_hyperparameters[dict_key] = dict_hyperparameters.pop(dict_key[5:])
    else:
        dict_keys = [key[5:] for key in list(dict_hyperparameters.keys())]
        for dict_key in dict_keys:
            dict_hyperparameters[dict_key] = dict_hyperparameters.pop("clf__" + dict_key)


class Hyperparameter_Search():
    """Class for training a classifier and finding the most ideal hyperparameters."""

    def __init__(self, model_class, hyp_grid: dict, scaler=None, random_state: int = 66):
        """
        Initializes our classifier evaluator.

        :param model_class: Model class (RandomForrest, SVC, ...)
        :param hyp_grid: Our hyperparameters for the hyperparameter search.
        :param hyp_search: Should we look for the best hyperameters?
        :param scaler: Do we want to scale our data? For example, StandardScaler
        :param n_folds: Number of folds in outer cross validation.
        :param random_state: Random state
        """
        self.model_class = model_class

        # Change the key names in the hyperparameter_grid dictionary (because of the pipeline)
        hyper_grid = hyp_grid.copy()
        _adjust_key_names(hyper_grid, add=True)
        self.hyp_grid = hyper_grid

        # Define Pipeline for grid search cross validation
        if scaler is not None:
            self.pipeline = Pipeline([("scale", scaler()), ("clf", model_class())])
            self.scaler = scaler
        else:
            self.pipeline = Pipeline([("clf", model_class())])
            self.scaler = None

        # Save the random state
        self.random_state = random_state

        # Prepare empty variables for saving the results
        self.best_hyp_comb = None
        self.accuracies = None
        self.hyperparameter_comb = None

    def hyperparameter_search(self, X, y, n_folds: int = 5):
        """
        Conducts hyperparameter search on training and validation set.

        :param X: Our training dataset
        :param y: Our training labels
        """
        # Create splits
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)  # Does not involve randomness
        kf_split = kf.split(X, y)

        # Do the inner cross validation to find the ideal hyperparameters
        search_cl = GridSearchCV(self.pipeline,
                                 param_grid=self.hyp_grid,
                                 cv=kf_split,
                                 scoring="accuracy",
                                 refit=True,
                                 n_jobs=-1,
                                 return_train_score=True)

        search_cl.fit(X, y)

        # Get the best hyperparameters based on our hyperparameter search
        best_hyp = search_cl.best_params_.copy()
        _adjust_key_names(dict_hyperparameters=best_hyp, add=False)

        # Append info for measuring performance
        self.best_hyp_comb = best_hyp
        self.accuracies = search_cl.cv_results_['mean_test_score']
        self.hyperparameter_comb = search_cl.cv_results_['params']

    def get_best_combo(self):
        """Returns the hyperparameters with the highest accuracy on the data."""
        return self.best_hyp_comb

    def print_results(self, top_k=None):
        """Returns all the results (hyp. combo and its accuracy)"""
        top_k = len(self.hyperparameter_comb) if top_k is None else top_k
        sorted_indexes = np.argsort(self.accuracies, axis=-1)[::-1][:top_k]

        print("***Ordered Hyperparameter Combinations***")
        for j in sorted_indexes:
            print(f"Acc: {self.accuracies[j]}, combo: {self.hyperparameter_comb[j]}")

