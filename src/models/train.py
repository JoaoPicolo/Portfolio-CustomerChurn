import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV


def perform_random_search(X_train: pd.DataFrame, y_train: pd.DataFrame,
                          model: any, parameters: any, n_iter: int, cv: int) -> RandomizedSearchCV:
    """ Applies random search over model

    Parameters:
    X_train: The data to be used during training
    y_train: The labels for each train data
    n_iter: Number of iterations to be used during parameters tuning in the model
    cv: The number of cross fold validations to perform

    Returns:
    search: The randomized search object containing the tunned model
    """

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=parameters,
        scoring="accuracy",
        random_state=42,
        n_iter=n_iter,
        cv=cv, verbose=2)

    search.fit(X_train, y_train)

    return search

def xgboost_train(X_train: pd.DataFrame, y_train: pd.DataFrame, n_iter: int = 5, cv: int = 5) -> RandomizedSearchCV:
    """ Trains the XGBoost model against the data

    Parameters:
    X_train: The data to be used during training
    y_train: The labels for each train data
    n_iter: Number of iterations to be used during parameters tuning in the model
    cv: The number of cross fold validations to perform

    Returns:
    search: The randomized search object containing the tunned model
    """
    params = {
        "max_depth": [3, 5, 6, 10, 15, 20],
        "learning_rate": [0.01, 0.1, 0.2, 0.3],
        "subsample": np.arange(0.5, 1.0, 0.1),
        "colsample_bytree": np.arange(0.4, 1.0, 0.1),
        "colsample_bylevel": np.arange(0.4, 1.0, 0.1),
        "n_estimators": [100, 500, 1000]
    }

    xgb_model = xgb.XGBClassifier(random_state=42)
    search = perform_random_search(X_train, y_train, xgb_model, params, n_iter, cv)
    return search

    return search


def log_reg_train(X_train: pd.DataFrame, y_train: pd.DataFrame, n_iter: int = 5, cv: int = 5) -> RandomizedSearchCV:
    """ Trains the Logistic Regression model against the data

    Parameters:
    X_train: The data to be used during training
    y_train: The labels for each train data
    n_iter: Number of iterations to be used during parameters tuning in the model
    cv: The number of cross fold validations to perform

    Returns:
    search: The randomized search object containing the tunned model
    """

    params = {
        "warm_start": [True, False],
        "C": np.arange(0, 1, 0.01),
        "solver": ["lbfgs", "liblinear", "newton-cg"]
    }

    log_reg = LogisticRegression(random_state=42)
    search = perform_random_search(X_train, y_train, log_reg, params, n_iter, cv)
    return search


def svm_train(X_train: pd.DataFrame, y_train: pd.DataFrame, n_iter: int = 5, cv: int = 5) -> RandomizedSearchCV:
    """ Trains the Support Vector Machine model against the data

    Parameters:
    X_train: The data to be used during training
    y_train: The labels for each train data
    n_iter: Number of iterations to be used during parameters tuning in the model
    cv: The number of cross fold validations to perform

    Returns:
    search: The randomized search object containing the tunned model
    """

    params = {
        "C": np.arange(0, 1, 0.01),
        "kernel": ["rbf", "linear", "sigmoid"],
        "gamma": np.arange(0.1, 1, 0.1),
        "tol": [1e-4, 1e-8, 1e-1]
    }

    svm = SVC(random_state=42)
    search = perform_random_search(X_train, y_train, svm, params, n_iter, cv)
    return search


def knn_train(X_train: pd.DataFrame, y_train: pd.DataFrame, n_iter: int = 5, cv: int = 5) -> RandomizedSearchCV:
    """ Trains the K-Nearest Neighbors model against the data

    Parameters:
    X_train: The data to be used during training
    y_train: The labels for each train data
    n_iter: Number of iterations to be used during parameters tuning in the model
    cv: The number of cross fold validations to perform

    Returns:
    search: The randomized search object containing the tunned model
    """

    params = {
        "n_neighbors": np.arange(1, 50, 1),
        "weights": ["uniform", "distance"],
        "algorithm": ["ball_tree", "kd_tree", "brute"]
    }

    knn = KNeighborsClassifier()
    search = perform_random_search(X_train, y_train, knn, params, n_iter, cv)
    return search