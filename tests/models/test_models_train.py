import sys

import pandas as pd
from sklearn.model_selection import RandomizedSearchCV

sys.path.append(".")
from src.models.train import xgboost_train, log_reg_train, svm_train, knn_train


def test_xgboost_train():
    """ Test training xgboost algorithm """
    data = {
        "age": [25, 30, 35, 26, 35, 69, 25, 21],
        "is_student": [1, 0, 1, 1, 0, 1, 0, 1]
    }
    dataframe = pd.DataFrame(data)
    X_train, y_train = dataframe["age"], dataframe["is_student"]
    search = xgboost_train(X_train, y_train, n_iter=1, cv=2)
    assert search.__class__ == RandomizedSearchCV


def test_log_reg_train():
    """ Test training Logistic Regression algorithm """
    data = {
        "age": [25, 30, 35, 26, 35, 69, 25, 21],
        "is_student": [1, 0, 1, 1, 0, 1, 0, 1]
    }
    dataframe = pd.DataFrame(data)
    X_train, y_train = dataframe["age"], dataframe["is_student"]
    X_train = X_train.to_numpy()
    search = log_reg_train(X_train.reshape(-1, 1), y_train.values.ravel(), n_iter=1, cv=2)
    assert search.__class__ == RandomizedSearchCV


def test_svm_train():
    """ Test training SVM algorithm """
    data = {
        "age": [25, 30, 35, 26, 35, 69, 25, 21],
        "is_student": [1, 0, 1, 1, 0, 1, 0, 1]
    }
    dataframe = pd.DataFrame(data)
    X_train, y_train = dataframe["age"], dataframe["is_student"]
    X_train = X_train.to_numpy()
    search = svm_train(X_train.reshape(-1, 1), y_train.values.ravel(), n_iter=1, cv=2)
    assert search.__class__ == RandomizedSearchCV


def test_knn_train():
    """ Test training KNN algorithm """
    data = {
        "age": [25, 30, 35, 26, 35, 69, 25, 21],
        "is_student": [1, 0, 1, 1, 0, 1, 0, 1]
    }
    dataframe = pd.DataFrame(data)
    X_train, y_train = dataframe["age"], dataframe["is_student"]
    X_train = X_train.to_numpy()
    search = knn_train(X_train.reshape(-1, 1), y_train.values.ravel(), n_iter=1, cv=2)
    assert search.__class__ == RandomizedSearchCV
