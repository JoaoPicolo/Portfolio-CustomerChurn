import sys

import pandas as pd

sys.path.append(".")
from src.models.xgboost import xgboost_train, xgboost_test
from sklearn.model_selection import RandomizedSearchCV


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


def test_xgboost_test():
    """ Test test xgboost algorithm """
    data = {
        "age": [25, 30, 35, 26, 35, 69, 25, 21, 35, 69, 25, 21],
        "is_student": [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1]
    }
    dataframe = pd.DataFrame(data)
    df1 = dataframe.iloc[:6, :]
    df2 = dataframe.iloc[6:, :]
    X_train, y_train = df1["age"], df1["is_student"]
    X_test, y_test = df2["age"], df2["is_student"]
    search = xgboost_train(X_train, y_train, n_iter=1, cv=2)
    res = xgboost_test(search, X_test, y_test)
    assert res.shape == y_test.shape
