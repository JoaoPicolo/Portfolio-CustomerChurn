import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV

def model_test(search: RandomizedSearchCV, X_test: pd.DataFrame, y_test: pd.DataFrame) -> np.ndarray:
    """ Testes the trained model using RandomizedSearchCV against the data

    Parameters:
    search: The randomized search object
    X_test: The data to be tested
    y_test: The labels for each test data

    Returns:
    y_pred: The predicted values for each test data
    """
    y_pred = search.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    return y_pred