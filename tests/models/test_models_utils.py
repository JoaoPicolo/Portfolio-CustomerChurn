import sys

import pandas as pd

sys.path.append(".")
from src.models.utils import get_train_test_data

def test_get_train_test_data():
    """ Tests the split data method """
    data = {
        "name": ["John", "Sarah", "Mike", "Paul"],
        "age": [25, 30, 35, 26],
        "is_student": [True, False, True, True]
    }
    dataframe = pd.DataFrame(data)
    X_train, X_test, _, _ = get_train_test_data(dataframe, ["is_student"], test_size=0.5)
    assert X_train.shape == X_test.shape
