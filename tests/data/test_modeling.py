import sys

import numpy as np
import pandas as pd

sys.path.append(".")
from src.data.modeling import convert_categorical, categorical_one_hot_encoder

def test_convert_categorical():
    """
        Test if the series are being correctly converted to numerical
    """
    column = pd.Series(["Val1", "Val2", "Val1", "Val4", "Val3", "Val2"])
    target_type = "float64"
    numerical = convert_categorical(column, to_replace=["Val1", "Val2", "Val3", "Val4"],
                                    values=[0, 1, 2, 3], target_type=target_type)
    
    assert numerical.dtype == target_type

def test_categorical_one_hot_encoder():
    """
        Test if the dataframe is being converted to hot-encoded values
    """
    data = {
        "name": ["John", "Sarah", "Mike"],
        "age": [25, 30, 35],
        "is_student": [True, False, True]
    }
    dataframe = pd.DataFrame(data)
    numerical_dataframe = categorical_one_hot_encoder(dataframe=dataframe, columns_to_encode=["name", "is_student"]) 
    all_columns_float = True

    for col in numerical_dataframe.columns:
        if numerical_dataframe[col].dtype != "float64":
            all_columns_float = False
            break
    
    assert all_columns_float == True