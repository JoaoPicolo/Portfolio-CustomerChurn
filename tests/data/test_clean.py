import sys

import numpy as np
import pandas as pd

sys.path.append(".")
from src.data.clean import convert_str_to_type, spline_missing_values

def test_convert_str_col():
    """
        Test if the series are being correctly converted
    """
    column = pd.Series(["", "2.64", "0.5", "", "0.07", "-0.9"])
    type = "float64"
    converted = convert_str_to_type(column=column, type=type)
    assert converted.dtype == type


def test_spline_missing():
    """
        Test if the series is being correclty interpolated
    """
    column = pd.Series([np.nan, 2.64, 0.5, np.nan, 0.07, -0.9])
    interpolated = spline_missing_values(column=column)
    assert interpolated.isnull().sum() == 0