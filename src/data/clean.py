import numpy as np
import pandas as pd

def convert_str_to_type(column: pd.Series, type: str = "float64") -> pd.Series:
    """ Converts a string column to the specified type

    Parameters:
    column: Column to be manipulated
    type: Type to perform casting

    Returns:
    manipulated: Column converted to given type
    """
    manipulated = column.replace(' ', np.nan)
    manipulated = manipulated.astype(type)

    return manipulated


def spline_missing_values(column: pd.Series) -> pd.Series:
    """ Handle missing values by interpolation

    Parameters:
    column: Column to be manipulated
    method: Method to interpolate, available at: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.interpolate.html


    Returns:
    manipulated: Interpolated column without missing values
    """
    manipulated = column.interpolate(method="spline", limit_direction='backward', order=1)

    return manipulated