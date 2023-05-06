from typing import List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def get_train_test_data(
        dataframe: pd.DataFrame, target_variables: List[str],
        test_size: float = 0.2
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """ Splits the dataset for training and testing

    Arguments:
    dataframe: Dataframe to be split
    target_variables: Target variables to be used by the model
    test_size: Quantity of data that will go to the test set
    """
    features = list(dataframe.columns)
    for var in target_variables:
        features.remove(var)

    X = dataframe[features]
    y = dataframe[target_variables]

    return train_test_split(X, y, test_size=test_size, random_state=42)