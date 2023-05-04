from typing import List

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

def convert_categorical(column: pd.Series, to_replace: List[any], values: List[any], target_type: str) -> pd.Series:
    """ Turns a categorical Series into a numerical

    Arguments:
    column: Series to be converted
    to_replace: Values to be replaced
    values: Values to replace
    target_type: The target type of the transformed column

    Returns:
    transformed: The transformed series
    """
    transformed = column.replace(to_replace, values)
    transformed = transformed.astype(target_type)

    return transformed

def categorical_one_hot_encoder(dataframe: pd.DataFrame, columns_to_encode: List[str]) -> pd.DataFrame:
    """ Uses one hot encoder to transofrm categorical variables into numeric

    Arguments:
    dataframe: Dataframe to be transformed
    columns_to_encode: List of columns to be converted

    Returns:
    transformed_df: Dataframe with transformed columns
    """
    ohe = OneHotEncoder(handle_unknown="ignore")
    categorical_processing = Pipeline(steps=[("ohe", ohe)])
    preprocessing = ColumnTransformer(transformers=[("categorical", categorical_processing, columns_to_encode)],
                                      remainder="passthrough")

    transformed_data = preprocessing.fit_transform(dataframe)
    transformed_cols = preprocessing.get_feature_names_out(dataframe.columns)
    transformed_df = pd.DataFrame(transformed_data, columns=transformed_cols)

    return transformed_df