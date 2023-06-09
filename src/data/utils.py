import pandas as pd

def load_data_to_dataframe(data_path: str) -> pd.DataFrame:
    """ Loads the .csv path into a pandas dataframe

    Arguments:
    data_path: Path to the .csv file

    Returns:
    dataframe: Returns the created pandas dataframe
    """
    dataframe = pd.read_csv(data_path)
    return dataframe


def save_data_to_csv(dataframe: pd.DataFrame, data_path: str):
    """ Saves the dataframe in the provided path as a .csv file

    Arguments:
    dataframe: Dataframe to be saved
    data_path: Path to the .csv file
    """
    dataframe.to_csv(data_path, index=False)