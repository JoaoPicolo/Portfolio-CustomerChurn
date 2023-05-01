import sys
import pandas as pd

sys.path.append(".")
from src.data.utils import load_data_to_dataframe, save_data_to_csv


def test_dataframe_load():
    """
        Test if the dataframe is being correctly loaded
    """
    dataframe = load_data_to_dataframe("data/raw/telco_customer_churn.csv")
    assert type(dataframe) == pd.DataFrame


def test_dataframe_save():
    """
        Test if the dataframe is being correctly saved
    """
    dataframe = pd.read_csv("data/raw/telco_customer_churn.csv")
    result = save_data_to_csv(dataframe, data_path="data/raw/telco_customer_churn.csv")
    
    assert result == None