from data.utils import load_data_to_dataframe, save_data_to_csv
from data.clean import convert_str_to_type, spline_missing_values

def main():
    try:
        dataframe = load_data_to_dataframe(
            data_path="../data/raw/telco_customer_churn.csv")
    except:
        print("It was not possible to read the provided .csv file")
        exit(0)

    dataframe["TotalCharges"] = convert_str_to_type(column=dataframe["TotalCharges"], type="float64")
    dataframe["TotalCharges"] = spline_missing_values(dataframe["TotalCharges"])

    save_data_to_csv(dataframe=dataframe, data_path="../data/interim/telco_customer_churn.csv")


if __name__ == "__main__":
    main()