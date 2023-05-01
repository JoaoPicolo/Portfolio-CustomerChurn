from data.utils import load_data_to_dataframe, save_data_to_csv

def main():
    try:
        dataframe = load_data_to_dataframe(
            data_path="../data/raw/telco_customer_churn.csv")
    except:
        print("It was not possible to read the provided .csv file")
        exit(0)

    print(dataframe.head())


if __name__ == "__main__":
    main()