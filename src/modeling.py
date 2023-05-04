from data.utils import load_data_to_dataframe, save_data_to_csv
from data.modeling import convert_categorical, categorical_one_hot_encoder

def main():
    try:
        dataframe = load_data_to_dataframe(
            data_path="../data/interim/telco_customer_churn.csv")
    except:
        print("It was not possible to read the provided .csv file")
        exit(0)


    # Replaces target variable with numerical values
    target_column = convert_categorical(column=dataframe["Churn"], to_replace=["Yes", "No"], values=[1, 0], target_type="float64")

    # Drops columns that will not be used or is already stored (target column)
    dataframe = dataframe.drop(["customerID", "Churn"], axis="columns")

    # Uses hot encoder to encode categorical variables
    categorical_columns = ["gender", "Partner", "Dependents", "PhoneService", 
                          "MultipleLines", "InternetService", "OnlineSecurity",
                          "OnlineBackup", "DeviceProtection", "TechSupport",
                          "StreamingTV", "StreamingMovies", "Contract",
                          "PaperlessBilling", "PaymentMethod"]
    numerical_dataframe = categorical_one_hot_encoder(dataframe=dataframe, columns_to_encode=categorical_columns)   
    numerical_dataframe["churn"] = target_column 

    save_data_to_csv(dataframe=numerical_dataframe, data_path="../data/processed/telco_customer_churn.csv")

if __name__ == "__main__":
    main()