from models.xgboost import xgboost_train, xgboost_test
from models.utils import  evaluate_cv_search, get_train_test_data
from data.utils import load_data_to_dataframe

def main():
    try:
        dataframe = load_data_to_dataframe(
            data_path="../data/processed/telco_customer_churn.csv")
    except:
        print("It was not possible to read the provided .csv file")
        exit(0)

    X_train, X_test, y_train, y_test = get_train_test_data(dataframe, ["churn"])
    xgboost_search = xgboost_train(X_train, y_train, n_iter=10)
    evaluate_cv_search(xgboost_search)
    _ = xgboost_test(xgboost_search, X_test, y_test)


if __name__ == "__main__":
    main()