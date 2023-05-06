from models.train import xgboost_train, log_reg_train, svm_train, knn_train
from models.test import model_test
from models.utils import  get_train_test_data
from data.utils import load_data_to_dataframe

def main():
    try:
        dataframe = load_data_to_dataframe(
            data_path="../data/processed/telco_customer_churn.csv")
    except:
        print("It was not possible to read the provided .csv file")
        exit(0)

    X_train, X_test, y_train, y_test = get_train_test_data(dataframe, ["churn"])
    
    search = xgboost_train(X_train, y_train, n_iter=5)
    print("Results from XGBoost:")
    _ = model_test(search, X_test, y_test)
    print(search.best_estimator_)
    print(search.best_score_)
    print(search.best_params_)
    print("\n")

    search = log_reg_train(X_train, y_train.values.ravel(), n_iter=5)
    print("Results from Logistic Regression:")
    _ = model_test(search, X_test, y_test)
    print(search.best_estimator_)
    print(search.best_score_)
    print(search.best_params_)
    print("\n")

    search = svm_train(X_train, y_train.values.ravel(), n_iter=5)
    print("Results from SVM:")
    _ = model_test(search, X_test, y_test)
    print(search.best_estimator_)
    print(search.best_score_)
    print(search.best_params_)
    print("\n")

    search = knn_train(X_train, y_train.values.ravel(), n_iter=5)
    print("Results from KNN:")
    _ = model_test(search, X_test, y_test)
    print(search.best_estimator_)
    print(search.best_score_)
    print(search.best_params_)
    print("\n")


if __name__ == "__main__":
    main()