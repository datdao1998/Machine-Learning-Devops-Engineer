# script doc string
'''
This script to perfom test & logging for churn_library.py file

Author: Dao Quoc Dat
Date: August 23, 2022
'''

from genericpath import isfile
import os
import logging
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err
    return df


def test_eda(perform_eda, data):
    '''
    test perform eda function
    '''
    perform_eda(data)
    eda_path = "./images/eda"

    check_files = [
        'customer_age_distribution.png',
        'total_transaction_distribution.png',
        'churn_distribution.png',
        'marital_status_distribution.png',
        'heatmap.png']

    # Checking if eda images are available
    try:
        for file in check_files:
            assert os.path.isfile(os.path.join(eda_path, file))
        logging.info("Testing perform_eda: SUCCESS")
    except AssertionError as err:
        logging.warning("Testing perform_eda: Missing file when perform EDA")
        raise err


def test_encoder_helper(encoder_helper, data):
    '''
    test encoder helper
    '''

    # Encoded columns
    encoded_cols = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    # Encode funtion
    data = encoder_helper(data, encoded_cols, 'Churn')

    try:
        for category in encoded_cols:
            assert category in data.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper: Some columns are missing.")
        return err
    return data


def test_perform_feature_engineering(perform_feature_engineering, data):
    '''
    test perform_feature_engineering
    '''

    # Perform feature engineering function
    x_train, x_test, y_train, y_test = perform_feature_engineering(
        data, 'Churn')

    try:
        # validate shape and length
        assert x_train.shape[0] > 0
        assert x_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Some data is missing")
        raise err

    try:
        # validate matching train/test
        assert len(x_train) == len(y_train)
        assert len(x_test) == len(y_test)
    except AssertionError as err:
        logging.error("The shapes of train/test split don't match")
        raise err
    return x_train, x_test, y_train, y_test


def test_train_models(train_models, x_train, x_test, y_train, y_test):
    '''
    test train_models
    '''
    train_models(x_train, x_test, y_train, y_test)

    # test result images
    results_path = "./images/results/"
    try:
        dir_val = os.listdir(results_path)
        assert len(dir_val) > 0
    except FileNotFoundError as err:
        logging.error("Testing train_models: Some result images are missing")
        raise err

    # test model saving
    model_path = "./models/"
    try:
        dir_val = os.listdir(model_path)
        assert len(dir_val) > 0
        logging.info("Testing train_models: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing train_models: Failed when saving models")
        raise err


if __name__ == "__main__":
    # test load data
    data = test_import(cls.import_data)

    # test perform eda
    test_eda(cls.perform_eda, data)

    # test encoder function
    data = test_encoder_helper(cls.encoder_helper, data)

    # test feature engineering
    x_train, x_test, y_train, y_test = test_perform_feature_engineering(
        cls.perform_feature_engineering, data)

    # test train model
    test_train_models(cls.train_models, x_train, x_test, y_train, y_test)
