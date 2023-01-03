"""
Common functions module test
"""
import pandas as pd
import numpy as np
import pytest
from joblib import load
from starter.ml.data import process_data
from starter.ml.model import inference
import yaml

with open('config.yml') as f:
    config = yaml.load(f, Loader= yaml.FullLoader)
cat_features = config['cat_features']

@pytest.fixture
def data():
    """
    Get dataset
    """
    df = pd.read_csv("data/clean_census.csv")
    return df


def test_process_data(data):
    """
    Check split have same number of rows for X and y
    """
    encoder = load("model/encoder.joblib")
    lb = load("model/lb.joblib")

    

    X_test, y_test, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label="salary", encoder=encoder, lb=lb, training=False)

    assert len(X_test) == len(y_test)


def test_process_encoder(data):
    """
    Check split have same number of rows for X and y
    """
    encoder_test = load("model/encoder.joblib")
    lb_test = load("model/lb.joblib")

    _, _, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label="salary", training=True)

    _, _, _, _ = process_data(
        data,
        categorical_features=cat_features,
        label="salary", encoder=encoder_test, lb=lb_test, training=False)

    assert encoder.get_params() == encoder_test.get_params()
    assert lb.get_params() == lb_test.get_params()


def test_inference_below():
    """
    Check inference performance
    """
    model = load("model/model.joblib")
    encoder = load("model/encoder.joblib")
    lb = load("model/lb.joblib")

    array = np.array([[
                     53,
                     "Private",
                     234721,
                     "11th",
                     "Married-civ-spouse",
                     "Own-child",
                     "Handlers-cleaners",
                     "Black",
                     "Male",
                     40,
                     "United-States"
                     ]])
    df_temp = pd.DataFrame(data=array, columns=[
        "age",
        "workclass",
        "fnlgt",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "hours-per-week",
        "native-country",
    ])

    X, _, _, _ = process_data(
                df_temp,
                categorical_features=cat_features,
                encoder=encoder, lb=lb, training=False)
    pred = inference(model, X)
    y = lb.inverse_transform(pred)[0]
    assert y == "<=50K"