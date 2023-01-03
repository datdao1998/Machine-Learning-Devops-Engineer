"""
This module outputs the performance of the model on slices of the data for categorical features.
"""
import os
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import load
from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics
import yaml 

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")

DATA_PATH = 'data/clean_census.csv'
ARTIFACTS_PATH = 'model'

with open('config.yml') as f:
    config = yaml.load(f, Loader= yaml.FullLoader)

def classification_slide_metric():
    """ Check performance on categorical features """

    df = pd.read_csv("data/clean_census.csv")
    _, test = train_test_split(df, test_size=0.20)

    trained_model = load(f"{ARTIFACTS_PATH}/model.joblib")
    encoder = load(f"{ARTIFACTS_PATH}/encoder.joblib")
    lb = load(f"{ARTIFACTS_PATH}/lb.joblib")

    features = config['cat_features']

    X_test, y_test, _, _ = process_data(
        test,
        categorical_features=features,
        label="salary", encoder=encoder, lb=lb, training=False)

    y_preds = trained_model.predict(X_test)

    print(X_test.shape)

    prc, rcl, fb = compute_model_metrics(y_test, y_preds)

    print('Precision Score : ',prc)
    print('Recall Score : ',rcl)
    print('FBeta Score : ', fb)

if __name__ == '__main__':
    classification_slide_metric()