import pandas as pd
from sklearn.model_selection import train_test_split
from joblib import load
import logging
from starter.ml.data import process_data
from starter.ml.model import compute_model_metrics
import yaml

with open('config.yml') as f:
    config = yaml.load(f, Loader= yaml.FullLoader)

def accuracy():
    """
    Execute accuracy
    """
    df = pd.read_csv("data/clean_census.csv")
    _, test = train_test_split(df, test_size=0.20)

    trained_model = load("model/model.joblib")
    encoder = load("model/encoder.joblib")
    lb = load("model/lb.joblib")

    features = config['cat_features']

    slice_values = []

    for cat in features:
        for cls in test[cat].unique():
            df_temp = test[test[cat] == cls]

            X_test, y_test, _, _ = process_data(
                df_temp,
                categorical_features=features,
                label="salary", encoder=encoder, lb=lb, training=False)

            y_preds = trained_model.predict(X_test)

            prc, rcl, fb = compute_model_metrics(y_test, y_preds)

            line = "[%s->%s] Precision: %s " \
                   "Recall: %s FBeta: %s" % (cat, cls, prc, rcl, fb)
            logging.info(line)
            slice_values.append(line)

    with open('model/slice_output.txt', 'w') as out:
        for slice_value in slice_values:
            out.write(slice_value + '\n')

if __name__ == '__main__':
    accuracy()