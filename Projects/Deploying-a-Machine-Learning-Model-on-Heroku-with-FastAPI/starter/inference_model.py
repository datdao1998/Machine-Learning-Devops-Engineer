"""
Contains a function to make batch or single prediction and
write predictioss into a file 
"""
from joblib import load
from starter.ml.data import process_data
from starter.ml.model import inference

def infer(data, cat_features):
    """
    Load model and run inference
    Parameters
    ----------
    root_path
    data
    cat_features
    Returns
    -------
    prediction
    """ 
    
    model = load("model/model.joblib")
    encoder = load("model/encoder.joblib")
    lb = load("model/lb.joblib")

    X, _, _, _ = process_data(data, categorical_features=cat_features, encoder=encoder, lb=lb, training=False)
    print('X shape', X.shape)
    pred = inference(model, X)
    prediction = lb.inverse_transform(pred)[0]

    return prediction