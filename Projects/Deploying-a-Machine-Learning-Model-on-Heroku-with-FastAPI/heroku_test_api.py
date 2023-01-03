import requests
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")


features = {
        "age": 30,
        "workclass": "State-gov",
        "fnlgt": 141297,
        "education": "Bachelors",
        "marital_status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "Asian-Pac-Islander",
        "sex": "Male",
        "hoursPerWeek": 40,
        "nativeCountry": "India"
    }


app_url = "https://deploy-ml-pipeline.herokuapp.com/inference"

r = requests.post(app_url, json=features)
assert r.status_code == 200

logging.info(f"Status code: {r.status_code}")
logging.info(f"Response body: {r.json()}")