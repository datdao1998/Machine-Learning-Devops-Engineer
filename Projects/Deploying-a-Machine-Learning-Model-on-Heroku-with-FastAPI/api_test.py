
"""
Api servermodule test
"""
import pytest
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    """
    Get dataset
    """
    api_client = TestClient(app)
    return api_client


def test_get(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json() == {"Message": "Hello!"}


def test_get_malformed(client):
    r = client.get("/wrong_url")
    assert r.status_code != 200


def test_post_high_salary(client):
    r = client.post("/inference", json={
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
    })
    assert r.status_code == 200
    assert r.json() == {"prediction": ">50K"}


def test_post_low_salary(client):
    r = client.post("/inference", json={
        "age": 19,
        "workclass": "Private",
        "fnlgt": 0,
        "education": "HS-grad",
        "marital_status": "Never-married",
        "occupation": "Other-service",
        "relationship": "Own-child",
        "race": "Black",
        "sex": "Male",
        "hoursPerWeek": 40,
        "nativeCountry": "United-States"
    })
    assert r.status_code == 200
    assert r.json() == {"prediction": "<=50K"}
