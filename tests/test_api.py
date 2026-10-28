
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_predict_valid_input():
    payload = {
        "Pregnancies": 2,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 80,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 30
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "label" in data
    assert "probabilities" in data


def test_predict_missing_feature():
    payload = {
        "Pregnancies": 2,
        "Glucose": 120
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_invalid_input():
    payload = {
        "Pregnancies": -1,
        "Glucose": 120,
        "BloodPressure": 70,
        "SkinThickness": 25,
        "Insulin": 80,
        "BMI": 28.5,
        "DiabetesPedigreeFunction": 0.5,
        "Age": 30
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
