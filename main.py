
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fastapi import FastAPI
from pydantic import BaseModel, Field


# Find model.joblib relative to this file
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

# Load the trained pipeline once when the API starts
model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="Diabetes Prediction API",
    description="Predicts a diabetes class using a trained ML model.",
    version="1.0.0"
)


# Define the expected input structure
class DiabetesInput(BaseModel):
    Pregnancies: int = Field(ge=0)
    Glucose: float = Field(ge=0)
    BloodPressure: float = Field(ge=0)
    SkinThickness: float = Field(ge=0)
    Insulin: float = Field(ge=0)
    BMI: float = Field(ge=0)
    DiabetesPedigreeFunction: float = Field(ge=0)
    Age: int = Field(gt=0)


@app.get("/")
def home():
    return {
        "message": "Diabetes Prediction API is running",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True
    }


@app.post("/predict")
def predict(data: DiabetesInput):
    # Convert validated request data into a DataFrame
    input_data = pd.DataFrame(
        [data.model_dump()]
    )

    # Apply the same zero handling used during training
    zero_columns = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    input_data[zero_columns] = (
        input_data[zero_columns].replace(0, np.nan)
    )

    # Get the predicted class and class probabilities
    prediction = int(model.predict(input_data)[0])
    probabilities = model.predict_proba(input_data)[0]

    return {
        "prediction": prediction,
        "label": (
            "Diabetes-positive class"
            if prediction == 1
            else "Diabetes-negative class"
        ),
        "probabilities": {
            str(int(cls)): round(float(prob), 4)
            for cls, prob in zip(model.classes_, probabilities)
        },
        "note": (
            "Educational ML demonstration only, "
            "not a medical diagnosis."
        )
    }
