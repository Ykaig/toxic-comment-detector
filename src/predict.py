# src/predict.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pathlib
from typing import List

# --- Global objects ---
# Load the trained artifacts once when the API starts
# This is much more efficient than loading them on every request.
artifacts_path = pathlib.Path("artifacts")
vectorizer = joblib.load(artifacts_path / "vectorizer.pkl")
model = joblib.load(artifacts_path / "model.pkl")


# --------------------

# Defines the input data model using Pydantic
class InputData(BaseModel):
    text: str


# Defines the output data model
class OutputData(BaseModel):
    toxic: float
    severe_toxic: float
    obscene: float
    threat: float
    insult: float
    identity_hate: float


# Create the FastAPI app instance
app = FastAPI(
    title="Toxic Comment Classifier - Real API",
    description="API that uses a trained TF-IDF and Logistic Regression model to classify comments.",
    version="1.0.0"
)


@app.get("/")
def read_root():
    return {"status": "API is running. Use the /predict endpoint for predictions."}


@app.post("/predict", response_model=OutputData)
def predict(data: InputData):
    """
    Real prediction endpoint.
    Accepts text input, transforms it using the loaded vectorizer,
    and returns the model's prediction probabilities.
    """
    print(f"Received input text: '{data.text}'")

    # The input text must be in a list or iterable for the vectorizer
    text_to_vectorize = [data.text]

    # 1. Vectorize the input text
    vectorized_text = vectorizer.transform(text_to_vectorize)

    # 2. Get prediction probabilities from the model
    # The result is a list of arrays, one for each class
    prediction_probabilities = model.predict_proba(vectorized_text)

    # 3. Format the output to match the Pydantic model
    # We take the first (and only) element from the prediction result
    output = {
        "toxic": prediction_probabilities[0][0],
        "severe_toxic": prediction_probabilities[0][1],
        "obscene": prediction_probabilities[0][2],
        "threat": prediction_probabilities[0][3],
        "insult": prediction_probabilities[0][4],
        "identity_hate": prediction_probabilities[0][5]
    }

    print(f"Returning real prediction: {output}")
    return output

# To run the API locally:
# uvicorn src.predict:app --reload