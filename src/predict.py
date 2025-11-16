# src/predict.py
from fastapi import FastAPI
from pydantic import BaseModel


# Defines the input data model using Pydantic
# This ensures automatic validation of incoming data
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


# Creates the FastAPI application instance
app = FastAPI(
    title="Toxic Comment Classifier - Dummy API",
    description="A dummy API that respects the input/output contract.",
    version="0.0.1"
)


@app.get("/")
def read_root():
    return {"status": "API is running. Use the /predict endpoint for predictions."}


@app.post("/predict", response_model=OutputData)
def predict(data: InputData):
    """
    Dummy prediction endpoint.
    Accepts text input and always returns the same static response,
    while respecting the correct output structure.
    """
    print(f"Received input text: '{data.text}'")

    # Dummy response that follows the expected output contract
    dummy_prediction = {
        "toxic": 0.0,
        "severe_toxic": 0.0,
        "obscene": 0.0,
        "threat": 0.0,
        "insult": 0.0,
        "identity_hate": 0.0
    }

    print(f"Returning dummy prediction: {dummy_prediction}")
    return dummy_prediction

# To run the API locally:
# uvicorn src.predict:app --reload