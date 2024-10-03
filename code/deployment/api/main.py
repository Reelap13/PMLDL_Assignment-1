
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

model = joblib.load('model.joblib')

class InputData(BaseModel):
    features: list[float]

@app.post('/predict')
def predict(data: InputData):
    features = np.array(data.features).reshape(1, -1)
    prediction = model.predict(features)
    return {'prediction': int(prediction[0])}