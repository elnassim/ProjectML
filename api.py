from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()
model = joblib.load("model_selection.joblib")

class InputData(BaseModel):
    age: int
    area: str
    gender: str
    education: str
    years_experience: int
    marital_status: str
    socio_professional_group: str

@app.post("/predict")
def predict(data: InputData):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    return {"predicted_income": round(prediction, 2)}