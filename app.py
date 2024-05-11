from typing import Literal
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
from pydantic import BaseModel
import pandas as pd

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load("model.pkl")


@app.get("/")
async def main():
    return {"message": "Sba7 el manga"}


class RequestBodyModel(BaseModel):
    gender: Literal["Female", "Male"]
    seniorCitizen: Literal[0, 1]
    partner: Literal["Yes", "No"]
    dependents: Literal["Yes", "No"]
    tenure: int
    phoneService: Literal["Yes", "No"]
    multipleLines: Literal["Yes", "No", "No phone service"]
    internetService: Literal["DSL", "Fiber optic", "No"]
    onlineSecurity: Literal["Yes", "No", "No internet service"]
    onlineBackup: Literal["Yes", "No", "No internet service"]
    deviceProtection: Literal["Yes", "No", "No internet service"]
    techSupport: Literal["Yes", "No", "No internet service"]
    streamingTV: Literal["Yes", "No", "No internet service"]
    streamingMovies: Literal["Yes", "No", "No internet service"]
    contract: Literal["Month-to-month", "One year", "Two year"]
    paperlessBilling: Literal["Yes", "No"]
    paymentMethod: Literal[
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    monthlyCharges: float
    totalCharges: float


cols = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


@app.post("/predict")
async def predict(body: RequestBodyModel):

    data = body.model_dump()
    input_list = ["1"] + list(data.values())

    df = pd.DataFrame([input_list], columns=cols)
    print(df)

    prediction = model.predict(df)

    print(prediction)

    return {"prediction": prediction[0].item()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
