from fastapi import FastAPI
import joblib
import pandas as pd

# initialize API
app = FastAPI()

# load trained model
model = joblib.load("demand_forecast_model.pkl")

@app.get("/")
def home():
    return {"message": "Demand Forecast API Running"}

@app.post("/predict")
def predict_sales(
    store_id: int,
    item_id: int,
    price: float,
    promo: int,
    weekday: int,
    month: int,
    lag_1: float,
    lag_7: float,
    rolling_mean: float
):

    # convert input into dataframe
    data = pd.DataFrame({
        "store_id":[store_id],
        "item_id":[item_id],
        "price":[price],
        "promo":[promo],
        "weekday":[weekday],
        "month":[month],
        "lag_1":[lag_1],
        "lag_7":[lag_7],
        "rolling_mean":[rolling_mean]
    })

    prediction = model.predict(data)

    return {"predicted_sales": float(prediction[0])}