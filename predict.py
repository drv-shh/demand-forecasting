import joblib
import pandas as pd

# load trained model
model = joblib.load("demand_forecast_model.pkl")

# example input
sample = pd.DataFrame({
    "store_id":[1],
    "item_id":[10],
    "price":[20.5],
    "promo":[1],
    "weekday":[5],
    "month":[7],
    "lag_1":[30],
    "lag_7":[28],
    "rolling_mean":[29]
})

prediction = model.predict(sample)

print("Predicted Sales:", prediction[0])