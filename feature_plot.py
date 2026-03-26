import joblib
import matplotlib.pyplot as plt

model = joblib.load("demand_forecast_model.pkl")

importance = model.feature_importances_

features = [
    "store_id",
    "item_id",
    "price",
    "promo",
    "weekday",
    "month",
    "lag_1",
    "lag_7",
    "rolling_mean"
]

plt.barh(features, importance)
plt.title("Feature Importance")
plt.show()