# Training Model
from sklearn.metrics import mean_absolute_error, mean_squared_error

import numpy as np

import pandas as pd
from xgboost import XGBRegressor
import joblib

data = pd.read_csv("train.csv")
data["date"] = pd.to_datetime(data["date"])


data["store_id"] = data["store_id"].str.replace("store_", "").astype(int)

data["item_id"] = data["item_id"].str.replace("item_", "").astype(int)

#sort ds
data = data.sort_values(by=["store_id", "item_id", "date"])

#last day
data["lag_1"] = data.groupby(["store_id", "item_id"])["sales"].shift(1)
# last week
data["lag_7"] = data.groupby(["store_id", "item_id"])["sales"].shift(7)


data["rolling_mean"] = (
    data.groupby(["store_id", "item_id"])["sales"]
    .shift(1)
    .rolling(7)
    .mean()
)
data = data.dropna()

features = ["store_id","item_id","price","promo","weekday","month","lag_1","lag_7","rolling_mean"]


target = "sales"
X = data[features]
y = data[target]

print("Feature dataset shape:", X.shape)
print("Target dataset shape:", y.shape)

# XG B model
model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    random_state=42
)
model.fit(X, y)

prediction = model.predict(X)

mae = mean_absolute_error(y, prediction)
rmse = np.sqrt(mean_squared_error(y, prediction))

print("MAE:", mae)
print("RMSE:", rmse)

joblib.dump(model, "demand_forecast_model.pkl")
print("Model saved successfully!")