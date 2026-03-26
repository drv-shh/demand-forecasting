import pandas as pd

# Loading dataset
data = pd.read_csv("train.csv")
data['date'] = pd.to_datetime(data['date'])

print(data.head())
data.info()