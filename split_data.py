import pandas as pd

# Loading dataset
data = pd.read_csv("retail_sales.csv")

# Convert date column to datetime
data['date'] = pd.to_datetime(data['date'])

# Sort by date (VERY IMPORTANT for time series)
data = data.sort_values(by='date')

print(data.head())
print(data.info())

#Splitting dataset
split_index = int(len(data) * 0.8)


train = data.iloc[:split_index]


test = data.iloc[split_index:]

print("Train size:", train.shape)
print("Test size:", test.shape)

#Saving dataset
train.to_csv("train.csv", index=False)
test.to_csv("test.csv", index=False)