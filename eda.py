import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv("train.csv")
data['date'] = pd.to_datetime(data['date'])

# Total sales per day
daily_sales = data.groupby('date')['sales'].sum()

plt.figure(figsize=(10,5))
plt.plot(daily_sales)

plt.title("Daily Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.show()