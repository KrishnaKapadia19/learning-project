import requests
import pandas as pd

url = "https://api.kite.trade/instruments"
response = requests.get(url)

with open("csv_files\instruments.csv", "wb") as f:
    f.write(response.content)

df = pd.read_csv("csv_files\instruments.csv")
print(df.head())
print(f"Total instruments: {len(df)}")
