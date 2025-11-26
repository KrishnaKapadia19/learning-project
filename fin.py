import yfinance as yf
import pandas as pd

data = yf.download("TATAMOTORS.NS", interval="5m", period="7d")


data.reset_index(inplace=True)

if isinstance(data.columns, pd.MultiIndex):

    new_cols = []
    for col in data.columns:
        if col[0] != "":
            new_cols.append(col[0])
        else:
            new_cols.append(col[1])
    data.columns = new_cols

print("COLUMNS:", data.columns.tolist())
data["Datetime"] = data["Datetime"].dt.tz_convert("Asia/Kolkata")

data["Ticker"] = "TATAMOTORS.NS"

data["date"] = data["Datetime"].dt.date
data["time"] = data["Datetime"].dt.time

renamed_data = data[
    ["date", "time", "Ticker", "Open", "High", "Low", "Close", "Volume"]
]
renamed_data.rename(
    columns={
        # 'index':'index',
        # 'Datetime':'Datetime',
        "date": "date",
        "time": "time",
        "tickers": "tickers",
        "Open": "Open",
        "High": "High",
        "Low": "Low",
        "Close": "Close",
        "Volume": "Volume",
    }
)


renamed_data.to_csv("output.csv", index=False)

print(renamed_data.head(3))
# import datetime

# def time_to_seconds_from_9am(time_str):
#     # Convert string to datetime object (24-hour format)
#     time_obj = datetime.datetime.strptime(time_str, "%H:%M:%S")
#     # Reference start time 9:00 AM
#     start_seconds = 9 * 3600
#     # Total seconds from midnight
#     total_seconds = time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second
#     # Seconds from 9 AM
#     seconds_from_9am = total_seconds - start_seconds
#     return seconds_from_9am

# print(time_to_seconds_from_9am("09:00:01"))  # 0
# print(time_to_seconds_from_9am("10:00"))  # 3600
# print(time_to_seconds_from_9am("15:30"))
