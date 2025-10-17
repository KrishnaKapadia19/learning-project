# import yfinance as yf
# import pandas as pd
# import datetime
# import os


# symbols=['TATAMOTORS.NS', 'SBIN.NS','ACC.NS','IGARASHI.NS']
# # symbols = ["TATAMOTORS.NS"]


# for i, symbol in enumerate(symbols):
#     data = yf.download(symbol, interval="1m", period="7d")
#     if data.empty:
#         print(f"No data for {symbol}, skipping...")
#         continue
#     data.reset_index(inplace=True)
#     data["Datetime"] = data["Datetime"].dt.tz_convert("Asia/Kolkata")

#     data["Ticker"] = symbol
#     data["date"] = data["Datetime"].dt.date
#     data["date"] = data["date"].apply(lambda x: int(x.strftime("%y%m%d")))
#     data["time"] = data["Datetime"].dt.time
#     data["seconds"] = (
#         data["Datetime"].dt.hour * 3600
#         + data["Datetime"].dt.minute * 60
#         + data["Datetime"].dt.second
#     )
#     data = data[(data["seconds"] >= 33300) & (data["seconds"] <= 55800)]
#     data.drop(columns=["Datetime"], inplace=True)
#     data["symbol"] = len(data) * [symbol.replace(".NS", "")]
#     if isinstance(data.columns, pd.MultiIndex):
#         new_col = []
#         for col in data.columns:
#             if col[0] != "":
#                 new_col.append(col[0])
#             else:
#                 new_col.append(col[1])
#         data.columns = new_col

#     # print(data.head())
#     # print(data.columns)
#     formatted_data = data[
#         ["symbol", "date", "seconds", "Open", "High", "Low", "Close", "Volume"]
#     ]
#     formatted_data.rename(
#         columns={
#             "symbol": "symbol",
#             "date": "date",
#             "seconds": "time",
#             "Open": "Open",
#             "High": "High",
#             "Low": "Low",
#             "Close": "Close",
#             "Volume": "Volume",
#         },
#         inplace=True,
#     )
#     # print(formatted_data.head())

#     # formatted_data['Ticker'] = symbols

#     filename = "csv_files\DATA_output.csv"
#     if i == 0:
#         # First symbol: create file
#         formatted_data.to_csv(filename, index=False)
#     else:
#         # Subsequent symbols: append
#         formatted_data.to_csv(filename, mode="a", index=False, header=False)
# print(f"File created: {filename}")


import yfinance as yf
import mysql.connector
from datetime import datetime

# MySQL connection
conn = mysql.connector.connect(
    host="localhost", user="root", password="Krishna@794", database="Stock_DB"
)
cursor = conn.cursor()

# Ensure table exists (with `time` column)
create_table_query = """
CREATE TABLE IF NOT EXISTS Stock (
    id INT AUTO_INCREMENT PRIMARY KEY,
    Stock_name VARCHAR(50),
    date INT,
    time INT,
    Open FLOAT,
    High FLOAT,
    Low FLOAT,
    Close FLOAT,
    Volume BIGINT

)
"""
cursor.execute(create_table_query)

symbols = ["TATAMOTORS.NS", "SBIN.NS", "ACC.NS", "IGARASHI.NS"]

for symbol in symbols:
    data = yf.download(symbol, interval="1m", period="7d")
    if data.empty:
        print(f"No data for {symbol}, skipping...")
        continue

    data.reset_index(inplace=True)
    data["Datetime"] = data["Datetime"].dt.tz_convert("Asia/Kolkata")

    # Convert date to integer YYMMDD
    data["date"] = data["Datetime"].dt.date.apply(lambda x: int(x.strftime("%y%m%d")))

    # Convert time to seconds
    data["time"] = (
        data["Datetime"].dt.hour * 3600
        + data["Datetime"].dt.minute * 60
        + data["Datetime"].dt.second
    )

    # Filter time between 9:15 AM (33300) and 3:30 PM (55800)
    data = data[(data["time"] >= 33300) & (data["time"] <= 55800)]

    # Prepare data for DB
    records = [
        (
            symbol.replace(".NS", ""),
            float(row["Open"]),
            float(row["High"]),
            float(row["Low"]),
            float(row["Close"]),
            int(row["Volume"]),
            int(row["time"]),
            int(row["date"]),
        )
        for _, row in data.iterrows()
    ]

    if records:
        insert_query = """
        INSERT INTO Stock (Stock_name, Open, High, Low, Close, Volume, time, date)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        cursor.executemany(insert_query, records)  # bulk insert
        conn.commit()
        print(f"Inserted {len(records)} rows for {symbol}")

cursor.close()
conn.close()
print("All data inserted successfully!")
