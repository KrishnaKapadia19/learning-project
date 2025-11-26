import yfinance as yf, os, pandas as pd
import numpy as np

symbols = ["TATAMOTORS.NS", "SBIN.NS"]


for symbol in symbols:
    data = yf.download(symbol, interval="1h", period="3mo")
    data.reset_index(inplace=True)

    data["date"] = data["Datetime"].dt.date
    data["time"] = data["Datetime"].dt.time
    data.drop(columns=["Datetime"], inplace=True)
    data["symbol"] = len(data) * [symbol.replace(".NS", "")]
    if isinstance(data.columns, pd.MultiIndex):
        new_col = []
        for col in data.columns:
            if col[0] != "":
                new_col.append(col[0])
            else:
                new_col.append(col[1])
        data.columns = new_col

    print(data.head())
    print(data.columns)
    formatted_data = data[
        ["symbol", "date", "time", "Open", "High", "Low", "Close", "Volume"]
    ]
    formatted_data.rename(
        columns={
            "symbol": "symbol",
            "date": "date",
            "time": "time",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
            "Close": "Close",
            "Volume": "Volume",
        },
        inplace=False,
    )
    print(formatted_data.head())

    # formatted_data['Ticker'] = symbols

    filename = f"{symbol.replace('.NS','')}.csv"
    formatted_data.to_csv(filename, index=False)
    print(f"File created: {filename}")

    directory = os.getcwd()
    df = pd.read_csv(filename)
    df["date"] = pd.to_datetime(df["date"])
    # print(len(df))
    unique_dates = df[
        "date"
    ].unique()  # Replace 'date_column' with your actual column name
    print(unique_dates)

    for i in unique_dates:
        i = pd.to_datetime(i).date()
        year = str(i.year)
        month = i.strftime("%b")
        day = f"{i.day:02d}"
        folder_path = os.path.join(directory, year, month, day, symbol)

        if os.path.exists(folder_path):
            day_data = df[df["date"].dt.date == i]
            file_path = os.path.join(folder_path, f"{i}.csv")
            day_data.to_csv(file_path, index=False)
            print(f"CSV file saved at: {file_path}")
            print(f"{folder_path} exists")

        else:
            os.makedirs(folder_path)
            day_data = df[df["date"].dt.date == i]
            file_path = os.path.join(folder_path, f"{i}.csv")
            day_data.to_csv(file_path, index=False)
            print(f"CSV file saved at: {file_path}")
            print(f"{folder_path} created")
