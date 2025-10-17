import pandas as pd

cash_stocks = pd.read_csv("csv_files\cash_stocks.csv")
all_stock_data = pd.read_csv("csv_files\DATA_output.csv")


def get_instrument_token(symbol):
    match = cash_stocks[cash_stocks["tradingsymbol"] == symbol]
    if len(match) > 0:
        return str(match.iloc[0]["instrument_token"])
    else:
        return None


def stock_data(Symbol, Date=None, Time=None):
    col_date = "date" if "date" in all_stock_data.columns else "Date"
    col_time = "time" if "time" in all_stock_data.columns else "Time"
    if "symbol" not in all_stock_data.columns:
        return "None"
    df = all_stock_data[all_stock_data["symbol"].str.upper() == Symbol.upper()]
    if Date:
        df = df[df[col_date] == Date]

    if Time is not None:
        df = df[df[col_time].astype(int) == int(Time)]

    if not df.empty:
        return df.iloc[0][["Open", "High", "Low", "Close", "Volume", "time"]].to_dict()
    else:
        return None


# print(all_stock_data.columns)
# data = stock_data(all_stock_data)
