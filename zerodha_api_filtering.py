import pandas as pd


def get_indices(df):

    print(df.columns)
    indices_df = df[df["segment"] == "INDICES"]
    indices_df.to_csv("csv_files\indices.csv", index=False)
    return "Indices saved to indices.csv"


def get_cash_stock_instrument(df):
    cash_stocks = df[(df["segment"] == "NSE") & (df["instrument_type"] == "EQ")]
    return cash_stocks.to_csv("csv_files\cash_stocks.csv", index=False)


def get_instrument_token(df):
    user_inputed_symbol = input("Enter the name of the stock : ").upper()
    cash_stocks = pd.read_csv("csv_files\cash_stocks.csv")
    match = cash_stocks[cash_stocks["tradingsymbol"] == user_inputed_symbol.upper()]
    if len(match) > 0:
        return match.iloc[0]["instrument_token"]


# def get_instrument_token(symbol):
#     # user_inputed_symbol = input("Enter the name of the stock : ").upper()
#     cash_stocks = pd.read_csv("csv_files\cash_stocks.csv")
#     match = cash_stocks[cash_stocks["tradingsymbol"].str.upper() == symbol.upper()]
#     if len(match) > 0:
#         return match.iloc[0]["instrument_token"]


def filter_instruments(df):
    indices = get_indices(df)
    cash_stock_instrumet = get_cash_stock_instrument(df)
    instrument_token = get_instrument_token(df)
    return indices, cash_stock_instrumet, instrument_token


df = pd.read_csv("csv_files\instruments.csv")
x, y, z = filter_instruments(df)
# print(filter_instruments(df))
print(x, "\n", y, "\n", z)
