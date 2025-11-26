# import yfinance as yf,os,pandas as pd

# symbols = ['TATAMOTORS.NS', 'SBIN.NS']

# data= yf.download('TATAMOTORS.NS',interval='1h',period='3mo')
# data.reset_index(inplace=True)


# if isinstance(data.columns,pd.MultiIndex):
#     new_col=[]
#     for col in data.columns:
#         if col[0]!='':
#             new_col.append(col[0])
#         else:
#             new_col.append(col[1])
#     data.columns = new_col

# data['Ticker'] ='TATAMOTORS.NS'
# # data['Ticker'] = 'SBIN.NS',
# data['date'] = data['Datetime'].dt.date
# data['time'] = data['Datetime'].dt.time
# # data = data.ffill()
# formatted_data = data[['date', 'time', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume']]
# formatted_data.rename(columns={
#     'date':'date',
#     'time':'time',
#     'Ticker': 'Ticker',
#     'Open': 'Open',
#     'High': 'High',
#     'Low': 'Low',
#     'Close': 'Close',
#     'Volume': 'Volume',
# }, )

# formatted_data.to_csv('csv_files\kk.csv', index=False)


# print(data.head())
# print(data.tail())

# directory= os.getcwd()
# df = pd.read_csv('csv_files\kk.csv')
# df['date'] = pd.to_datetime(df['date'])
# # print(len(df))
# unique_dates = df['date'].unique()

# for i in unique_dates:
#     i= pd.to_datetime(i).date()
#     year = str(i.year)
#     month = i.strftime("%b")
#     day = f"{i.day:02d}"
#     folder_path = os.path.join(directory, year,month,day)


#     if os.path.exists(folder_path):
#         day_data = df[df['date'].dt.date == i]
#         file_path=os.path.join(folder_path, f"csv_files\{i}.csv")
#         day_data.to_csv(file_path,index = False)
#         print(f'{folder_path} exists')

#     else:
#         os.makedirs(folder_path)
#         day_data = df[df['date'].dt.date == i]
#         file_path=os.path.join(folder_path, f"csv_files\{i}.csv")
#         day_data.to_csv(file_path,index = False)
#         print(f"CSV file saved at: {file_path}")
#         print(f'{folder_path} created')


# user_id = data["user_id"]
# symbol = data["symbol"].upper()

# stock_data = yf.download(symbol + ".NS", interval="1m", period="7d")
# if stock_data.empty:
#     return jsonify({"error": f"No data for {symbol}"}), 440

# stock_data.reset_index(inplace=True)
# stock_data["Datetime"] = stock_data["Datetime"].dt.tz_convert("Asia/Kolkata")
# stock_data["date"] = stock_data["Datetime"].dt.date.apply(
#     lambda x: int(x.strftime("%y%m%d"))
# )
# stock_data["time"] = (
#     stock_data["Datetime"].dt.hour * 3600
#     + stock_data["Datetime"].dt.minute * 60
#     + stock_data["Datetime"].dt.second
# )

# stock_data = stock_data[
#     (stock_data["time"] >= 33300) & (stock_data["time"] <= 55800)
# ]

# records = [
#     (
#         user_id,
#         symbol,
#         float(row["Open"]),
#         float(row["High"]),
#         float(row["Low"]),
#         float(row["Close"]),
#         int(row["Volume"]),
#         int(row["time"]),
#         int(row["date"]),
#     )
#     for _, row in stock_data.iterrows()
# ]

# if records:
#     conn = mysql.connector.connect(
#         host="localhost", user="root", password="Krishna@794", database="Stock_DB"
#     )
#     cursor = conn.cursor()
#     insert_query = """
#     INSERT INTO UserStock (user_id, Stock_name, Open, High, Low, Close, Volume, time, date)
#     VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
#     """
#     cursor.executemany(insert_query, records)
#     conn.commit()
#     cursor.close()
#     conn.close()

# return jsonify(
#     {"message": f"{len(records)} rows for {symbol} stored for user {user_id}"}
# )
#------------------------------
import pandas as pd
# import os

# def read_stock_file(symbol, year="2023", month="JAN", day="02"):
#     # Construct file path dynamically
#     base_path = "dataset"
#     file_name = f"{symbol.lower()}_cash.feather"
#     file_path = os.path.join(base_path, year, month, day, file_name)

#     # Check file existence and read
#     if not os.path.exists(file_path):
#         return f"File not found: {file_path}"

#     df = pd.read_feather(file_path)
#     return df

# # Example usage
# symbol = input("Enter symbol: ").strip().lower()  # e.g., abb
# df = read_stock_file(symbol)
# print(df.head())


# -------------

df = pd.read_feather(r"dataset\2023\JAN\06\banknifty_call.feather")
print(df.columns)
# print(df.to_csv("banknify_call.csv"))
print("created")


