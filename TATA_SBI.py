import yfinance as yf, os, pandas as pd

# data= yf.download(['TATAMOTORS.NS', 'SBIN.NS'],interval='1h',period='3mo')

# data.reset_index(inplace=True)

# data['date'] = data['Datetime'].dt.date
# data['time'] = data['Datetime'].dt.time

# data.drop(columns=['Datetime'], inplace=True)

# if isinstance(data.columns,pd.MultiIndex):
#     new_col=[]
#     for col in data.columns:
#         if col[0]!='':
#             new_col.append(col[0])
#         else:
#             new_col.append(col[1])
#     data.columns = new_col

# data['TATAMOTORS'] ='TATAMOTORS.NS'
# data['SBIN'] ='SBIN.NS'
# # print(data.head())
# # print(data.tail())
# print(data.columns)

# data['date'] = pd.to_datetime(data['date'])
# formatted_data = data[[ 'date', 'time','TATAMOTORS','SBIN','Open', 'High', 'Low', 'Close', 'Volume']]
# formatted_data.rename(columns={
#     # 'Datetime':'Datetime',
#     'date':'date',
#     'time':'time',
#     'TATAMOTORS': 'TATAMOTORS',
#     'SBIN': 'SBIN',
#     'Open': 'Open',
#     'High': 'High',
#     'Low': 'Low',
#     'Close': 'Close',
#     'Volume': 'Volume',
# }, )
# data.to_csv('TATA_SBI.csv', index=False)
# print(f"file created 'TATA_SBI.csv'")


directory = os.getcwd()
df = pd.read_csv("csv_files\TATA_SBI.csv")
df["date"] = pd.to_datetime(df["date"])
# print(len(df))
unique_dates = df["date"].unique()  # Replace 'date_column' with your actual column name
print(unique_dates)

for i in unique_dates:
    i = pd.to_datetime(i).date()
    year = str(i.year)
    month = i.strftime("%b")
    day = f"{i.day:02d}"
    folder_path = os.path.join(directory, year, month, day)

    if os.path.exists(folder_path):
        day_data = df[df["date"].dt.date == i]
        file_path = os.path.join(folder_path, f"csv_files\{i}.csv")
        day_data.to_csv(file_path, index=False)
        print(f"CSV file saved at: {file_path}")
        print(f"{folder_path} exists")

    else:
        os.makedirs(folder_path)
        day_data = df[df["date"].dt.date == i]
        file_path = os.path.join(folder_path, f"csv_files\{i}.csv")
        day_data.to_csv(file_path, index=False)
        print(f"CSV file saved at: {file_path}")
        print(f"{folder_path} created")
