# dict_1 = {
#     "name": ["sbin", "sbin", "sbin", "sbin", "sbin"],
#     "time": ["10:00", "11:00", "12:00", "13:00", "14:00"],
#     "o": [100, 98, 102, 105, 108],
#     "h": [101, 98, 102, 108, 107],
#     "l": [102, 98, 102, 102, 104],
#     "c": [103, 98, 102, 103, 108],
# }

# import pandas as pd

# df = pd.DataFrame(dict_1)
# # print(df)

# for i in range(len(df)):
#     if i == 0:
#         continue
#     print(
#         df["time"].iloc[i],
#         df["time"].iloc[i - 1],
#     )

import pandas as pd
import mysql.connector
import logging, os

# df = pd.read_feather(r"datafolder\sbin_future.feather")
# print(df.head())
# def db_fether_compare(x:) -> :

#     # Setup logging
#     logging.basicConfig(filename="compare_log.txt", level=logging.INFO, format="%(asctime)s - %(message)s")

#     # Connect to DB once
#     conn = mysql.connector.connect(
#         host="122.176.143.66", user="nikhil", password="welcome@123", database="historicaldb"
#     )

#     # --- Folder containing feather files ---
#     folder_path = "datafolder"

#     # --- Loop over all feather files ---
#     for file_name in os.listdir(folder_path):
#         if file_name.endswith(".feather"):
#             file_path = os.path.join(folder_path, file_name)
#             table_name = file_name.replace(".feather", "") # get table name
#             df_feather = pd.read_feather(file_path)
#             fether_col=sorted(df_feather.columns.to_list())

#             print(f"\n📄 Checking file: {file_name}  (Table: {table_name})")

#         cursor = conn.cursor()
#         query = f"SELECT * FROM {table_name} WHERE date = 230105 "
#         cursor.execute(query)

#         columns = [desc[0] for desc in cursor.description]
#         df_db = pd.DataFrame(cursor.fetchall(), columns=columns)
#         print(df_db.to_csv(f"sample\{file_name}.fether"))
#         # Save for later comparison
#         # df_db.to_feather(f"db_{df_feather}.feather")

#         fether_col = sorted(df_feather.columns.to_list())
#         # print(fether_col)
#         db_col = sorted(df_db.columns.to_list())
#         # print(db_col)

#         if len(df_db)==(len(df_feather)) and fether_col==(db_col):
#             print("same")
#             logging.info("same")
#         else:
#             print(f"Feather file column {sorted(df_feather.columns.to_list())} \n rows : {len(df_db)}, \n database file column are {sorted(df_db.columns.to_list())} \n rows : {len(df_feather)}")
#             logging.info(f"Feather file column {sorted(df_feather.columns.to_list())} \n rows : {len(df_db)}, database file column are {sorted(df_db.columns.to_list())} rows : {len(df_feather)}")
#             continue

# print(db_fether_compare())

# nested = {"a": {"x": 1}, "b": {"y": 2}}
# flat = {}

# for outer_k, inner_dict in nested.items():
#     for inner_k, value in inner_dict.items():
#         flat[f"{outer_k}_{inner_k}"] = value

# print(flat)

# folder_name = r"datafolder"
# input_symbol=input("Enter name of symbol: ")
# file_name = f"{input_symbol.lower()}_cash.feather"

# file_path = os.path.join(folder_name, file_name)


# file_path = rf"{folder_name}\{file_name}"
# if not os.path.exists(file_path):
#     print(f"File not found for {file_name}")

# df = pd.read_feather(file_path)
# df=df.head(5)
# # df = df.head(10)
# nested_dict = {}

folder_name = r"datafolder"
input_symbol = input("Enter name of symbol: ")

temp_dict={}
nested_dict = {}
logger = logging.getLogger()
logging.basicConfig(
    filename="test.txt", level=logging.INFO, format="%(asctime)s - %(message)s"
)

for file in os.listdir(folder_name):
    if file.endswith(".feather") and file.lower().startswith(input_symbol):
        file_path = os.path.join(folder_name, file)

        # read only 1st row
        df = pd.read_feather(file_path)
        df=df.head(1000)

        symbol = file.replace(".feather", "")  # file base name
        base_symbol = symbol.split("_")[0]  # expiry e.g. acc from acc_cash
        # instrument = symbol
        unique_symbols = df['symbol'].unique()
        # print(sorted(unique_symbols))
    
        for _, row in df.iterrows():
            indices_ist = str(row["symbol"]) if "expiry" in row.index else None
            date = str(row["date"]).split(".")[0]
            # create base_symbol key if not exists
            if base_symbol not in nested_dict:
                nested_dict[base_symbol] = {}
            # create date key if not exists
            if date not in nested_dict[base_symbol]:
                nested_dict[base_symbol][date] = {}

            # print(nested_dict[base_symbol][date])
            # decide key
            key = indices_ist if indices_ist else unique_symbols
            
            if key not in nested_dict[base_symbol][date]:
                nested_dict[base_symbol][date][key] = []
            # temp_dict = {base_symbol:{date:{key}}}
            # print(temp_dict)

            # nested_dict[symbol][date][instrument].setdefault(time, {})

            row_values = [
                int(row["time"]),
                int(row["open"]),
                int(row["high"]),
                int(row["low"]),
                int(row["close"]),
            ]
            if "volume" in row and "oi" in row:
                row_values.append(row["volume"])
                row_values.append(row["oi"])

            nested_dict[base_symbol][date][key].append(row_values)
            # Append value in key
            # k=nested_dict[key].append(row_values)
            # print(k)
            temp_dict={base_symbol: {date: {key: nested_dict[base_symbol][date][key]}}}
    print(temp_dict)
    logger.info(f"{temp_dict}")
