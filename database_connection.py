import mysql.connector

conn = mysql.connector.connect(
    # host="localhost", user="root", password="Krishna@794", database="Stock_DB"
    host="122.176.143.66", user="nikhil", password="welcome@123", database="historicaldb"
    
)

cursor = conn.cursor()


# ----Create table---------------------

# cursor.execute("DROP TABLE IF EXISTS userstock")

# ----Create table---------------------

# create_table_query = """
# CREATE TABLE IF NOT EXISTS UserStock    (
#     id INT AUTO_INCREMENT PRIMARY KEY,  -- unique row ID
#     Stock_name VARCHAR(50),
#     date INT,
#     time INT,
#     Open FLOAT,
#     High FLOAT,
#     Low FLOAT,
#     Close FLOAT,
#     Volume BIGINT
# )
# """
# cursor.execute(create_table_query)

# print("Table created successfully!")


# ------fetch data---------------------------------------

# query = """
# SELECT * FROM sbin_future
# WHERE date = 220201
# """

# cursor.execute(query)

import pandas as pd

query = "SELECT * FROM sbin_future WHERE date = 220201"
cursor.execute(query)

columns = [desc[0] for desc in cursor.description]
df_db = pd.DataFrame(cursor.fetchall(), columns=columns)

# Save for later comparison
df_db.to_feather("db_output.feather")
print("✅ Database data saved to db_output.feather")

# ------ADD MEW COLUMN---------------------------------------

# ✅ Add new column `time`
# alter_query = """
# ALTER TABLE Stock
# ADD COLUMN time VARCHAR(100);
# """
# print("✅ Column 'time' added successfully!")
# cursor.execute(alter_query)


# -------insert query--------------------------------------------

# insert_query = """
# insert into stock (Stock_name, Open, High, Low, Close, Volume)
# VALUES (%s, %s, %s, %s, %s, %s)
# """
# cursor.execute(insert_query, stock_data)


# -------delete query--------------------------------------------

# cursor.execute("DROP TABLE IF EXISTS UserStock")

# -------remove data query--------------------------------------------

# cursor.execute("TRUNCATE TABLE UserStock")

conn.commit()
# print("Table droped successfully!")
print("Table trunket successfully!")

# --- Close ---
cursor.close()
conn.close()
