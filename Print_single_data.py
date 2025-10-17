# import os

# # Specify the folder name
# folder_name = "my_folder"

# # Create the folder
# os.makedirs(folder_name, exist_ok=True)  # `exist_ok=True` avoids error if folder exists
# print(f"Folder '{folder_name}' created successfully!")

# import yfinance as yf

# symbols = ['AAPL', 'GOOGL', 'MSFT']

# def download_data():
#     for symbol in symbols:
#         data = yf.download(symbol, start='2020-01-01', end='2023-01-01')
#         data.to_csv(f'{symbol}_data.csv')

# if __name__ == "__main__":
#     download_data()

import yfinance as yf
import os
from datetime import datetime

# Define the ticker symbol
ticker = "F"  # Ford Motor Company (you can change this to any other symbol)

# Download historical data
data = yf.download(ticker, start="2024-01-01", end="2024-01-02", interval="1d")

# Define folder structure
year = "2024"
month = ["JAN", "feb", "MAR,", "APR", "MAY", "JUNE", "JULY"]
day = "day"
folder_path = os.path.join(year, month, day)

# Create directories if they don't exist
os.makedirs(folder_path, exist_ok=True)

# Define file name and save path
file_name = "csv_files\datamotor.csv"
file_path = os.path.join(month, file_name)

# Save data to CSV
data.to_csv(month)

print(f"Data saved to {month}")
