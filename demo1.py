import pandas as pd
df = pd.read_feather(r"dataset\2023\JAN\12\banknifty_put.feather")
if df.to_csv("zebra.csv"):
    True


