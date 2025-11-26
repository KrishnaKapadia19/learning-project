d1={}
if "k1" not in d1:
    d1["k1"] = [1]

d1["k1"] = [2]
# print(d1)

import pandas as pd 

df = pd.read_feather(r"datafolder/wipro_put.feather")
df=df.head(1000)

# d1={}
# d1.append(df)

result = {}

for _, row in df.iterrows():
    symbol = row[3]  # WIPRO23FEB23350PE
    values = [
        row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11]
    ]
    
    result.setdefault(symbol, [])
    result[symbol].append(values)

print(result)