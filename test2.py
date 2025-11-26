import pandas as pd
from enum import Enum
import os
import logging
import os

from typing import Dict, List

dataset_folder_name = "dataset"


class FileSegmentType(Enum):
    CASH = "_cash"
    CALL = "_call"
    PUT = "_put"


month_map = {"01": "JAN"}

cwd = os.getcwd()
dataset_path = os.path.join(cwd, dataset_folder_name)


def seconds_to_hms(seconds: int) -> str:
    if seconds > 86399:
        raise ValueError(f"{seconds} is not valid please enter less then 86399")
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


#  --- stroing data in file format----
def load_mapper(
    maper_dict: Dict[int, Dict[str, List[float]]],  # date:symbol:list[data]
    symbol_names: list[str],
    start_date: int,
    end_date: int,
):
    for current_date in range(start_date, end_date + 1):
        if current_date not in maper_dict:
            maper_dict[current_date] = {}
        for underlying_name in symbol_names:
            for file_type in FileSegmentType:
                file_name = f"{underlying_name.lower()}{file_type.value}.feather"
                year = f"20{str(current_date)[:2]}"
                month = month_map[str(current_date)[2:4]]
                day = str(current_date)[4:]
                file_path = os.path.join(dataset_path, year, month, day, file_name)

                if not os.path.exists(file_path):
                    continue

                df = pd.read_feather(file_path)
                for i in range(len(df)):
                    date = df["date"].iloc[i]
                    # time = seconds_to_hms(int(df["time"].iloc[i]))
                    time = int(df["time"].iloc[i])
                    symbol = df["symbol"].iloc[i]
                    open = float(df["open"].iloc[i])
                    high = float(df["high"].iloc[i])
                    low = float(df["low"].iloc[i])
                    close = float(df["close"].iloc[i])

                    if symbol not in maper_dict[current_date]:
                        maper_dict[current_date][symbol] = []
                    maper_dict[current_date][symbol].append(
                        [time, open, high, low, close]
                    )
            # print("done for the filename, date", file_name, current_date)


maper_dict: Dict[int, Dict[str, List[float]]] = {}
start_date = 230101
end_date = 230104
load_mapper(maper_dict, ["banknifty"], start_date, end_date)

# print(maper_dict[230102]["BANKNIFTY05JAN2343000CE"], len(maper_dict[230102]["BANKNIFTY05JAN2343000CE"]))


def nearest_expiry_price(
    maper_dict: dict,
    action_price: float,
    action_time: int,
    time_window_sec: int,
    target_price: int,
):

    for date, symbol_dict in maper_dict.items():
        closest_record = {}
        target_record = {}
        momentum_record = {}
        closest_diff = float("inf")
        for symbol, rows in symbol_dict.items():

            # Only CALL options
            # if "CE" not in symbol:
            #     continue

            for row in rows:
                time, op, hi, lo, cl = row

                if abs(time - action_time) > time_window_sec:
                    continue
                # Find how close close price is to action price
                action_diff = abs(cl - action_price)

                # Update only if this candle is closer to 30
                if action_diff < closest_diff:
                    closest_diff = action_diff
                    closest_record = {
                        "date": date,
                        "symbol": symbol,
                        "time": time,
                        "open": op,
                        "high": hi,
                        "low": lo,
                        "close": cl,
                    }
            if not closest_record:
                continue

        # Output result
        if not closest_record:
            continue

        symbol = closest_record["symbol"]
        # entry_price = closest_record["close"]
        # target_price = entry_price + (0.05* entry_price)
        # stoploss_price = entry_price - (0.05 * entry_price)
        # exit_time = 15 * 3600 + 15 * 60
        # target_record = {}
        # stoploss_record = {}
        # exit_record={}

        # --- for target---
        for row in maper_dict[date][symbol]:
            time, op, high, low, close = row
            # print(closest_record["close"]+2)
            if op >= closest_record["close"] + 2 and time > closest_record["time"]:
                momentum_record = {
                    "date": date,
                    "symbol": symbol,
                    "time": time,
                    "open": op,
                    "high": high,
                    "low": low,
                    "close": close,
                }
                break

        if not momentum_record:
            print(f"{closest_record["date"]} --> No CALL AVAILABLE FOR THIS DATE")
            print("---------")
        if "time" not in momentum_record:
            continue
        entry_price = momentum_record["open"]
        target_price = entry_price + (0.05 * entry_price)
        stoploss_price = entry_price - (0.05 * entry_price)
        exit_time = 15 * 3600 + 15 * 60
        target_record = {}
        stoploss_record = {}
        exit_record = {}
        for row in maper_dict[date][symbol]:
            time, op, high, low, close = row
            if high >= target_price and time > momentum_record["time"]:
                target_record = {
                    "date": date,
                    "symbol": symbol,
                    "time": time,
                    "open": op,
                    "high": high,
                    "low": low,
                    "close": close,
                }
                break
            elif low <= stoploss_price and time > momentum_record["time"]:
                stoploss_record = {
                    "date": date,
                    "symbol": symbol,
                    "time": time,
                    "open": op,
                    "high": high,
                    "low": low,
                    "close": close,
                }
                break
            elif time == exit_time:
                exit_record = {
                    "date": date,
                    "symbol": symbol,
                    "time": time,
                    "open": op,
                    "high": high,
                    "low": low,
                    "close": close,
                }
                break
        if momentum_record:
            print(
                f"Moment trade data -->"
                f"{closest_record["date"]}"
                f"{closest_record["symbol"]}"
                f"{closest_record["time"]}"
                f"O:{momentum_record['open']} H:{momentum_record['high']} "
                f"L:{momentum_record['low']} C:{momentum_record['close']} | "
                f"Entry level at : {momentum_record['open']} \n"
            )

            if target_record:
                print(
                    f"Target data -->",
                    f"{target_record["date"]}"
                    f"{target_record["symbol"]}"
                    f"{target_record["time"]}"
                    f"O:{target_record['open']} H:{target_record['high']} "
                    f"L:{target_record['low']} C:{target_record['close']} | "
                    f"TARGET hit : exit at {target_price} \n"
                )
                print("---------")

            elif stoploss_record:
                print(
                    f"Stoploss data -->",
                    f"{stoploss_record["date"]}"
                    f"{stoploss_record["symbol"]}"
                    f"{stoploss_record["time"]}"
                    f"O:{stoploss_record['open']} H:{stoploss_record['high']} "
                    f"L:{stoploss_record['low']} C:{stoploss_record['close']} | "
                    f"STOPLOSS hit : exit at {stoploss_price} \n"
                )
                print("---------")
            elif exit_record:
                print(
                    f"No taget or stoploss hited today -->"
                    f"{exit_record["date"]}"
                    f"{exit_record["symbol"]}"
                    f"{exit_record["time"]}"
                    f"O:{exit_record['open']} H:{exit_record['high']} "
                    f"L:{exit_record['low']} C:{exit_record['close']} | "
                    f"EXIT hit at : {exit_record['close']} \n"
                )
                print("---------")


action_price = 30

action_time = 9 * 3600 + 30 * 60
time_window_sec = 20
target_price = 0
stoploss_price = 0
nearest_expiry_price(
    maper_dict, action_price, action_time, time_window_sec, target_price
)
