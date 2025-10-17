import yfinance as yf
import mysql.connector
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify
from connection_test import get_instrument_token, stock_data
import pandas as pd
import numpy as np
import os

token_bp = Blueprint("get_token", __name__)
stock_bp = Blueprint("get_stock", __name__)
UserStock_bp = Blueprint("user_stock", __name__)
data_resampling_bp = Blueprint("data_resampling", __name__)
indicators_bp = Blueprint("stock_indicators", __name__)


@token_bp.route("/get_token", methods=["POST"])
def get_token():
    data = request.get_json()
    # return jsonify(data)
    if not data or "symbol" not in data:
        return jsonify({"error": "No symbol provided"}), 400

    symbol = data["symbol"].upper()
    token = get_instrument_token(symbol)

    if token:
        return jsonify({"symbol": symbol, "token": token})
    else:
        return jsonify({"error": f"Symbol {symbol} not found"}), 404


@stock_bp.route("/get_stock", methods=["POST"])
def get_stock():

    data = request.get_json()
    # print(data)
    required_keys = ["symbol", "date", "time"]
    if not data or not all(k in data for k in required_keys):
        return jsonify({"error": "No symbol provided"}), 400

    symbol = data["symbol"].upper()
    date = data["date"]
    time = data["time"]
    stock = stock_data(symbol, date, time)
    if stock:
        # Insert into database
        conn = mysql.connector.connect(
            host="localhost", user="root", password="Krishna@794", database="Stock_DB"
        )
        cursor = conn.cursor()
        insert_query = """
        INSERT INTO Stock (Stock_name, Open, High, Low, Close, Volume,time)
        VALUES (%s, %s, %s, %s, %s, %s,%s)
        """
        cursor.execute(
            insert_query,
            (
                symbol,
                stock["Open"],
                stock["High"],
                stock["Low"],
                stock["Close"],
                stock["Volume"],
                stock["time"],
            ),
        )
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({"symbol": symbol, **stock})
    else:
        return jsonify({"error": f"No data found for symbol {symbol}"}), 404


def download_stock_data(symbols_chunk, start_date, end_date, interval="1m"):
    """Download multiple symbols in bulk"""
    return yf.download(
        symbols_chunk,
        start=start_date,
        end=end_date,
        interval=interval,
        group_by="ticker",
        threads=True,
    )


@UserStock_bp.route("/get_user_stock", methods=["POST"])
def get_user_stock():
    total_start = time.time()
    data = request.get_json()

    if not data or "symbol" not in data:
        return jsonify({"error": "symbol is required"}), 400

    symbols = data["symbol"]
    if not isinstance(symbols, list):
        symbols = [symbols]

    summary = []

    # Connect to DB once
    conn = mysql.connector.connect(
        host="localhost", user="root", password="Krishna@794", database="Stock_DB"
    )
    cursor = conn.cursor()

    today = int(datetime.now().strftime("%y%m%d"))
    download_symbols = {}

    # Step 1: Determine missing data per symbol
    for sym in symbols:
        symbol = sym.upper()
        cursor.execute(
            "SELECT DISTINCT date FROM UserStock WHERE Stock_name=%s ORDER BY date DESC LIMIT 7",
            (symbol,),
        )
        rows = cursor.fetchall()
        existing_dates = [r[0] for r in rows]
        latest_db_date = max(existing_dates) if existing_dates else None

        if latest_db_date is None or latest_db_date < today:
            start_date = (
                (
                    datetime.strptime(str(latest_db_date), "%y%m%d") + timedelta(days=1)
                ).strftime("%Y-%m-%d")
                if latest_db_date
                else (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")
            )
            download_symbols[symbol] = start_date
        else:
            summary.append(
                {
                    "symbol": symbol,
                    "rows_inserted": 0,
                    "message": "Data already up to date",
                    "latest_date_in_db": latest_db_date,
                }
            )

    if not download_symbols:
        cursor.close()
        conn.close()
        return (
            jsonify(
                {
                    "result": summary,
                    "total_runtime_seconds": round(time.time() - total_start, 2),
                }
            ),
            200,
        )

    # Step 2: Parallel download in chunks
    chunk_size = 5
    symbols_list = list(download_symbols.keys())
    tasks = [
        symbols_list[i : i + chunk_size]
        for i in range(0, len(symbols_list), chunk_size)
    ]

    def process_chunk(chunk):
        start_dates = [download_symbols[s] for s in chunk]
        start_date = min(start_dates)
        end_date = datetime.now().strftime("%Y-%m-%d")
        yf_symbols = [s + ".NS" for s in chunk]

        df = download_stock_data(yf_symbols, start_date, end_date)

        chunk_summary = []
        for symbol in chunk:
            yf_symbol = symbol + ".NS"
            if yf_symbol not in df.columns.levels[0]:
                chunk_summary.append(
                    {
                        "symbol": symbol,
                        "rows_inserted": 0,
                        "message": "No data downloaded",
                    }
                )
                continue

            symbol_df = df[yf_symbol].copy()
            if symbol_df.empty:
                chunk_summary.append(
                    {
                        "symbol": symbol,
                        "rows_inserted": 0,
                        "message": "No data in downloaded range",
                    }
                )
                continue

            symbol_df.reset_index(inplace=True)
            symbol_df["Datetime"] = symbol_df["Datetime"].dt.tz_convert("Asia/Kolkata")
            symbol_df["date"] = symbol_df["Datetime"].dt.date.apply(
                lambda x: int(x.strftime("%y%m%d"))
            )
            symbol_df["time"] = (
                symbol_df["Datetime"].dt.hour * 3600
                + symbol_df["Datetime"].dt.minute * 60
                + symbol_df["Datetime"].dt.second
            )

            # Filter NSE market hours
            symbol_df = symbol_df[
                (symbol_df["time"] >= 33300) & (symbol_df["time"] <= 55800)
            ]

            records = [
                (
                    symbol,
                    float(row["Open"]) if not pd.isna(row["Open"]) else None,
                    float(row["High"]) if not pd.isna(row["High"]) else None,
                    float(row["Low"]) if not pd.isna(row["Low"]) else None,
                    float(row["Close"]) if not pd.isna(row["Close"]) else None,
                    int(row["Volume"]) if not pd.isna(row["Volume"]) else None,
                    int(row["time"]),
                    int(row["date"]),
                )
                for _, row in symbol_df.iterrows()
            ]

            rows_inserted = 0
            if records:
                insert_query = """
                    INSERT IGNORE INTO UserStock
                    (Stock_name, Open, High, Low, Close, Volume, time, date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                """
                cursor.executemany(insert_query, records)
                conn.commit()
                rows_inserted = len(records)

            chunk_summary.append(
                {
                    "symbol": symbol,
                    "rows_inserted": rows_inserted,
                    "data_range": f"{download_symbols[symbol]} to {end_date}",
                }
            )

        return chunk_summary

    # Execute chunks in parallel
    with ThreadPoolExecutor(max_workers=min(len(tasks), 5)) as executor:
        results = executor.map(process_chunk, tasks)
        for r in results:
            summary.extend(r)

    cursor.close()
    conn.close()
    total_runtime = round(time.time() - total_start, 2)

    return jsonify({"result": summary, "total_runtime_seconds": total_runtime}), 200


@data_resampling_bp.route("/resample_stock", methods=["POST"])
def resample_stock():
    """
    Resample stored 1-minute stock data into higher timeframes (e.g. 5min, 15min, 30min, 60min).

    Example Request:
    {
        "symbol": "SBIN"               # or ["SBIN", "TCS"]
        "minute": 5,
        "start_date": "2025-10-01",    # optional
        "end_date": "2025-10-10"       # optional
    }
    """

    # --- Read and validate request ---
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    if "symbol" not in data or "minute" not in data:
        return jsonify({"error": "Fields 'symbol' and 'minute' are required"}), 400

    # --- Normalize symbol input ---
    symbol_input = data["symbol"]
    if isinstance(symbol_input, str):
        symbols = [symbol_input.upper()]
    elif isinstance(symbol_input, list):
        symbols = [s.upper() for s in symbol_input if isinstance(s, str)]
        if not symbols:
            return jsonify({"error": "Symbol list is empty or invalid"}), 400
    else:
        return jsonify({"error": "Field 'symbol' must be a string or list"}), 400

    minute = int(data["minute"])
    start_date = data.get("start_date")
    end_date = data.get("end_date")

    # --- Database connection ---
    conn = mysql.connector.connect(
        host="localhost", user="root", password="Krishna@794", database="Stock_DB"
    )
    cursor = conn.cursor(dictionary=True)

    results = {}

    for symbol in symbols:
        # --- Build SQL query ---
        query = """
            SELECT Stock_name, Open, High, Low, Close, Volume, time, date
            FROM UserStock
            WHERE Stock_name = %s
        """
        params = [symbol]

        if start_date:
            start_int = int(
                datetime.strptime(start_date, "%Y-%m-%d").strftime("%y%m%d")
            )
            query += " AND date >= %s"
            params.append(start_int)

        if end_date:
            end_int = int(datetime.strptime(end_date, "%Y-%m-%d").strftime("%y%m%d"))
            query += " AND date <= %s"
            params.append(end_int)

        query += " ORDER BY date, time"

        cursor.execute(query, tuple(params))
        rows = cursor.fetchall()

        if not rows:
            results[symbol] = {"message": f"No data found for {symbol}"}
            continue

        df = pd.DataFrame(rows)

        # --- Convert date + time into Datetime ---
        df["Datetime"] = pd.to_datetime(
            df["date"].astype(str).apply(lambda x: f"20{x[:2]}-{x[2:4]}-{x[4:]}")
            + " "
            + df["time"].apply(lambda t: f"{t//3600:02}:{(t%3600)//60:02}:{t%60:02}")
        )

        df.set_index("Datetime", inplace=True)
        df = df.sort_index()

        # --- Resample OHLCV data ---
        rule = f"{minute}T"
        resampled = (
            df.resample(rule)
            .agg(
                {
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum",
                }
            )
            .dropna()
            .reset_index()
        )

        resampled["Datetime"] = resampled["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")

        results[symbol] = {
            "interval": f"{minute}min",
            "rows": len(resampled),
            "data": resampled.to_dict(orient="records"),
        }

    cursor.close()
    conn.close()

    return jsonify(results), 200


@indicators_bp.route("/stock_indicators", methods=["POST"])
def stock_indicators():
    """
    Resample stored 1-minute CASH feather data into higher timeframes (e.g. 5min, 15min, 30min, 60min).

    Example request:
    {
        "symbol": "ABB",        # or ["ABB", "SBIN"]
        "minute": 5,
        "year": "2023",         # optional
        "month": "JAN",         # optional
        "day": "02"             # optional
    }
    """
    data = request.get_json()
    if not data or "symbol" not in data or "minute" not in data:
        return jsonify({"error": "symbol and minute are required"}), 400

    # --- Normalize input ---
    symbol_input = data["symbol"]
    if isinstance(symbol_input, str):
        symbols = [symbol_input.upper()]
    elif isinstance(symbol_input, list):
        symbols = [s.upper() for s in symbol_input if isinstance(s, str)]
        if not symbols:
            return jsonify({"error": "Symbol list is empty or invalid"}), 400
    else:
        return jsonify({"error": "Field 'symbol' must be a string or list"}), 400

    minute = int(data["minute"])
    sma_periods = data.get("sma_periods", [5])
    rsi_periods = data.get("rsi_periods", [])
    if isinstance(sma_periods, int):
        sma_periods = [sma_periods]
    elif not isinstance(sma_periods, list):
        sma_periods = []
    if isinstance(rsi_periods, int):
        rsi_periods = [rsi_periods]
    elif not isinstance(rsi_periods, list):
        rsi_periods = []
    base_path = os.path.normpath("dataset")
    results = {}

    # --- Determine target paths ---
    year = data.get("year")
    month = data.get("month")
    day = data.get("day")

    target_paths = []

    try:
        if year:
            year_path = os.path.join(base_path, year)
            months = [month] if month else sorted(os.listdir(year_path))
            for m in months:
                month_path = os.path.join(year_path, m)
                days = [day] if day else sorted(os.listdir(month_path))
                for d in days:
                    target_paths.append(os.path.join(month_path, d))
        else:
            # Auto-detect latest date if no year is provided
            years = sorted(os.listdir(base_path), reverse=True)
            latest_year = years[0]
            months = sorted(
                os.listdir(os.path.join(base_path, latest_year)), reverse=True
            )
            latest_month = months[0]
            days = sorted(
                os.listdir(os.path.join(base_path, latest_year, latest_month)),
                reverse=True,
            )
            latest_day = days[0]
            target_paths.append(
                os.path.join(base_path, latest_year, latest_month, latest_day)
            )
    except Exception as e:
        return jsonify({"error": f"Failed to locate dataset folders: {str(e)}"}), 500

    # --- Process each symbol across all target paths ---
    for symbol in symbols:
        all_resampled = []
        source_files = []

        for target_path in target_paths:
            file_name = f"{symbol.lower()}_cash.feather"
            file_path = os.path.join(target_path, file_name)

            if not os.path.exists(file_path):
                continue

            try:
                df = pd.read_feather(file_path)
                new_data = pd.DataFrame(df)
                # # print(df)

                # for i in range(len(new_data)):
                #     if i == 0:
                #         continue
                #     print(
                #         new_data["time"].iloc[i],
                #         new_data["time"].iloc[i - 1],
                #     )

            except Exception as e:
                continue

            if df.empty:
                continue

            if "Datetime" not in df.columns:
                if "date" in df.columns and "time" in df.columns:
                    df["Datetime"] = pd.to_datetime(
                        df["date"].astype(str) + df["time"].astype(str).str.zfill(6),
                        format="%y%m%d%H%M%S",
                        errors="coerce",
                    )
                    if df["Datetime"].isna().any():
                        df["Datetime"] = pd.to_datetime(
                            df["date"], format="%y%m%d"
                        ) + pd.to_timedelta(df["time"].astype(int), unit="s")
                elif "timestamp" in df.columns:
                    df.rename(columns={"timestamp": "Datetime"}, inplace=True)
                else:
                    continue

            df["Datetime"] = pd.to_datetime(df["Datetime"])
            df.set_index("Datetime", inplace=True)
            df = df.sort_index()

            required_cols = {"open", "high", "low", "close"}
            if not required_cols.issubset(df.columns):
                continue

            rule = f"{minute}T"
            resampled = (
                df.resample(rule)
                .agg(
                    {
                        "open": "first",
                        "high": "max",
                        "low": "min",
                        "close": "last",
                    }
                )
                .dropna()
                .reset_index()
            )

            # --- Calculate SMA(s) ---
            for period in sma_periods:
                resampled[f"SMA_{period}"] = (
                    resampled["close"].rolling(window=period).mean()
                )

            # --- Calculate RSI(s) with EMA method ---
            for period in rsi_periods:
                delta = resampled["close"].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)

                # EMA of gains and losses
                avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
                avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

                rs = avg_gain / avg_loss
                resampled[f"RSI_{period}"] = 100 - (100 / (1 + rs))


            # # --- Detect Engulfing Patterns ---
            resampled["pattern"] = None
            resampled["signal"] = None
            resampled["prev_open"] = None
            resampled["prev_close"] = None

            for i in range(1, len(resampled)):
                prev = resampled.iloc[i - 1]
                curr = resampled.iloc[i]

                prev_open, prev_close = prev["open"], prev["close"]
                curr_open, curr_close = curr["open"], curr["close"]

                prev_body = abs(prev_close - prev_open)
                curr_body = abs(curr_close - curr_open)

                # --- Strong Bullish Engulfing ---
                if (
                    prev_close < prev_open and                  # previous bearish
                    curr_close > curr_open and                  # current bullish
                    curr_open <= prev_close and                 # open below previous close
                    curr_close >= prev_open and                 # close above previous open
                    curr_body > prev_body  

                ):
                    resampled["prev_open"] = prev_open
                    resampled["prev_close"] = prev_close
                    resampled.loc[i, "pattern"] = "Strong_Bullish_Engulfing"
                    resampled.loc[i, "signal"] = "BUY"


                # --- Strong Bearish Engulfing ---
                elif (
                    prev_close > prev_open  # previous bullish
                    and curr_close < curr_open  # current bearish
                    and curr_open >= prev_close  # open above previous close
                    and curr_close <= prev_open  # close below previous open
                    and curr_body > prev_body  # stronger body
                ):
                    resampled["prev_open"] = prev_open
                    resampled["prev_close"] = prev_close
                    resampled.loc[i, "pattern"] = "Strong_Bearish_Engulfing"
                    resampled.loc[i, "signal"] = "SELL"


            all_resampled.append(resampled)
            source_files.append(file_path)

        if all_resampled:
            combined = (
                pd.concat(all_resampled).sort_values("Datetime").reset_index(drop=True)
            )
            combined["Datetime"] = combined["Datetime"].dt.strftime("%Y-%m-%d %H:%M:%S")
            results[symbol] = {
                "interval": f"{minute}min",
                "rows": len(combined),
                "data": combined.to_dict(orient="records"),
            }
        else:
            results[symbol] = {
                "error": f"No CASH data found for {symbol} in the specified period."
            }

    return jsonify(results), 200
