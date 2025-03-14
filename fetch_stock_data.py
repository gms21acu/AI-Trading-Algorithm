import pandas as pd
import os

def load_stock_data_from_csv(tickers, data_directory):
    data = {}
    for ticker in tickers:
        file_path = os.path.join(data_directory, f"{ticker}_data.csv")
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, parse_dates=['Date'], index_col='Date')
            if 'Close' in df.columns:
                data[ticker] = df[['Close']]
            else:
                print(f"No 'Close' column found in {ticker}_data.csv")
        else:
            print(f"CSV file for {ticker} not found in {data_directory}")
    return data

tickers = ['F', 'AAPL', 'NATI', 'NKE', 'GOOGL', 'MSFT', 'NVDA', 'AMZN', 'META', 'TSLA', 
           'AMD', 'INTC', 'ORCL', 'IBM', 'CSCO', 'JNJ', 'PFE', 'MRNA', 'UNH', 'ABBV', 
           'JPM', 'GS', 'BAC', 'WFC', 'C', 'WMT', 'PG', 'KO', 'MCD', 'HD', 'XOM', 
           'CVX', 'GE', 'BA', 'CAT']

data_directory = r'C:\Users\grije\OneDrive\Documents\code\AI\stock tarder_gms21a\cvs data'

data = load_stock_data_from_csv(tickers, data_directory)

for ticker, df in data.items():
    print(f"Loaded data for {ticker}:")
    print(df.head())
