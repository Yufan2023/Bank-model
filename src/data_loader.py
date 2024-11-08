import yfinance as yf
import pandas as pd

def download_data(tickers, start_date, end_date):
    data = pd.DataFrame()
    for column_name, ticker in tickers.items():
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        data[column_name] = stock_data['Close']
    return data
