import yfinance as yf
import pandas as pd

def download_stock_data(tickers, start_date, end_date):
    data = pd.DataFrame()
    for column_name, ticker in tickers.items():
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        data[column_name] = stock_data['Close']
    return data

def download_economic_data(data, economic_tickers, start_date, end_date):
    for column_name, ticker in economic_tickers.items():
        econ_data = yf.download(ticker, start=start_date, end=end_date)
        data[column_name] = econ_data['Close']
    return data
