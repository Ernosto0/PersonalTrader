import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

def fetch_stock_data(ticker, period='2y'):
    """
    Fetch historical stock data for the given ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        period (str): Time period to fetch data for (default: '2y')
        
    Returns:
        pandas.DataFrame: DataFrame containing historical stock data
    """
    try:
        # Create a Ticker object
        stock = yf.Ticker(ticker)
        
        # Fetch historical data
        df = stock.history(period=period)
        
        # Reset index to make Date a column
        df = df.reset_index()
        
        # Rename columns for clarity
        df = df.rename(columns={
            'Date': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })
        
        return df
    
    except Exception as e:
        raise Exception(f"Error fetching stock data for {ticker}: {str(e)}") 