import yfinance as yf # type: ignore
import pandas as pd
from datetime import datetime, timedelta
from utils.logging import get_logger

# Get logger for this module
logger = get_logger('data_fetcher')

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
    
def get_current_data(ticker):
    """
    Fetch current stock data for the given ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        
    Returns:
        dict: Dictionary containing current stock data
    """
    try:
        logger.debug(f"Fetching current data for {ticker}")
        
        # Create a Ticker object
        stock = yf.Ticker(ticker)
        
        # Get current data
        current_data = stock.info
        
        # Extract key data
        result = {
            'current_price': current_data.get('currentPrice', current_data.get('regularMarketPrice', 0)),
            'previous_close': current_data.get('previousClose', 0),
            'open': current_data.get('open', 0),
            'day_high': current_data.get('dayHigh', 0),
            'day_low': current_data.get('dayLow', 0),
            'volume': current_data.get('volume', 0),
            'market_cap': current_data.get('marketCap', 0),
            'pe_ratio': current_data.get('trailingPE', 0),
            'dividend_yield': current_data.get('dividendYield', 0),
            'bid': current_data.get('bid', 0),
            'ask': current_data.get('ask', 0),
            'fifty_day_avg': current_data.get('fiftyDayAverage', 0),
            'two_hundred_day_avg': current_data.get('twoHundredDayAverage', 0),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        
        logger.debug(f"Current price for {ticker}: ${result['current_price']}")
        
        return result
    
    except Exception as e:
        logger.error(f"Error fetching current data for {ticker}: {str(e)}", exc_info=True)
        raise Exception(f"Error fetching current data for {ticker}: {str(e)}") 
    