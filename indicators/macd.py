import pandas as pd
import numpy as np

def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    """
    Calculate Moving Average Convergence Divergence (MACD).
    
    Args:
        data (pandas.DataFrame): DataFrame containing stock data with 'close' column
        fast_period (int): Period for fast EMA (default: 12)
        slow_period (int): Period for slow EMA (default: 26)
        signal_period (int): Period for signal line (default: 9)
        
    Returns:
        pandas.DataFrame: DataFrame with original data and MACD columns
    """
    try:
        # Make a copy of the data to avoid modifying the original
        df = data.copy()
        
        # Calculate EMAs
        df['ema_fast'] = df['close'].ewm(span=fast_period, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=slow_period, adjust=False).mean()
        
        # Calculate MACD line
        df['macd_line'] = df['ema_fast'] - df['ema_slow']
        
        # Calculate signal line
        df['signal_line'] = df['macd_line'].ewm(span=signal_period, adjust=False).mean()
        
        # Calculate MACD histogram
        df['macd_histogram'] = df['macd_line'] - df['signal_line']
        
        # Drop intermediate columns
        df = df.drop(['ema_fast', 'ema_slow'], axis=1)
        
        return df
    
    except Exception as e:
        raise Exception(f"Error calculating MACD: {str(e)}") 