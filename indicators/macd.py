import pandas as pd
import numpy as np
from utils.logging import get_logger

# Get logger for this module
logger = get_logger('indicators')

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
        logger.debug(f"Calculating MACD with fast_period={fast_period}, slow_period={slow_period}, signal_period={signal_period}")
        
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
        
        # Log the latest MACD values
        latest_macd = df['macd_line'].iloc[-1]
        latest_signal = df['signal_line'].iloc[-1]
        latest_histogram = df['macd_histogram'].iloc[-1]
        
        logger.debug(f"MACD calculation completed. Latest values: MACD={latest_macd:.2f}, Signal={latest_signal:.2f}, Histogram={latest_histogram:.2f}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error calculating MACD: {str(e)}", exc_info=True)
        raise Exception(f"Error calculating MACD: {str(e)}") 