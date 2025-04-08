import pandas as pd
import numpy as np
from utils.logging import get_logger

# Get logger for this module
logger = get_logger('indicators')

def calculate_rsi(data, period=14):
    """
    Calculate Relative Strength Index (RSI).
    
    Args:
        data (pandas.DataFrame): DataFrame containing stock data with 'close' column
        period (int): Period for RSI calculation (default: 14)
        
    Returns:
        pandas.DataFrame: DataFrame with original data and RSI column
    """
    try:
        logger.debug(f"Calculating RSI with period={period}")
        
        # Make a copy of the data to avoid modifying the original
        df = data.copy()
        
        # Calculate price changes
        df['price_change'] = df['close'].diff()
        
        # Separate gains and losses
        df['gain'] = df['price_change'].apply(lambda x: x if x > 0 else 0)
        df['loss'] = df['price_change'].apply(lambda x: -x if x < 0 else 0)
        
        # Calculate average gain and loss over the specified period
        df['avg_gain'] = df['gain'].rolling(window=period).mean()
        df['avg_loss'] = df['loss'].rolling(window=period).mean()
        
        # Calculate RS (Relative Strength)
        df['rs'] = df['avg_gain'] / df['avg_loss']
        
        # Calculate RSI
        df['rsi'] = 100 - (100 / (1 + df['rs']))
        
        # Drop intermediate columns
        df = df.drop(['price_change', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs'], axis=1)
        
        # Log the latest RSI value
        latest_rsi = df['rsi'].iloc[-1]
        logger.debug(f"RSI calculation completed. Latest RSI: {latest_rsi:.2f}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error calculating RSI: {str(e)}", exc_info=True)
        raise Exception(f"Error calculating RSI: {str(e)}")