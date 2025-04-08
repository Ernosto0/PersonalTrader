import pandas as pd
import numpy as np
from utils.logging import get_logger

# Get logger for this module
logger = get_logger('indicators')

def calculate_sma(data, periods=[20, 50, 200]):
    """
    Calculate Simple Moving Average (SMA) for multiple periods.
    
    Args:
        data (pandas.DataFrame): DataFrame containing stock data with 'close' column
        periods (list): List of periods to calculate SMA for (default: [20, 50, 200])
        
    Returns:
        pandas.DataFrame: DataFrame with original data and SMA columns
    """
    try:
        logger.debug(f"Calculating SMA for periods: {periods}")
        
        # Make a copy of the data to avoid modifying the original
        df = data.copy()
        
        # Calculate SMA for each period
        for period in periods:
            column_name = f'sma_{period}'
            df[column_name] = df['close'].rolling(window=period).mean()
            logger.debug(f"Calculated SMA-{period}: {df[column_name].iloc[-1]:.2f}")
        
        logger.debug("SMA calculation completed")
        return df
    
    except Exception as e:
        logger.error(f"Error calculating SMA: {str(e)}", exc_info=True)
        raise Exception(f"Error calculating SMA: {str(e)}") 