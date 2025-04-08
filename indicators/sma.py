import pandas as pd
import numpy as np

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
        # Make a copy of the data to avoid modifying the original
        df = data.copy()
        
        # Calculate SMA for each period
        for period in periods:
            column_name = f'sma_{period}'
            df[column_name] = df['close'].rolling(window=period).mean()
        
        return df
    
    except Exception as e:
        raise Exception(f"Error calculating SMA: {str(e)}") 