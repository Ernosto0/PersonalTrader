import logging
import os
from pathlib import Path

def setup_logger():
    """
    Set up logging configuration.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    try:
        # Get the directory of the current file
        current_dir = Path(__file__).parent.parent
        
        # Create logs directory if it doesn't exist
        logs_dir = current_dir / 'logs'
        if not logs_dir.exists():
            logs_dir.mkdir()
        
        # Configure logger
        logger = logging.getLogger('stock_assistant')
        logger.setLevel(logging.INFO)
        
        # Create file handler
        log_file = logs_dir / 'stock_assistant.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    except Exception as e:
        print(f"Error setting up logger: {str(e)}")
        # Return a basic logger if setup fails
        return logging.getLogger('stock_assistant') 