import logging
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler
import datetime

def setup_logger(name='stock_assistant', log_level=logging.INFO):
    """
    Set up logging configuration with advanced features.
    
    Args:
        name (str): Name of the logger
        log_level (int): Logging level (default: logging.INFO)
        
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
        logger = logging.getLogger(name)
        logger.setLevel(log_level)
        
        # Clear any existing handlers to avoid duplicates
        if logger.handlers:
            logger.handlers.clear()
        
        # Create rotating file handler (10MB max size, keep 5 backup files)
        log_file = logs_dir / f'{name}.log'
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        
        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        
        # Create formatter with timestamp, logger name, level, and message
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        # Special handler for AI responses with a separate file
        if name == 'ai_analyzer':
            ai_log_file = logs_dir / 'ai_responses.log'
            ai_handler = RotatingFileHandler(
                ai_log_file,
                maxBytes=20*1024*1024,  # 20MB
                backupCount=10
            )
            ai_handler.setLevel(logging.INFO)
            
            # Custom formatter for AI responses
            ai_formatter = logging.Formatter(
                '\n%(asctime)s - %(name)s - %(levelname)s\n'
                '==============================================\n'
                '%(message)s\n'
                '==============================================\n',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            ai_handler.setFormatter(ai_formatter)
            logger.addHandler(ai_handler)
        
        # Log startup message
        logger.info(f"Logger '{name}' initialized at {datetime.datetime.now()}")
        
        return logger
    
    except Exception as e:
        print(f"Error setting up logger: {str(e)}")
        # Return a basic logger if setup fails
        return logging.getLogger(name)

def get_logger(name='stock_assistant'):
    """
    Get an existing logger or create a new one.
    
    Args:
        name (str): Name of the logger
        
    Returns:
        logging.Logger: Logger instance
    """
    logger = logging.getLogger(name)
    
    # If logger doesn't have handlers, it hasn't been configured yet
    if not logger.handlers:
        return setup_logger(name)
    
    return logger

def log_ai_response(logger, ticker, analysis_type, response):
    """
    Log AI response in a formatted way that makes it easy to read in log files.
    
    Args:
        logger (logging.Logger): Logger instance
        ticker (str): Stock ticker symbol
        analysis_type (str): Type of analysis (price, technical, news, final)
        response (str): AI response text
    """
    header = f"AI {analysis_type.upper()} ANALYSIS FOR {ticker}"
    separator = "=" * len(header)
    
    message = f"{separator}\n{header}\n{separator}\n\n{response}"
    logger.info(message) 