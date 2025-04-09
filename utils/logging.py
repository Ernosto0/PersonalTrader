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
            
            # Special handler for final analysis results
            final_analysis_log_file = logs_dir / 'final_analysis.log'
            final_analysis_handler = RotatingFileHandler(
                final_analysis_log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=10
            )
            final_analysis_handler.setLevel(logging.INFO)
            
            # Create a filter that only accepts final analysis logs
            class FinalAnalysisFilter(logging.Filter):
                def filter(self, record):
                    return hasattr(record, 'final_analysis') and record.final_analysis is True
            
            # Custom formatter for final analysis
            final_analysis_formatter = logging.Formatter(
                '\n%(asctime)s - TICKER: %(ticker)s\n'
                '==============================================\n'
                '%(message)s\n'
                '==============================================\n',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            final_analysis_handler.setFormatter(final_analysis_formatter)
            final_analysis_handler.addFilter(FinalAnalysisFilter())
            logger.addHandler(final_analysis_handler)
        
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

def log_final_analysis(logger, ticker, analysis):
    """
    Log final analysis results in a dedicated log file.
    
    Args:
        logger (logging.Logger): Logger instance (should be 'ai_analyzer' logger)
        ticker (str): Stock ticker symbol
        analysis (dict): Analysis results dictionary
    """
    message = f"""TICKER: {ticker}
TIMESTAMP: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
CURRENT PRICE: ${analysis['price_analysis']['current_price']:.2f}

SUMMARY: {analysis['summary']}
DECISION: {analysis['decision']}
BUY RANGE: ${analysis['buy_range'][0]} - ${analysis['buy_range'][1]}
SELL RANGE: ${analysis['sell_range'][0]} - ${analysis['sell_range'][1]}
RISK LEVEL: {analysis['risk_level']}
REASON: {analysis['reason']}
CONFIDENCE: {analysis['confidence']}%
"""
    
    # Create custom record attributes for the filter
    extra = {
        'final_analysis': True,
        'ticker': ticker
    }
    
    # Log with extra parameters
    logger.info(message, extra=extra) 