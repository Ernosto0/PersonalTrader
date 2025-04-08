#!/usr/bin/env python
import argparse
import sys
import time
from data_fetcher.stock_data import fetch_stock_data, get_current_data
from data_fetcher.news_data import fetch_news_data
from indicators.sma import calculate_sma
from indicators.macd import calculate_macd
from indicators.rsi import calculate_rsi
from ai_analyzer.ai_model import analyze_stock
from utils.config import load_config
from utils.logging import get_logger

def parse_arguments():
    parser = argparse.ArgumentParser(description='AI Stock Trading Assistant')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol (e.g., AMZN)')
    parser.add_argument('--detailed', action='store_true', help='Show detailed analysis from each step')
    parser.add_argument('--log-level', type=str, default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    ticker = args.ticker.upper()
    show_detailed = args.detailed
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    logger = get_logger('stock_assistant')
    logger.setLevel(log_level)
    
    logger.info(f"Starting analysis for ticker: {ticker}")
    start_time = time.time()
    
    try:
        # Fetch current stock data
        logger.info(f"Fetching current data for {ticker}...")
        current_data = get_current_data(ticker)
        current_price = current_data['current_price']
        logger.info(f"Current price for {ticker}: ${current_price}")
        
        # Fetch historical stock data
        logger.info(f"Fetching historical stock data for {ticker}...")
        stock_data = fetch_stock_data(ticker)
        logger.debug(f"Retrieved {len(stock_data)} data points for {ticker}")
        
        # Calculate indicators
        logger.info(f"Calculating technical indicators for {ticker}...")
        sma_data = calculate_sma(stock_data)
        logger.debug("SMA calculation completed")
        
        macd_data = calculate_macd(stock_data)
        logger.debug("MACD calculation completed")
        
        rsi_data = calculate_rsi(stock_data)
        logger.debug("RSI calculation completed")
        
        # Fetch news data
        logger.info(f"Fetching news data for {ticker}...")
        news_data = fetch_news_data(ticker)
        logger.debug(f"Retrieved {len(news_data)} news articles for {ticker}")
        
        # Generate AI analysis
        logger.info(f"Generating AI analysis for {ticker}...")
        analysis = analyze_stock(ticker, stock_data, sma_data, macd_data, rsi_data, news_data, current_price)
        logger.debug("AI analysis completed")
        
        # Print results in a friendly format
        print("\n=== Stock Analysis Report ===")
        print(f"Ticker: {ticker}")
        print(f"Current Price: ${current_price}")
        print(f"Previous Close: ${current_data['previous_close']}")
        print(f"Day Range: ${current_data['day_low']} - ${current_data['day_high']}")
        print(f"52 Week Range: ${stock_data['low'].min():.2f} - ${stock_data['high'].max():.2f}")
        print(f"Volume: {current_data['volume']:,}")
        print("")
        print(f"Summary: {analysis['summary']}")
        print(f"Decision: {analysis['decision']}")
        
        # Check if buy range is valid
        if analysis['buy_range'][0] == 0 and analysis['buy_range'][1] == 0:
            print(f"Buy Range: Not specified")
        else:
            print(f"Buy Range: ${analysis['buy_range'][0]} - ${analysis['buy_range'][1]}")
        
        # Check if sell range is valid
        if analysis['sell_range'][0] == 0 and analysis['sell_range'][1] == 0:
            print(f"Sell Range: Not specified")
        else:
            print(f"Sell Range: ${analysis['sell_range'][0]} - ${analysis['sell_range'][1]}")
        
        print(f"Risk Level: {analysis['risk_level']}")
        print(f"Reason: {analysis['reason']}")
        print(f"Confidence: {analysis['confidence']}%")
        
        # Print detailed analyses if requested
        if show_detailed:
            print("\n=== Detailed Analysis ===")
            
            print("\n--- Price Analysis ---")
            print(analysis['price_analysis']['analysis'])
            
            print("\n--- Technical Indicator Analysis ---")
            print(analysis['technical_analysis']['analysis'])
            
            print("\n--- News Sentiment Analysis ---")
            print(analysis['news_analysis']['analysis'])
        
        # Log completion time
        elapsed_time = time.time() - start_time
        logger.info(f"Analysis completed for {ticker} in {elapsed_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {str(e)}", exc_info=True)
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    import logging
    main() 