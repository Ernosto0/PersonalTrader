#!/usr/bin/env python
import argparse
import sys
from data_fetcher.stock_data import fetch_stock_data
from data_fetcher.news_data import fetch_news_data
from indicators.sma import calculate_sma
from indicators.macd import calculate_macd
from indicators.rsi import calculate_rsi
from ai_analyzer.ai_model import analyze_stock
from utils.config import load_config
from utils.logging import setup_logger

def parse_arguments():
    parser = argparse.ArgumentParser(description='AI Stock Trading Assistant')
    parser.add_argument('--ticker', type=str, required=True, help='Stock ticker symbol (e.g., AMZN)')
    parser.add_argument('--detailed', action='store_true', help='Show detailed analysis from each step')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    ticker = args.ticker.upper()
    show_detailed = args.detailed
    
    # Setup logging
    logger = setup_logger()
    logger.info(f"Starting analysis for ticker: {ticker}")
    
    try:
        # Fetch stock data
        logger.info(f"Fetching stock data for {ticker}...")
        stock_data = fetch_stock_data(ticker)
        
        # Calculate indicators
        logger.info(f"Calculating technical indicators for {ticker}...")
        sma_data = calculate_sma(stock_data)
        macd_data = calculate_macd(stock_data)
        rsi_data = calculate_rsi(stock_data)
        
        # Fetch news data
        logger.info(f"Fetching news data for {ticker}...")
        news_data = fetch_news_data(ticker)
        
        # Generate AI analysis
        logger.info(f"Generating AI analysis for {ticker}...")
        analysis = analyze_stock(ticker, stock_data, sma_data, macd_data, rsi_data, news_data)
        
        # Print results in a friendly format
        print("\n=== Stock Analysis Report ===")
        print(f"Ticker: {ticker}")
        print(f"Summary: {analysis['summary']}")
        print(f"Decision: {analysis['decision']}")
        print(f"Buy Range: ${analysis['buy_range'][0]} - ${analysis['buy_range'][1]}")
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
        
    except Exception as e:
        logger.error(f"Error analyzing {ticker}: {str(e)}")
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 