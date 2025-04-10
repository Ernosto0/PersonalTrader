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
from ai_analyzer.track_token_usage import get_token_usage, reset_token_usage
from utils.config import load_config
from utils.logging import get_logger
from ai_analyzer.ai_model import MODAL

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='AI Stock Trading Assistant - An AI-powered tool for stock analysis and trading recommendations',
        epilog='Example usage:\n  python main.py --ticker AAPL  # Basic analysis\n  python main.py --ticker MSFT --detailed  # Detailed analysis\n  python main.py --ticker GOOG --backtest --periods 10  # Backtest with 10 periods\n  python main.py --ticker AMZN --show-tokens',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Required arguments
    parser.add_argument('--ticker', type=str, required=True, 
                        help='Stock ticker symbol (e.g., AAPL, MSFT, GOOG)')
    
    # Optional arguments
    parser.add_argument('--detailed', action='store_true', 
                        help='Show detailed analysis from each step including price, technical, news, and correlation analysis')
    parser.add_argument('--backtest', action='store_true', 
                        help='Run backtesting to validate model against historical data and calculate accuracy')
    parser.add_argument('--periods', type=int, default=5,
                        help='Number of historical periods to backtest (default: 5)')
    parser.add_argument('--show-tokens', action='store_true',
                        help='Show token usage and cost information for OpenAI API calls')
    parser.add_argument('--log-level', type=str, default='INFO', 
                        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                        help='Set the logging level for more or less verbose output')
    
    return parser.parse_args()

def run_backtest(ticker, periods=5, log_level='INFO', show_tokens=False):
    """
    Run backtesting to validate model against historical data
    
    Args:
        ticker (str): Stock ticker symbol
        periods (int): Number of historical periods to backtest
        log_level (str): Logging level
        show_tokens (bool): Whether to show token usage information
    """
    import pandas as pd
    from datetime import datetime, timedelta
    
    # Reset token usage counter before starting
    reset_token_usage()
    
    logger = get_logger('backtest')
    logger.setLevel(getattr(logging, log_level))
    
    logger.info(f"Starting backtesting for {ticker} with {periods} periods")
    
    # Fetch complete historical data
    stock_data = fetch_stock_data(ticker, years=3)  # Get 3 years of data for backtesting
    
    if len(stock_data) < 252 * 2:  # Need at least 2 years of data
        logger.error(f"Insufficient historical data for {ticker} to perform backtesting")
        print(f"Error: Insufficient historical data for {ticker} to perform backtesting")
        return
    
    # Define periods for backtesting
    period_length = 30  # days
    results = []
    
    # Create a dataframe to store results
    backtest_results = pd.DataFrame(columns=[
        'Date', 'Price', 'Decision', 'Buy_Low', 'Buy_High', 
        'Sell_Low', 'Sell_High', 'Result_30d', 'Result_90d', 'Accuracy'
    ])
    
    for i in range(periods):
        # Calculate the end date for this period
        # Start from the most recent periods and go backwards
        if i == 0:
            # For the first period, use data up to the last available date minus 90 days
            # (so we can verify the 90-day performance)
            end_idx = len(stock_data) - 90
        else:
            # For subsequent periods, go back by period_length days each time
            end_idx = len(stock_data) - 90 - (i * period_length)
        
        if end_idx < 252:  # Need at least 1 year of data
            logger.warning(f"Insufficient data for period {i+1}, skipping")
            continue
        
        # Get the date for this period
        period_date = stock_data.index[end_idx].strftime('%Y-%m-%d')
        
        # Get data up to this point for analysis
        period_data = stock_data.iloc[:end_idx].copy()
        
        logger.info(f"Analyzing period {i+1}/{periods} - Date: {period_date}")
        
        # Calculate indicators
        sma_data = calculate_sma(period_data)
        macd_data = calculate_macd(period_data)
        rsi_data = calculate_rsi(period_data)
        
        # Fetch news data is not practical for backtesting, use empty list
        news_data = []
        
        # Get the current price for this period
        current_price = period_data['close'].iloc[-1]
        
        # Generate analysis
        try:
            analysis = analyze_stock(ticker, period_data, sma_data, macd_data, rsi_data, news_data, current_price)
            
            # Get future prices for validation (30 and 90 days later)
            future_idx_30d = min(end_idx + 30, len(stock_data) - 1)
            future_idx_90d = min(end_idx + 90, len(stock_data) - 1)
            
            future_price_30d = stock_data['close'].iloc[future_idx_30d]
            future_price_90d = stock_data['close'].iloc[future_idx_90d]
            
            # Calculate returns
            return_30d = ((future_price_30d / current_price) - 1) * 100
            return_90d = ((future_price_90d / current_price) - 1) * 100
            
            # Determine if the recommendation was accurate
            decision = analysis['decision'].lower()
            accuracy = 0
            
            if 'buy' in decision or 'accumulate' in decision:
                # For buy recommendations, positive returns indicate accuracy
                accuracy = 1 if return_90d > 0 else 0
            elif 'sell' in decision or 'reduce' in decision:
                # For sell recommendations, negative returns indicate accuracy
                accuracy = 1 if return_90d < 0 else 0
            elif 'hold' in decision or 'neutral' in decision:
                # For hold recommendations, small changes indicate accuracy
                accuracy = 1 if abs(return_90d) < 10 else 0
            
            # Add to results
            backtest_results = pd.concat([backtest_results, pd.DataFrame([{
                'Date': period_date,
                'Price': current_price,
                'Decision': analysis['decision'],
                'Buy_Low': analysis['buy_range'][0],
                'Buy_High': analysis['buy_range'][1],
                'Sell_Low': analysis['sell_range'][0],
                'Sell_High': analysis['sell_range'][1],
                'Result_30d': return_30d,
                'Result_90d': return_90d,
                'Accuracy': accuracy
            }])], ignore_index=True)
            
            logger.info(f"Period {i+1} analyzed: {decision}, 30d: {return_30d:.2f}%, 90d: {return_90d:.2f}%, Accuracy: {accuracy}")
            
        except Exception as e:
            logger.error(f"Error analyzing period {i+1}: {str(e)}")
    
    # Calculate overall accuracy
    if not backtest_results.empty:
        overall_accuracy = backtest_results['Accuracy'].mean() * 100
        
        # Print results
        print("\n=== Backtesting Results ===")
        print(f"Ticker: {ticker}")
        print(f"Periods: {len(backtest_results)}")
        print(f"Overall Accuracy: {overall_accuracy:.2f}%")
        print("\nDetailed Results:")
        print(backtest_results[['Date', 'Price', 'Decision', 'Result_30d', 'Result_90d', 'Accuracy']].to_string(index=False))
        
        # Save results to CSV
        results_file = f"backtest_{ticker}_{datetime.now().strftime('%Y%m%d')}.csv"
        backtest_results.to_csv(results_file, index=False)
        print(f"\nResults saved to {results_file}")
        
        logger.info(f"Backtesting completed for {ticker} with overall accuracy: {overall_accuracy:.2f}%")
        
        # Display token usage if requested
        if show_tokens:
            display_token_usage(MODAL)
    else:
        print("No valid backtesting results generated")
        logger.warning("No valid backtesting results generated")

def display_token_usage(MODAL):
    """
    Display token usage information
    """
    usage = get_token_usage()
    
    print("\n=== Token Usage Information ===")
    print(f"Modal: {MODAL}")
    print(f"Total API Calls: {usage['calls']}")
    print(f"Total Tokens: {usage['total_tokens']:,}")
    print(f"Prompt Tokens: {usage['prompt_tokens']:,}")
    print(f"Completion Tokens: {usage['completion_tokens']:,}")
    print(f"Estimated Cost: ${usage['cost']:.4f}")

def main():
    # Parse command line arguments
    args = parse_arguments()
    ticker = args.ticker.upper()
    show_detailed = args.detailed
    show_tokens = args.show_tokens
    
    # Reset token usage counter before starting
    reset_token_usage()
    
    # Setup logging
    log_level = getattr(logging, args.log_level)
    logger = get_logger('stock_assistant')
    logger.setLevel(log_level)
    
    # Check if backtesting is requested
    if args.backtest:
        run_backtest(ticker, args.periods, args.log_level, show_tokens)
        return
    
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
        print(f"Strategy: {analysis['strategy']}")
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
        
        # Add market correlation information
        if 'correlation_analysis' in analysis:
            correlation = analysis['correlation_analysis']
            print("\n=== Market Correlation ===")
            print(f"Correlation with S&P 500: {correlation['correlation']:.2f}")
            print(f"Beta: {correlation['beta']:.2f}")
            print(f"Alpha: {correlation['alpha']:.4f}")
            
            # Print interpretation of correlation metrics
            if correlation['correlation'] > 0.8:
                print("Strong positive correlation with the market")
            elif correlation['correlation'] > 0.5:
                print("Moderate positive correlation with the market")
            elif correlation['correlation'] > 0.2:
                print("Weak positive correlation with the market")
            elif correlation['correlation'] > -0.2:
                print("Little to no correlation with the market")
            else:
                print("Negative correlation with the market")
                
            # Print interpretation of beta
            if correlation['beta'] > 1.5:
                print("High volatility compared to the market")
            elif correlation['beta'] > 1.0:
                print("More volatile than the market")
            elif correlation['beta'] > 0.5:
                print("Less volatile than the market")
            else:
                print("Much less volatile than the market")
        
        # Print detailed analyses if requested
        if show_detailed:
            print("\n=== Detailed Analysis ===")
            
            print("\n--- Price Analysis ---")
            print(analysis['price_analysis']['analysis'])
            
            print("\n--- Technical Indicator Analysis ---")
            print(analysis['technical_analysis']['analysis'])
            
            print("\n--- News Sentiment Analysis ---")
            print(analysis['news_analysis']['analysis'])
            
            # Add correlation analysis if available
            if 'correlation_analysis' in analysis:
                print("\n--- Market Correlation Analysis ---")
                print(analysis['correlation_analysis']['analysis'])
        
        # Display token usage if requested
        if show_tokens:
            display_token_usage(MODAL)
        
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