import openai
from utils.config import load_config

def analyze_price_data(ticker, stock_data):
    """
    Analyze historical price data to identify patterns and trends.
    
    Args:
        ticker (str): Stock ticker symbol
        stock_data (pandas.DataFrame): DataFrame containing stock price data
        
    Returns:
        dict: Dictionary containing price analysis results
    """
    try:
        # Load OpenAI API key from config
        config = load_config()
        api_key = config.get('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in configuration")
        
        # Set OpenAI API key
        openai.api_key = api_key
        
        # Prepare data for analysis
        latest_price = stock_data['close'].iloc[-1]
        highest_price = stock_data['high'].max()
        lowest_price = stock_data['low'].min()
        avg_price = stock_data['close'].mean()
        price_volatility = stock_data['close'].std()
        
        # Calculate price change percentages
        price_change_1m = ((stock_data['close'].iloc[-1] / stock_data['close'].iloc[-22]) - 1) * 100 if len(stock_data) >= 22 else 0
        price_change_3m = ((stock_data['close'].iloc[-1] / stock_data['close'].iloc[-66]) - 1) * 100 if len(stock_data) >= 66 else 0
        price_change_6m = ((stock_data['close'].iloc[-1] / stock_data['close'].iloc[-132]) - 1) * 100 if len(stock_data) >= 132 else 0
        price_change_1y = ((stock_data['close'].iloc[-1] / stock_data['close'].iloc[-252]) - 1) * 100 if len(stock_data) >= 252 else 0
        price_change_2y = ((stock_data['close'].iloc[-1] / stock_data['close'].iloc[0]) - 1) * 100
        
        # Create prompt for OpenAI
        prompt = f"""
        Analyze the following historical price data for {ticker} and provide insights:
        
        Current Price: ${latest_price:.2f}
        Highest Price (2 years): ${highest_price:.2f}
        Lowest Price (2 years): ${lowest_price:.2f}
        Average Price (2 years): ${avg_price:.2f}
        Price Volatility: ${price_volatility:.2f}
        
        Price Changes:
        - 1 Month: {price_change_1m:.2f}%
        - 3 Months: {price_change_3m:.2f}%
        - 6 Months: {price_change_6m:.2f}%
        - 1 Year: {price_change_1y:.2f}%
        - 2 Years: {price_change_2y:.2f}%
        
        Please provide a detailed analysis of the price trends, patterns, and potential price levels for support and resistance.
        Focus on identifying key price levels, trend direction, and any notable patterns in the price history.
        """
        
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional stock analyst specializing in technical price analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        # Extract analysis from response
        price_analysis = response.choices[0].message.content
        
        return {
            'analysis': price_analysis,
            'current_price': latest_price,
            'highest_price': highest_price,
            'lowest_price': lowest_price,
            'avg_price': avg_price,
            'price_volatility': price_volatility,
            'price_changes': {
                '1m': price_change_1m,
                '3m': price_change_3m,
                '6m': price_change_6m,
                '1y': price_change_1y,
                '2y': price_change_2y
            }
        }
    
    except Exception as e:
        raise Exception(f"Error analyzing price data for {ticker}: {str(e)}")

def analyze_technical_indicators(ticker, sma_data, macd_data, rsi_data):
    """
    Analyze technical indicators to identify trading signals.
    
    Args:
        ticker (str): Stock ticker symbol
        sma_data (pandas.DataFrame): DataFrame containing SMA indicators
        macd_data (pandas.DataFrame): DataFrame containing MACD indicators
        rsi_data (pandas.DataFrame): DataFrame containing RSI indicators
        
    Returns:
        dict: Dictionary containing technical analysis results
    """
    try:
        # Load OpenAI API key from config
        config = load_config()
        api_key = config.get('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in configuration")
        
        # Set OpenAI API key
        openai.api_key = api_key
        
        # Prepare data for analysis
        latest_price = sma_data['close'].iloc[-1]
        latest_sma_20 = sma_data['sma_20'].iloc[-1]
        latest_sma_50 = sma_data['sma_50'].iloc[-1]
        latest_sma_200 = sma_data['sma_200'].iloc[-1]
        
        # Calculate SMA trends
        sma_20_trend = (latest_sma_20 / sma_data['sma_20'].iloc[-5] - 1) * 100 if len(sma_data) >= 5 else 0
        sma_50_trend = (latest_sma_50 / sma_data['sma_50'].iloc[-5] - 1) * 100 if len(sma_data) >= 5 else 0
        sma_200_trend = (latest_sma_200 / sma_data['sma_200'].iloc[-5] - 1) * 100 if len(sma_data) >= 5 else 0
        
        # MACD data
        latest_macd = macd_data['macd_line'].iloc[-1]
        latest_signal = macd_data['signal_line'].iloc[-1]
        latest_histogram = macd_data['macd_histogram'].iloc[-1]
        
        # MACD trend
        macd_trend = (latest_macd / macd_data['macd_line'].iloc[-5] - 1) * 100 if len(macd_data) >= 5 else 0
        
        # RSI data
        latest_rsi = rsi_data['rsi'].iloc[-1]
        rsi_trend = (latest_rsi / rsi_data['rsi'].iloc[-5] - 1) * 100 if len(rsi_data) >= 5 else 0
        
        # Create prompt for OpenAI
        prompt = f"""
        Analyze the following technical indicators for {ticker} and provide insights:
        
        Current Price: ${latest_price:.2f}
        
        Simple Moving Averages (SMA):
        - SMA (20): ${latest_sma_20:.2f} (Trend: {sma_20_trend:.2f}%)
        - SMA (50): ${latest_sma_50:.2f} (Trend: {sma_50_trend:.2f}%)
        - SMA (200): ${latest_sma_200:.2f} (Trend: {sma_200_trend:.2f}%)
        
        Moving Average Convergence Divergence (MACD):
        - MACD Line: {latest_macd:.2f} (Trend: {macd_trend:.2f}%)
        - Signal Line: {latest_signal:.2f}
        - Histogram: {latest_histogram:.2f}
        
        Relative Strength Index (RSI):
        - RSI: {latest_rsi:.2f} (Trend: {rsi_trend:.2f}%)
        
        Please provide a detailed analysis of the technical indicators, including:
        1. What do the SMA crossovers indicate?
        2. What does the MACD signal suggest?
        3. What does the RSI value indicate about overbought/oversold conditions?
        4. Are there any bullish or bearish signals from these indicators?
        5. What are the key support and resistance levels based on these indicators?
        """
        
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional stock analyst specializing in technical analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        # Extract analysis from response
        technical_analysis = response.choices[0].message.content
        
        return {
            'analysis': technical_analysis,
            'sma_data': {
                'sma_20': latest_sma_20,
                'sma_50': latest_sma_50,
                'sma_200': latest_sma_200,
                'sma_20_trend': sma_20_trend,
                'sma_50_trend': sma_50_trend,
                'sma_200_trend': sma_200_trend
            },
            'macd_data': {
                'macd_line': latest_macd,
                'signal_line': latest_signal,
                'histogram': latest_histogram,
                'macd_trend': macd_trend
            },
            'rsi_data': {
                'rsi': latest_rsi,
                'rsi_trend': rsi_trend
            }
        }
    
    except Exception as e:
        raise Exception(f"Error analyzing technical indicators for {ticker}: {str(e)}")

def analyze_news_sentiment(ticker, news_data):
    """
    Analyze news articles to determine sentiment and potential impact on stock price.
    
    Args:
        ticker (str): Stock ticker symbol
        news_data (list): List of dictionaries containing news articles
        
    Returns:
        dict: Dictionary containing news sentiment analysis results
    """
    try:
        # Load OpenAI API key from config
        config = load_config()
        api_key = config.get('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in configuration")
        
        # Set OpenAI API key
        openai.api_key = api_key
        
        # Format news headlines
        news_headlines = "\n".join([f"- {article['title']}" for article in news_data[:10]])
        
        # Create prompt for OpenAI
        prompt = f"""
        Analyze the following news articles about {ticker} and determine the overall sentiment and potential impact on the stock price:
        
        Recent News Headlines:
        {news_headlines}
        
        Please provide a detailed analysis of the news sentiment, including:
        1. What is the overall sentiment (positive, negative, or neutral)?
        2. Are there any significant events or announcements that could impact the stock price?
        3. What is the potential short-term and long-term impact of these news items?
        4. Are there any emerging trends or themes in the news coverage?
        """
        
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional stock analyst specializing in news sentiment analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        # Extract analysis from response
        news_analysis = response.choices[0].message.content
        
        return {
            'analysis': news_analysis,
            'article_count': len(news_data)
        }
    
    except Exception as e:
        raise Exception(f"Error analyzing news sentiment for {ticker}: {str(e)}")

def generate_final_recommendation(ticker, price_analysis, technical_analysis, news_analysis):
    """
    Generate a final trading recommendation based on all analyses.
    
    Args:
        ticker (str): Stock ticker symbol
        price_analysis (dict): Dictionary containing price analysis results
        technical_analysis (dict): Dictionary containing technical analysis results
        news_analysis (dict): Dictionary containing news sentiment analysis results
        
    Returns:
        dict: Dictionary containing final analysis results
    """
    try:
        # Load OpenAI API key from config
        config = load_config()
        api_key = config.get('OPENAI_API_KEY')
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in configuration")
        
        # Set OpenAI API key
        openai.api_key = api_key
        
        # Extract RSI data
        rsi_value = technical_analysis['rsi_data']['rsi']
        rsi_trend = technical_analysis['rsi_data']['rsi_trend']
        
        # Create prompt for OpenAI
        prompt = f"""
        Based on the following analyses for {ticker}, provide a comprehensive trading recommendation:
        
        PRICE ANALYSIS:
        {price_analysis['analysis']}
        
        TECHNICAL INDICATOR ANALYSIS:
        {technical_analysis['analysis']}
        
        NEWS SENTIMENT ANALYSIS:
        {news_analysis['analysis']}
        
        RSI VALUE: {rsi_value:.2f} (Trend: {rsi_trend:.2f}%)
        
        Please provide a concise final analysis with the following format:
        1. A short summary of the stock's condition
        2. A clear Buy/Sell/Hold decision
        3. Suggested Buy and Sell price ranges
        4. Risk level (Low/Medium/High)
        5. A short reason for the recommendation
        6. Confidence score (%)
        """
        
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a professional stock analyst providing clear, direct trading recommendations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        # Extract analysis from response
        analysis_text = response.choices[0].message.content
        
        # Parse the analysis text to extract structured data
        lines = analysis_text.split('\n')
        
        # Initialize analysis dictionary
        analysis = {
            'summary': '',
            'decision': '',
            'buy_range': [0, 0],
            'sell_range': [0, 0],
            'risk_level': '',
            'reason': '',
            'confidence': 0
        }
        
        # Parse the analysis text (simplified)
        for line in lines:
            if 'summary' in line.lower():
                analysis['summary'] = line.split(':', 1)[1].strip()
            elif 'decision' in line.lower():
                analysis['decision'] = line.split(':', 1)[1].strip()
            elif 'buy range' in line.lower():
                range_str = line.split(':', 1)[1].strip()
                analysis['buy_range'] = [float(x.strip('$').strip()) for x in range_str.split('-')]
            elif 'sell range' in line.lower():
                range_str = line.split(':', 1)[1].strip()
                analysis['sell_range'] = [float(x.strip('$').strip()) for x in range_str.split('-')]
            elif 'risk level' in line.lower():
                analysis['risk_level'] = line.split(':', 1)[1].strip()
            elif 'reason' in line.lower():
                analysis['reason'] = line.split(':', 1)[1].strip()
            elif 'confidence' in line.lower():
                confidence_str = line.split(':', 1)[1].strip()
                analysis['confidence'] = int(confidence_str.strip('%'))
        
        # Add detailed analyses to the result
        analysis['price_analysis'] = price_analysis
        analysis['technical_analysis'] = technical_analysis
        analysis['news_analysis'] = news_analysis
        
        return analysis
    
    except Exception as e:
        raise Exception(f"Error generating final recommendation for {ticker}: {str(e)}")

def analyze_stock(ticker, stock_data, sma_data, macd_data, rsi_data, news_data):
    """
    Analyze stock data using OpenAI to generate trading recommendations.
    This function orchestrates the multi-step analysis process.
    
    Args:
        ticker (str): Stock ticker symbol
        stock_data (pandas.DataFrame): DataFrame containing stock price data
        sma_data (pandas.DataFrame): DataFrame containing SMA indicators
        macd_data (pandas.DataFrame): DataFrame containing MACD indicators
        rsi_data (pandas.DataFrame): DataFrame containing RSI indicators
        news_data (list): List of dictionaries containing news articles
        
    Returns:
        dict: Dictionary containing analysis results
    """
    try:
        # Step 1: Analyze price data
        price_analysis = analyze_price_data(ticker, stock_data)
        
        # Step 2: Analyze technical indicators
        technical_analysis = analyze_technical_indicators(ticker, sma_data, macd_data, rsi_data)
        
        # Step 3: Analyze news sentiment
        news_analysis = analyze_news_sentiment(ticker, news_data)
        
        # Step 4: Generate final recommendation
        final_analysis = generate_final_recommendation(ticker, price_analysis, technical_analysis, news_analysis)
        
        return final_analysis
    
    except Exception as e:
        raise Exception(f"Error analyzing stock {ticker}: {str(e)}") 