import openai
from utils.config import load_config
from utils.logging import get_logger, log_ai_response, log_final_analysis
from ai_analyzer.track_token_usage import track_token_usage, get_token_usage, reset_token_usage

# Define the default model
MODEL = "gpt-4o-mini"

# Get logger for this module
logger = get_logger('ai_analyzer')

def set_model(model_name):
    """
    Set the global model to use for API calls
    
    Args:
        model_name (str): The OpenAI model name to use
        
    Returns:
        str: The previous model name
    """
    global MODEL
    previous = MODEL
    MODEL = model_name
    logger.info(f"Changed model from {previous} to {MODEL}")
    return previous

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
        logger.info(f"Starting price analysis for {ticker}")
        
        # Load OpenAI API key from config
        openai.api_key = GetOpenaikey()
        
        # Prepare data for analysis
        latest_price = stock_data['close'].iloc[-1]
        highest_price = stock_data['high'].max()
        lowest_price = stock_data['low'].min()
        avg_price = stock_data['close'].mean()
        price_volatility = stock_data['close'].std()
        
        # Add volume analysis
        if 'volume' in stock_data.columns:
            avg_volume = stock_data['volume'].mean()
            latest_volume = stock_data['volume'].iloc[-1]
            volume_change = ((latest_volume / stock_data['volume'].iloc[-5:].mean()) - 1) * 100 if len(stock_data) >= 5 else 0
            volume_trend = "Increasing" if volume_change > 10 else "Decreasing" if volume_change < -10 else "Stable"
            
            logger.debug(f"Volume data prepared: latest={latest_volume:.0f}, avg={avg_volume:.0f}, change={volume_change:.2f}%")
        else:
            avg_volume = 0
            latest_volume = 0
            volume_change = 0
            volume_trend = "Unknown"
            
            logger.debug("Volume data not available in dataset")
        
        logger.debug(f"Price data prepared: latest=${latest_price:.2f}, high=${highest_price:.2f}, low=${lowest_price:.2f}")
        
        # Calculate price change percentages
        price_change_1m = ((stock_data['close'].iloc[-1] / stock_data['close'].iloc[-22]) - 1) * 100 if len(stock_data) >= 22 else 0
        price_change_3m = ((stock_data['close'].iloc[-1] / stock_data['close'].iloc[-66]) - 1) * 100 if len(stock_data) >= 66 else 0
        price_change_6m = ((stock_data['close'].iloc[-1] / stock_data['close'].iloc[-132]) - 1) * 100 if len(stock_data) >= 132 else 0
        price_change_1y = ((stock_data['close'].iloc[-1] / stock_data['close'].iloc[-252]) - 1) * 100 if len(stock_data) >= 252 else 0
        price_change_2y = ((stock_data['close'].iloc[-1] / stock_data['close'].iloc[0]) - 1) * 100
        
        logger.debug(f"Price changes calculated: 1m={price_change_1m:.2f}%, 3m={price_change_3m:.2f}%, 6m={price_change_6m:.2f}%, 1y={price_change_1y:.2f}%, 2y={price_change_2y:.2f}%")
        
        # Create prompt for OpenAI
        prompt = f"""
        Analyze the following historical price data for {ticker} and provide insights:
        
        Current Price: ${latest_price:.2f}
        Highest Price (2 years): ${highest_price:.2f}
        Lowest Price (2 years): ${lowest_price:.2f}
        Average Price (2 years): ${avg_price:.2f}
        Price Volatility: ${price_volatility:.2f}
        
        Volume Information:
        - Latest Volume: {latest_volume:,.0f} shares
        - Average Volume: {avg_volume:,.0f} shares
        - Volume Change (vs 5-day avg): {volume_change:.2f}%
        - Volume Trend: {volume_trend}
        
        Price Changes:
        - 1 Month: {price_change_1m:.2f}%
        - 3 Months: {price_change_3m:.2f}%
        - 6 Months: {price_change_6m:.2f}%
        - 1 Year: {price_change_1y:.2f}%
        - 2 Years: {price_change_2y:.2f}%
        
        Please provide a detailed analysis of the price trends, patterns, and potential price levels for support and resistance.
        Focus on identifying key price levels, trend direction, any notable patterns in the price history, and how volume trends may confirm or contradict price movements.
        """
        
        logger.debug("Sending price analysis request to OpenAI")
        logger.debug(f"Price analysis prompt:\n{prompt}")
        
        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a professional stock analyst specializing in technical price analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        # Track token usage
        track_token_usage(response)
        
        # Extract analysis from response
        price_analysis = response.choices[0].message.content
        logger.debug("Received price analysis from OpenAI")
        
        # Log AI response in a formatted way
        log_ai_response(logger, ticker, "price", price_analysis)
        
        logger.info(f"Completed price analysis for {ticker}")
        
        return {
            'analysis': price_analysis,
            'current_price': latest_price,
            'highest_price': highest_price,
            'lowest_price': lowest_price,
            'avg_price': avg_price,
            'price_volatility': price_volatility,
            'volume_data': {
                'latest_volume': latest_volume,
                'avg_volume': avg_volume,
                'volume_change': volume_change,
                'volume_trend': volume_trend
            },
            'price_changes': {
                '1m': price_change_1m,
                '3m': price_change_3m,
                '6m': price_change_6m,
                '1y': price_change_1y,
                '2y': price_change_2y
            }
        }
    
    except Exception as e:
        logger.error(f"Error analyzing price data for {ticker}: {str(e)}", exc_info=True)
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
        logger.info(f"Starting technical indicator analysis for {ticker}")
        
        # Set OpenAI API key
        openai.api_key = GetOpenaikey()
        
        # Prepare data for analysis
        latest_price = sma_data['close'].iloc[-1]
        latest_sma_20 = sma_data['sma_20'].iloc[-1]
        latest_sma_50 = sma_data['sma_50'].iloc[-1]
        latest_sma_200 = sma_data['sma_200'].iloc[-1]
        
        # Calculate SMA trends
        sma_20_trend = (latest_sma_20 / sma_data['sma_20'].iloc[-5] - 1) * 100 if len(sma_data) >= 5 else 0
        sma_50_trend = (latest_sma_50 / sma_data['sma_50'].iloc[-5] - 1) * 100 if len(sma_data) >= 5 else 0
        sma_200_trend = (latest_sma_200 / sma_data['sma_200'].iloc[-5] - 1) * 100 if len(sma_data) >= 5 else 0
        
        logger.debug(f"SMA data prepared: SMA20=${latest_sma_20:.2f}, SMA50=${latest_sma_50:.2f}, SMA200=${latest_sma_200:.2f}")
        
        # MACD data
        latest_macd = macd_data['macd_line'].iloc[-1]
        latest_signal = macd_data['signal_line'].iloc[-1]
        latest_histogram = macd_data['macd_histogram'].iloc[-1]
        
        # MACD trend
        macd_trend = (latest_macd / macd_data['macd_line'].iloc[-5] - 1) * 100 if len(macd_data) >= 5 else 0
        
        logger.debug(f"MACD data prepared: MACD={latest_macd:.2f}, Signal={latest_signal:.2f}, Histogram={latest_histogram:.2f}")
        
        # RSI data
        latest_rsi = rsi_data['rsi'].iloc[-1]
        rsi_trend = (latest_rsi / rsi_data['rsi'].iloc[-5] - 1) * 100 if len(rsi_data) >= 5 else 0
        
        logger.debug(f"RSI data prepared: RSI={latest_rsi:.2f}, Trend={rsi_trend:.2f}%")
        
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
        
        Key Price-to-SMA Relationships:
        - Price to SMA(20) Ratio: {(latest_price/latest_sma_20):.2f}
        - Price to SMA(50) Ratio: {(latest_price/latest_sma_50):.2f}
        - Price to SMA(200) Ratio: {(latest_price/latest_sma_200):.2f}
        
        SMA Relationships:
        - SMA(20) to SMA(50) Ratio: {(latest_sma_20/latest_sma_50):.2f}
        - SMA(50) to SMA(200) Ratio: {(latest_sma_50/latest_sma_200):.2f}
        
        Please provide a detailed analysis of the technical indicators, including:
        1. What do the SMA crossovers indicate?
        2. What does the MACD signal suggest?
        3. What does the RSI value indicate about overbought/oversold conditions?
        4. Are there any bullish or bearish signals from these indicators?
        5. What are the key support and resistance levels based on these indicators?
        6. Is there a golden cross or death cross formation?
        7. How do the price-to-SMA relationships inform your analysis?
        """
        
        logger.debug("Sending technical analysis request to OpenAI")
        logger.debug(f"Technical analysis prompt:\n{prompt}")
        
        # Call OpenAI API with global MODEL
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a professional stock analyst specializing in technical analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        # Track token usage
        track_token_usage(response)
        
        # Extract analysis from response
        technical_analysis = response.choices[0].message.content
        logger.debug("Received technical analysis from OpenAI")
        
        # Log AI response in a formatted way
        log_ai_response(logger, ticker, "technical", technical_analysis)
        
        logger.info(f"Completed technical indicator analysis for {ticker}")
        
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
        logger.error(f"Error analyzing technical indicators for {ticker}: {str(e)}", exc_info=True)
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
        logger.info(f"Starting news sentiment analysis for {ticker}")
        
        openai.api_key = GetOpenaikey()
        
        # Format news headlines with summaries when available
        news_content = []
        for idx, article in enumerate(news_data[:10]):
            headline = f"- {article['title']}"
            if 'summary' in article and article['summary']:
                summary = article['summary'].strip()
                if len(summary) > 200:  # Truncate long summaries
                    summary = summary[:197] + "..."
                headline += f"\n  Summary: {summary}"
            elif 'description' in article and article['description']:
                desc = article['description'].strip()
                if len(desc) > 200:  # Truncate long descriptions
                    desc = desc[:197] + "..."
                headline += f"\n  Description: {desc}"
                
            # Add source and date if available
            if 'source' in article and article['source']:
                headline += f" (Source: {article['source']})"
            if 'date' in article and article['date']:
                headline += f" - {article['date']}"
                
            news_content.append(headline)
            
        news_headlines = "\n".join(news_content)
        
        logger.debug(f"News data prepared: {len(news_data)} articles, using top 10 for analysis with enhanced context")
        
        # Create prompt for OpenAI
        prompt = f"""
        Analyze the following news articles about {ticker} and determine the overall sentiment and potential impact on the stock price:
        
        Recent News Headlines and Summaries:
        {news_headlines}
        
        Please provide a detailed analysis of the news sentiment, including:
        1. What is the overall sentiment (positive, negative, or neutral)?
        2. Are there any significant events or announcements that could impact the stock price?
        3. What is the potential short-term and long-term impact of these news items?
        4. Are there any emerging trends or themes in the news coverage?
        5. Rate the reliability and importance of the news sources on a scale of 1-10.
        6. Indicate if any news contradicts market movement or technical indicators.
        """
        
        logger.debug("Sending news sentiment analysis request to OpenAI")
        logger.debug(f"News sentiment analysis prompt:\n{prompt}")
        
        # Call OpenAI API with global MODEL
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a professional stock analyst specializing in news sentiment analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        # Track token usage
        track_token_usage(response)
        
        # Extract analysis from response
        news_analysis = response.choices[0].message.content
        logger.debug("Received news sentiment analysis from OpenAI")
        
        # Log AI response in a formatted way
        log_ai_response(logger, ticker, "news", news_analysis)
        
        logger.info(f"Completed news sentiment analysis for {ticker}")
        
        return {
            'analysis': news_analysis,
            'article_count': len(news_data)
        }
    
    except Exception as e:
        logger.error(f"Error analyzing news sentiment for {ticker}: {str(e)}", exc_info=True)
        raise Exception(f"Error analyzing news sentiment for {ticker}: {str(e)}")

def generate_final_recommendation(ticker, price_analysis, technical_analysis, news_analysis, current_price=None):
    """
    Generate a final trading recommendation based on all analyses.
    
    Args:
        ticker (str): Stock ticker symbol
        price_analysis (dict): Dictionary containing price analysis results
        technical_analysis (dict): Dictionary containing technical analysis results
        news_analysis (dict): Dictionary containing news sentiment analysis results
        current_price (float, optional): Current price of the stock. If None, will use price from price_analysis.
        
    Returns:
        dict: Dictionary containing final analysis results
    """
    try:
        logger.info(f"Generating final trading recommendation for {ticker}")

        openai.api_key = GetOpenaikey()
        
        # Extract RSI data
        rsi_value = technical_analysis['rsi_data']['rsi']
        rsi_trend = technical_analysis['rsi_data']['rsi_trend']
        
        # Get current price (use parameter if provided, otherwise from price_analysis)
        if current_price is None:
            current_price = price_analysis['current_price']
            
        logger.debug(f"Using current price for recommendation: ${current_price}")
        
        # Create prompt for OpenAI
        prompt = f"""
        Based on the following analyses for {ticker}, provide a comprehensive trading recommendation:

        CURRENT PRICE: ${current_price:.2f}
        
        PRICE ANALYSIS:
        {price_analysis['analysis']}
        
        TECHNICAL INDICATOR ANALYSIS:
        {technical_analysis['analysis']}
        
        NEWS SENTIMENT ANALYSIS:
        {news_analysis['analysis']}
        
        RSI VALUE: {rsi_value:.2f} (Trend: {rsi_trend:.2f}%)
        
        Please provide a concise final analysis with the following format:
        1. Give a short strategy for ticker
        2. Decision: A clear Buy/Sell/Hold decision
        3. Buy Range: $X - $Y (specify a clear price range with numbers only)
        4. Sell Range: $X - $Y (specify a clear price range with numbers only)
        5. Risk Level: Low/Medium/High
        6. Reason: A short reason for the recommendation
        7. Confidence: X% (provide a percentage)
        """
        
        logger.debug("Sending final recommendation request to OpenAI")
        logger.debug(f"Final recommendation prompt summary for {ticker} (prompt too long to log in full)")
        
        # Call OpenAI API with global MODEL
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a professional stock analyst providing clear, direct trading recommendations, confident. ALWAYS format price ranges using simple dollar amounts like: $100 - $120. Always include specific numeric price ranges."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        # Track token usage
        track_token_usage(response)
        
        # Extract analysis from response
        analysis_text = response.choices[0].message.content
        logger.debug("Received final recommendation from OpenAI")
        
        # Log AI response in a formatted way
        log_ai_response(logger, ticker, "final", analysis_text)
        
        # Parse the analysis text to extract structured data
        lines = analysis_text.split('\n')
        
        # Initialize analysis dictionary
        analysis = {
            'strategy': '',
            'decision': '',
            'buy_range': [0, 0],
            'sell_range': [0, 0],
            'risk_level': '',
            'reason': '',
            'confidence': 0
        }
        
        # Parse the analysis text line by line
        for line in lines:
            if 'strategy' in line.lower() and ':' in line:
                analysis['strategy'] = line.split(':', 1)[1].strip()
            elif 'decision' in line.lower() and ':' in line:
                analysis['decision'] = line.split(':', 1)[1].strip()
            elif 'buy range' in line.lower() and ':' in line:
                try:
                    range_str = line.split(':', 1)[1].strip()
                    logger.debug(f"Parsing buy range: {range_str}")
                    
                    # Extract numbers using more robust approach
                    import re
                    numbers = re.findall(r'\$?\s*(\d+(?:\.\d+)?)', range_str)
                    logger.debug(f"Extracted numbers from buy range: {numbers}")
                    
                    if len(numbers) >= 2:
                        analysis['buy_range'] = [float(numbers[0]), float(numbers[1])]
                    elif len(numbers) == 1:
                        # If only one number is found, create a range around it
                        value = float(numbers[0])
                        lower = value * 0.95  # 5% below
                        upper = value * 1.05  # 5% above
                        analysis['buy_range'] = [round(lower, 2), round(upper, 2)]
                    else:
                        logger.warning(f"Could not parse buy range properly: {range_str}")
                except Exception as e:
                    logger.warning(f"Error parsing buy range: {str(e)}")
            
            elif 'sell range' in line.lower() and ':' in line:
                try:
                    range_str = line.split(':', 1)[1].strip()
                    logger.debug(f"Parsing sell range: {range_str}")
                    
                    # Extract numbers using more robust approach
                    import re
                    numbers = re.findall(r'\$?\s*(\d+(?:\.\d+)?)', range_str)
                    logger.debug(f"Extracted numbers from sell range: {numbers}")
                    
                    if len(numbers) >= 2:
                        analysis['sell_range'] = [float(numbers[0]), float(numbers[1])]
                    elif len(numbers) == 1:
                        # If only one number is found, create a range around it
                        value = float(numbers[0])
                        lower = value * 0.95  # 5% below
                        upper = value * 1.05  # 5% above
                        analysis['sell_range'] = [round(lower, 2), round(upper, 2)]
                    else:
                        logger.warning(f"Could not parse sell range properly: {range_str}")
                except Exception as e:
                    logger.warning(f"Error parsing sell range: {str(e)}")
            
            elif 'risk level' in line.lower() and ':' in line:
                analysis['risk_level'] = line.split(':', 1)[1].strip()
            elif 'reason' in line.lower() and ':' in line:
                analysis['reason'] = line.split(':', 1)[1].strip()
            elif 'confidence' in line.lower() and ':' in line:
                try:
                    confidence_str = line.split(':', 1)[1].strip()
                    # Extract number from confidence (handle cases like "90%" or "90 percent")
                    import re
                    confidence_match = re.search(r'(\d+)', confidence_str)
                    if confidence_match:
                        analysis['confidence'] = int(confidence_match.group(1))
                    else:
                        logger.warning(f"Could not parse confidence properly: {confidence_str}")
                except Exception as e:
                    logger.warning(f"Error parsing confidence: {str(e)}")
        
        # Additional parsing for missing fields
        # If strategy is missing, use the first paragraph or line that doesn't match any other category
        if not analysis['strategy']:
            logger.debug("Strategy field missing, attempting to extract from text")
            for line in lines:
                if (line and ':' not in line and 
                    'decision' not in line.lower() and 
                    'buy range' not in line.lower() and 
                    'sell range' not in line.lower() and
                    'risk level' not in line.lower() and
                    'reason' not in line.lower() and
                    'confidence' not in line.lower()):
                    analysis['strategy'] = line.strip()
                    logger.debug(f"Extracted strategy from text: {analysis['strategy']}")
                    break
            
            # If still empty, use the first line or decision as fallback
            if not analysis['strategy'] and lines:
                analysis['strategy'] = lines[0].strip() if ':' not in lines[0] else analysis['decision']
                logger.debug(f"Using fallback strategy: {analysis['strategy']}")
        
        # If reason is missing, look for it in the text or use part of the decision as fallback
        if not analysis['reason']:
            logger.debug("Reason field missing, attempting to extract from text")
            # First try to find a line that looks like a reason
            for line in lines:
                if (line and 'because' in line.lower() or 'due to' in line.lower() or 
                    'based on' in line.lower() or 'given' in line.lower()):
                    analysis['reason'] = line.strip()
                    logger.debug(f"Extracted reason from text: {analysis['reason']}")
                    break
            
            # If still empty, use decision as fallback, if decision is long enough
            if not analysis['reason'] and len(analysis['decision']) > 20:  # A reasonable length for a decision that contains reasoning
                analysis['reason'] = analysis['decision']
                logger.debug(f"Using decision as reason: {analysis['reason']}")
            # Or use strategy as fallback
            elif not analysis['reason'] and analysis['strategy']:
                analysis['reason'] = analysis['strategy']
                logger.debug(f"Using strategy as reason: {analysis['reason']}")
        
        # If we couldn't find price ranges in the specific lines, try to extract them from the entire text
        if analysis['buy_range'] == [0, 0] or analysis['sell_range'] == [0, 0]:
            logger.debug("Attempting to extract price ranges from full text")
            import re
            
            # Look for buy/sell patterns in the entire text
            if analysis['buy_range'] == [0, 0]:
                buy_patterns = [
                    r'buy.*?(?:between|range|at|around|near).*?\$\s*(\d+(?:\.\d+)?)\s*(?:to|-|and)\s*\$\s*(\d+(?:\.\d+)?)',
                    r'buy.*?(?:below|under|around|at).*?\$\s*(\d+(?:\.\d+)?)',
                    r'accumulate.*?(?:between|range|at|around|near).*?\$\s*(\d+(?:\.\d+)?)\s*(?:to|-|and)\s*\$\s*(\d+(?:\.\d+)?)',
                    r'entry.*?(?:between|range|at|around|near).*?\$\s*(\d+(?:\.\d+)?)\s*(?:to|-|and)\s*\$\s*(\d+(?:\.\d+)?)'
                ]
                
                for pattern in buy_patterns:
                    matches = re.search(pattern, analysis_text.lower())
                    if matches:
                        group_count = len(matches.groups())
                        if group_count >= 2:
                            analysis['buy_range'] = [float(matches.group(1)), float(matches.group(2))]
                            logger.debug(f"Extracted buy range from full text: {analysis['buy_range']}")
                            break
                        elif group_count == 1:
                            value = float(matches.group(1))
                            analysis['buy_range'] = [round(value * 0.95, 2), round(value * 1.05, 2)]
                            logger.debug(f"Extracted single buy price from full text and created range: {analysis['buy_range']}")
                            break
            
            if analysis['sell_range'] == [0, 0]:
                sell_patterns = [
                    r'sell.*?(?:between|range|at|around|near).*?\$\s*(\d+(?:\.\d+)?)\s*(?:to|-|and)\s*\$\s*(\d+(?:\.\d+)?)',
                    r'sell.*?(?:above|over|around|at).*?\$\s*(\d+(?:\.\d+)?)',
                    r'exit.*?(?:between|range|at|around|near).*?\$\s*(\d+(?:\.\d+)?)\s*(?:to|-|and)\s*\$\s*(\d+(?:\.\d+)?)',
                    r'target.*?(?:between|range|at|around|near).*?\$\s*(\d+(?:\.\d+)?)\s*(?:to|-|and)\s*\$\s*(\d+(?:\.\d+)?)'
                ]
                
                for pattern in sell_patterns:
                    matches = re.search(pattern, analysis_text.lower())
                    if matches:
                        group_count = len(matches.groups())
                        if group_count >= 2:
                            analysis['sell_range'] = [float(matches.group(1)), float(matches.group(2))]
                            logger.debug(f"Extracted sell range from full text: {analysis['sell_range']}")
                            break
                        elif group_count == 1:
                            value = float(matches.group(1))
                            analysis['sell_range'] = [round(value * 0.95, 2), round(value * 1.05, 2)]
                            logger.debug(f"Extracted single sell price from full text and created range: {analysis['sell_range']}")
                            break
        
        # If we still don't have price ranges, use current price as base for estimating ranges
        if analysis['buy_range'] == [0, 0] and analysis['decision'].lower() in ['buy', 'accumulate', 'strong buy']:
            lower = current_price * 0.90  # 10% below current price
            upper = current_price * 0.98  # 2% below current price
            analysis['buy_range'] = [round(lower, 2), round(upper, 2)]
            logger.debug(f"Using estimated buy range based on current price: {analysis['buy_range']}")
        
        if analysis['sell_range'] == [0, 0] and analysis['decision'].lower() in ['sell', 'reduce', 'strong sell']:
            lower = current_price * 1.02  # 2% above current price
            upper = current_price * 1.10  # 10% above current price
            analysis['sell_range'] = [round(lower, 2), round(upper, 2)]
            logger.debug(f"Using estimated sell range based on current price: {analysis['sell_range']}")
        
        logger.debug(f"Parsed recommendation data: Decision={analysis['decision']}, Risk={analysis['risk_level']}, Confidence={analysis['confidence']}%")
        logger.debug(f"Buy Range: {analysis['buy_range']}, Sell Range: {analysis['sell_range']}")
        
        # Add detailed analyses to the result
        analysis['price_analysis'] = price_analysis
        analysis['technical_analysis'] = technical_analysis
        analysis['news_analysis'] = news_analysis
        
        logger.info(f"Completed final trading recommendation for {ticker}")
        
        return analysis
    
    except Exception as e:
        logger.error(f"Error generating final recommendation for {ticker}: {str(e)}", exc_info=True)
        raise Exception(f"Error generating final recommendation for {ticker}: {str(e)}")

def analyze_market_correlation(ticker, stock_data, market_index='SPY'):
    """
    Analyze how the stock correlates with the broader market.
    
    Args:
        ticker (str): Stock ticker symbol
        stock_data (pandas.DataFrame): DataFrame containing stock price data
        market_index (str): Market index to compare with (default: 'SPY')
        
    Returns:
        dict: Dictionary containing correlation analysis results
    """
    try:
        from data_fetcher.stock_data import fetch_stock_data
        import numpy as np
        
        logger.info(f"Starting market correlation analysis for {ticker} vs {market_index}")
        
        # Set OpenAI API key
        openai.api_key = GetOpenaikey()
        
        # Try to fetch market index data for the same date range as the stock data
        start_date = stock_data.index[0]
        end_date = stock_data.index[-1]
        
        try:
            market_data = fetch_stock_data(market_index, start_date=start_date, end_date=end_date)
            logger.debug(f"Retrieved {len(market_data)} data points for {market_index}")
        except Exception as e:
            logger.warning(f"Error fetching data for {market_index}: {str(e)}. Using default 2-year period.")
            market_data = fetch_stock_data(market_index)
        
        # Align data to ensure matching dates
        # Convert to returns to make comparison more meaningful
        stock_returns = stock_data['close'].pct_change().dropna()
        market_returns = market_data['close'].pct_change().dropna()
        
        # Align dates by using only the dates that appear in both datasets
        common_dates = sorted(set(stock_returns.index).intersection(set(market_returns.index)))
        
        if len(common_dates) < 30:
            logger.warning(f"Not enough common dates ({len(common_dates)}) to calculate correlation. Minimum required: 30")
            return {
                'analysis': f"Insufficient data to calculate correlation between {ticker} and {market_index}.",
                'correlation': 0,
                'beta': 0,
                'alpha': 0
            }
        
        aligned_stock = stock_returns.loc[common_dates]
        aligned_market = market_returns.loc[common_dates]
        
        # Calculate correlation, beta, and alpha
        correlation = np.corrcoef(aligned_stock, aligned_market)[0, 1]
        
        # Beta = Covariance(stock, market) / Variance(market)
        beta = np.cov(aligned_stock, aligned_market)[0, 1] / np.var(aligned_market)
        
        # Alpha = Average Stock Return - (Beta * Average Market Return)
        alpha = aligned_stock.mean() - (beta * aligned_market.mean())
        
        logger.debug(f"Calculated correlation: {correlation:.2f}, beta: {beta:.2f}, alpha: {alpha:.4f}")
        
        # Calculate performance metrics
        stock_total_return = ((stock_data['close'].iloc[-1] / stock_data['close'].iloc[0]) - 1) * 100
        market_total_return = ((market_data['close'].iloc[-1] / market_data['close'].iloc[0]) - 1) * 100
        
        stock_volatility = stock_returns.std() * np.sqrt(252) * 100  # Annualized volatility
        market_volatility = market_returns.std() * np.sqrt(252) * 100  # Annualized volatility
        
        # Create prompt for OpenAI
        prompt = f"""
        Analyze the correlation between {ticker} and the market index {market_index}:
        
        Correlation: {correlation:.2f}
        Beta: {beta:.2f}
        Alpha: {alpha:.4f}
        
        Performance Comparison:
        - {ticker} Total Return: {stock_total_return:.2f}%
        - {market_index} Total Return: {market_total_return:.2f}%
        - {ticker} Volatility (annualized): {stock_volatility:.2f}%
        - {market_index} Volatility (annualized): {market_volatility:.2f}%
        
        Please provide a detailed analysis of what these metrics mean for {ticker}:
        1. Interpret the correlation value and what it means for diversification.
        2. Explain what the beta value indicates about the stock's volatility compared to the market.
        3. Analyze the alpha and its implications for the stock's performance.
        4. Based on these metrics, would this stock be suitable for index-tracking, hedging, or diversification?
        5. How might this stock perform in different market conditions (bull, bear, sideways)?
        """
        
        logger.debug("Sending market correlation analysis request to OpenAI")
        logger.debug(f"Market correlation analysis prompt summary for {ticker} (prompt too long to log in full)")
        
        # Call OpenAI API with global MODEL
        response = openai.ChatCompletion.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a professional stock analyst specializing in market correlation and risk analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        # Track token usage
        track_token_usage(response)
        
        # Extract analysis from response
        correlation_analysis = response.choices[0].message.content
        logger.debug("Received market correlation analysis from OpenAI")
        
        # Log AI response in a formatted way
        log_ai_response(logger, ticker, "correlation", correlation_analysis)
        
        logger.info(f"Completed market correlation analysis for {ticker} vs {market_index}")
        
        return {
            'analysis': correlation_analysis,
            'correlation': correlation,
            'beta': beta,
            'alpha': alpha,
            'returns': {
                'stock': stock_total_return,
                'market': market_total_return
            },
            'volatility': {
                'stock': stock_volatility,
                'market': market_volatility
            }
        }
    
    except Exception as e:
        logger.error(f"Error analyzing market correlation for {ticker}: {str(e)}", exc_info=True)
        raise Exception(f"Error analyzing market correlation for {ticker}: {str(e)}")

def analyze_stock(ticker, stock_data, sma_data, macd_data, rsi_data, news_data, current_price=None, model=None):
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
        current_price (float, optional): Current price of the stock. If None, will use price from historical data.
        model (str, optional): OpenAI model to use. If None, will use the default MODEL.
        
    Returns:
        dict: Dictionary containing analysis results
    """
    # Set global MODEL temporarily to the specified model for this analysis
    global MODEL
    original_model = MODEL
    
    try:
        # Use specified model or default if none provided
        if model is not None:
            logger.info(f"Temporarily setting model to {model} for this analysis")
            MODEL = model
            
        logger.info(f"Starting full stock analysis for {ticker} using model: {MODEL}")
        
        # Step 1: Analyze price data
        logger.info(f"Step 1/5: Analyzing price data for {ticker}")
        price_analysis = analyze_price_data(ticker, stock_data)
        
        # Use provided current price if available, otherwise use the one from price_analysis
        if current_price is None:
            current_price = price_analysis['current_price']
        else:
            # Update price_analysis with the current price
            price_analysis['current_price'] = current_price
            
        logger.debug(f"Using current price for analysis: ${current_price}")
        
        # Step 2: Analyze technical indicators
        logger.info(f"Step 2/5: Analyzing technical indicators for {ticker}")
        technical_analysis = analyze_technical_indicators(ticker, sma_data, macd_data, rsi_data)
        
        # Step 3: Analyze news sentiment
        logger.info(f"Step 3/5: Analyzing news sentiment for {ticker}")
        news_analysis = analyze_news_sentiment(ticker, news_data)
        
        # Step 4: Analyze market correlation
        logger.info(f"Step 4/5: Analyzing market correlation for {ticker}")
        correlation_analysis = analyze_market_correlation(ticker, stock_data)
        
        # Step 5: Generate final recommendation
        logger.info(f"Step 5/5: Generating final recommendation for {ticker}")
        final_analysis = generate_final_recommendation(ticker, price_analysis, technical_analysis, news_analysis, current_price)
        
        # Add correlation analysis to final analysis
        final_analysis['correlation_analysis'] = correlation_analysis
        
        # Log final analysis to the dedicated log file
        log_final_analysis(logger, ticker, final_analysis)
        
        logger.info(f"Completed full stock analysis for {ticker}")
        logger.info(f"Final decision for {ticker}: {final_analysis['decision']} (Confidence: {final_analysis['confidence']}%)")
        
        return final_analysis
    
    except Exception as e:
        logger.error(f"Error analyzing stock {ticker}: {str(e)}", exc_info=True)
        raise Exception(f"Error analyzing stock {ticker}: {str(e)}")
    
    finally:
        # Restore original MODEL
        if model is not None:
            logger.info(f"Restoring model back to {original_model}")
            MODEL = original_model

def GetOpenaikey():
    config = load_config()
    api_key = config.get('OPENAI_API_KEY')
    
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in configuration")
    return api_key