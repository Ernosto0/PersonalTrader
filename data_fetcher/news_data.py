import requests
from datetime import datetime, timedelta
from utils.config import load_config

def fetch_news_data(ticker, days=7):
    """
    Fetch recent news articles about the given ticker.
    
    Args:
        ticker (str): Stock ticker symbol
        days (int): Number of days to look back for news (default: 7)
        
    Returns:
        list: List of dictionaries containing news article data
    """
    try:
        # Load API key from config
        config = load_config()
        api_key = config.get('NEWS_API_KEY')
        
        if not api_key:
            raise ValueError("NEWS_API_KEY not found in configuration")
        
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Format dates for API
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Construct API URL
        base_url = 'https://newsapi.org/v2/everything'
        params = {
            'q': ticker,
            'from': start_date_str,
            'to': end_date_str,
            'language': 'en',
            'sortBy': 'publishedAt',
            'apiKey': api_key
        }
        
        # Make API request
        response = requests.get(base_url, params=params)
        
        # Check if request was successful
        if response.status_code != 200:
            raise Exception(f"News API request failed with status code {response.status_code}")
        
        # Parse response
        data = response.json()
        
        # Extract relevant information from articles
        articles = []
        for article in data.get('articles', []):
            articles.append({
                'title': article.get('title', ''),
                'description': article.get('description', ''),
                'url': article.get('url', ''),
                'published_at': article.get('publishedAt', ''),
                'source': article.get('source', {}).get('name', '')
            })
        
        return articles
    
    except Exception as e:
        raise Exception(f"Error fetching news data for {ticker}: {str(e)}") 