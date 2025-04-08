# AI Stock Trading Assistant

A command-line AI stock trading assistant that provides clear, direct trading suggestions based on technical indicators and news data.

## Features

- Fetches 2 years of stock price data using yfinance
- Calculates technical indicators:
  - Simple Moving Average (SMA)
  - Moving Average Convergence Divergence (MACD)
- Pulls latest news headlines about the stock
- Uses OpenAI to generate trading recommendations through a multi-step analysis process:
  1. Analyzes historical price data for patterns and trends
  2. Evaluates technical indicators for trading signals
  3. Assesses news sentiment and potential impact
  4. Combines all analyses for a comprehensive final recommendation
- Provides clear Buy/Sell/Hold decisions with price ranges
- Includes risk level assessment and confidence score

## Installation

1. Clone this repository:
```
git clone <repository-url>
cd ai-stock-assistant
```

2. Install the required dependencies:
```
pip install -r requirements.txt
```

3. Set up your API keys:
   - Copy the `.env.example` file to `.env`:
   ```
   cp .env.example .env
   ```
   - Edit the `.env` file and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   NEWS_API_KEY=your_news_api_key_here
   ```

## Usage

Run the assistant with a stock ticker symbol:

```
python main.py --ticker AMZN
```

To see detailed analysis from each step of the process:

```
python main.py --ticker AMZN --detailed
```

The assistant will:
1. Fetch historical stock data
2. Calculate technical indicators
3. Gather recent news
4. Perform a multi-step AI analysis:
   - Analyze price data for patterns and trends
   - Evaluate technical indicators for trading signals
   - Assess news sentiment and potential impact
   - Generate a comprehensive final recommendation

## Example Output

```
=== Stock Analysis Report ===
Ticker: AMZN
Summary: Amazon shows strong growth potential with positive technical indicators and favorable news.
Decision: Buy
Buy Range: $145.00 - $150.00
Sell Range: $165.00 - $170.00
Risk Level: Medium
Reason: Strong technical indicators and positive news sentiment indicate upward momentum.
Confidence: 75%

=== Detailed Analysis ===

--- Price Analysis ---
[Detailed price analysis text]

--- Technical Indicator Analysis ---
[Detailed technical analysis text]

--- News Sentiment Analysis ---
[Detailed news sentiment analysis text]
```

## Requirements

- Python 3.8+
- OpenAI API key
- News API key

## License

MIT 