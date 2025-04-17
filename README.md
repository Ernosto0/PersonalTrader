# AI Stock Trading Assistant

A command-line AI stock trading assistant that provides clear, direct trading suggestions based on technical indicators and news data.

## Features

- Fetches 2 years of stock price data using yfinance
- Calculates technical indicators:
  - Simple Moving Average (SMA)
  - Moving Average Convergence Divergence (MACD)
  - Relative Strength Index (RSI)
- Pulls latest news headlines about the stock
- Uses OpenAI to generate trading recommendations through a multi-step analysis process:
  1. Analyzes historical price data for patterns and trends
  2. Evaluates technical indicators for trading signals
  3. Assesses news sentiment and potential impact
  4. Analyzes market correlation with indices
  5. Combines all analyses for a comprehensive final recommendation
- Provides clear Buy/Sell/Hold decisions with price ranges
- Includes risk level assessment and confidence score
- Supports multiple OpenAI models for analysis
- Includes backtesting functionality to evaluate recommendation accuracy
- Tracks token usage and cost information

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

3. Set up your OpenAI API key (two options):
   
   **Option 1**: Using command line:
   ```
   python main.py --addopenaikey "your_openai_api_key_here"
   ```
   
   **Option 2**: Manually create a .env file:
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

### Basic Analysis

Run the assistant with a stock ticker symbol:

```
python main.py --ticker AMZN
```

### Detailed Analysis

To see detailed analysis from each step of the process:

```
python main.py --ticker AMZN --detailed
```

### Select OpenAI Model

Choose from various OpenAI models for your analysis:

```
python main.py --ticker AMZN --model gpt-4o
```

Available models:
- gpt-3.5-turbo (fastest, lowest cost)
- gpt-4o-mini (default, balanced performance/cost)
- gpt-4o (improved reasoning)
- gpt-4 (most comprehensive analysis)

### Track Token Usage

View token usage and estimated cost of the analysis:

```
python main.py --ticker AMZN --show-tokens
```

### Backtesting

Validate the model against historical data:

```
python main.py --ticker AMZN --backtest --periods 5
```

You can also combine options:

```
python main.py --ticker AMZN --backtest --periods 3 --model gpt-4o-mini --show-tokens
```

## Example Output

```
=== Stock Analysis Report ===
Ticker: AMZN
Current Price: $178.75
Previous Close: $177.23
Day Range: $176.50 - $179.88
52 Week Range: $118.35 - $180.25
Volume: 32,459,820

Strategy: Amazon shows strong growth potential with positive technical indicators and favorable news.
Decision: Buy
Buy Range: $175.00 - $180.00
Sell Range: $195.00 - $205.00
Risk Level: Medium
Reason: Strong technical indicators and positive news sentiment indicate upward momentum.
Confidence: 80%

=== Market Correlation ===
Correlation with S&P 500: 0.73
Beta: 1.25
Alpha: 0.0012
Moderate positive correlation with the market
More volatile than the market
```

## Requirements

- Python 3.8+
- OpenAI API key
- News API key (optional for enhanced news fetching)

## License

MIT 