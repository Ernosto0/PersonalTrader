from utils.logging import get_logger

# Get logger for this module
logger = get_logger('ai_analyzer')

# Token pricing per 1000 tokens (in USD)
TOKEN_PRICING = {
    "gpt-3.5-turbo": {"input": 0.0005, "output": 0.0015},
    "gpt-4o-mini": {"input": 0.0015, "output": 0.0060},
    "gpt-4": {"input": 0.03, "output": 0.06},
    "gpt-4o": {"input": 0.01, "output": 0.03}
}

# Global token usage tracker
token_usage = {
    "total_tokens": 0,
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "cost": 0.0,
    "calls": 0
}

def get_current_model():
    """
    Get the current model from ai_model.py
    
    Returns:
        str: Current model name
    """
    # Import here to avoid circular imports
    from ai_analyzer.ai_model import MODAL
    return MODAL

def track_token_usage(response):
    """
    Track token usage from an OpenAI API response
    
    Args:
        response: OpenAI API response object
        
    Returns:
        None
    """
    global token_usage
    
    # Get usage data from response
    usage = response.usage
    tokens_used = usage.total_tokens
    prompt_tokens = usage.prompt_tokens
    completion_tokens = usage.completion_tokens
    
    # Update global tracker
    token_usage["total_tokens"] += tokens_used
    token_usage["prompt_tokens"] += prompt_tokens
    token_usage["completion_tokens"] += completion_tokens
    token_usage["calls"] += 1
    
    # Get the model from the response
    model_name = response.model
    
    # Extract base model (e.g., "gpt-4o-mini" from a longer string)
    for model_key in TOKEN_PRICING.keys():
        if model_key in model_name:
            model = model_key
            break
    else:
        # Default to current model if model not found
        model = get_current_model()
    
    # Calculate cost based on model used in the response
    input_cost = (prompt_tokens / 1000) * TOKEN_PRICING[model]["input"]
    output_cost = (completion_tokens / 1000) * TOKEN_PRICING[model]["output"]
    call_cost = input_cost + output_cost
    token_usage["cost"] += call_cost
    
    logger.debug(f"API call using {model} used {tokens_used} tokens (prompt: {prompt_tokens}, completion: {completion_tokens}), cost: ${call_cost:.4f}")

def get_token_usage():
    """
    Get the current token usage statistics
    
    Returns:
        dict: Dictionary containing token usage statistics
    """
    return token_usage

def reset_token_usage():
    """
    Reset the token usage counter
    
    Returns:
        None
    """
    global token_usage
    token_usage = {
        "total_tokens": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "cost": 0.0,
        "calls": 0
    }
