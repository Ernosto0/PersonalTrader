import os
from pathlib import Path
from dotenv import load_dotenv

def load_config():
    """
    Load configuration settings from .env file.
    
    Returns:
        dict: Dictionary containing configuration settings
    """
    try:
        # Get the directory of the current file
        current_dir = Path(__file__).parent.parent
        
        # Path to .env file
        env_path = current_dir / '.env'
        
        # Check if .env file exists
        if not env_path.exists():
            # Create .env file from example if it doesn't exist
            example_path = current_dir / '.env.example'
            if example_path.exists():
                with open(example_path, 'r') as example_file:
                    with open(env_path, 'w') as env_file:
                        env_file.write(example_file.read())
                print(f"Created .env file at {env_path}")
                print("Please add your API keys to the .env file.")
            else:
                print("Warning: .env file not found and .env.example is not available.")
        
        # Load environment variables from .env file
        load_dotenv(env_path)
        
        # Create config dictionary from environment variables
        config = {
            'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY', ''),
            'NEWS_API_KEY': os.getenv('NEWS_API_KEY', '')
        }
        
        return config
    
    except Exception as e:
        raise Exception(f"Error loading configuration: {str(e)}") 