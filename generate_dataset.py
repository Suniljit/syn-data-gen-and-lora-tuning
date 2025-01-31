import json
import os

from dotenv import load_dotenv

from utils.data_generator import DataGenerator
from utils.data_loader import DataLoader

# Load OpenAI API key
load_dotenv() # Load environment variables from .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Get the API key from the environment variables

# Check that the API key is loaded
if OPENAI_API_KEY is None:
    raise ValueError("OpenAI API key not found in environment variables")

# Define OpenAI model
MODEL = "gpt-4o-mini"

# Define system prompt for data generator
SYSTEM_PROMPT = """
Generate five Q&A input/output pairs to create LLM training data\n\n\
The output format should have the following structure:\n\n\
Input: A question to Donald Trump\n\
Output: Donald Trump's answer\n\n\
"""

# Define path to save generated data
DATA_PATH = "data/training_data.json"


def main():
    # Intialize data loader
    data_loader = DataLoader()
    
    # Initialize data generator
    data_generator = DataGenerator(
        api_key=OPENAI_API_KEY,
        model=MODEL,
        system_prompt=SYSTEM_PROMPT
    )
    
    # Load data
    chunks = data_loader.prepare_data()
    
    # Generate synthetic data
    dataset = data_generator.generate_data(chunks)

    # Save data to file
    with open(DATA_PATH, "w") as f:
        json.dump(dataset, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()