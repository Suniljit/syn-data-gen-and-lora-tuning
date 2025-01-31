import json
import os

from dotenv import load_dotenv
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

# Load OpenAI API key
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") # Get the API key from the environment variables

# Check that the API key is loaded
if OPENAI_API_KEY is None:
    raise ValueError("OpenAI API key not found in environment variables")

# Load OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Define OpenAI model
EMB_MODEL = "text-embedding-3-small"

# Define data paths
DATA_PATH = "data/training_data.json" # Data to be checked
CLEAN_DATA_PATH = "data/clean_training_data.json" # Cleaned data

# Define the cosine similarity threshold for identifying duplicates
THRESHOLD = 0.99


def main():
    # Load data
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    
    # Generate embeddings for each question and answer pair
    embeddings = [
        {
            "input": pair["input"],
            "output": pair["output"],
            "embedding": get_embeddings(pair["input"] + " " + pair["output"])
        }
        for pair in data
    ]
    
    # Check for duplicates
    similar_pairs = check_duplicates(embeddings, THRESHOLD)
    
    # Remove duplicates
    cleaned_data = remove_duplicates(data, similar_pairs)
    
    # Save cleaned data to a json file
    with open(CLEAN_DATA_PATH, "w") as f:
        json.dump(cleaned_data, f, indent=4, ensure_ascii=False)


def get_embeddings(text: str) -> list:
    """
    Generate embeddings for the given text using a specified model.
    
    Args:
        text (str): The input text for which embeddings are to be generated.
    
    Returns:
        list: A list representing the embeddings of the input text.
    """
    response = client.embeddings.create(
        input=text, 
        model=EMB_MODEL
    )
    
    return response.data[0].embedding
    
def check_duplicates(embeddings: list, threshold: float) -> set:
    """
    Check for duplicate embeddings based on a similarity threshold.
    
    Args:
        embeddings (list): A list of dictionaries containing embeddings.
        threshold (float): A similarity threshold value between 0 and 1.
        
    Returns:
        set: A set of indices of embeddings that are considered similar.
    """
    # Initialize set to store similar pairs
    similar_pairs = set()
    
    # Initialize counter
    count = 0
    
    # Compare embeddings for similiarity and add similar pairs to the set
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sim = cosine_similarity(
                [embeddings[i]["embedding"]],
                [embeddings[j]["embedding"]]
            )[0][0]
            
            if sim > threshold:
                similar_pairs.add(j)
                count += 1
    
    print(f"Found {count} similar pairs")
    
    return similar_pairs
            
def remove_duplicates(data: list, similar_pairs: set) -> list:
    """
    Remove duplicates from a list based on a set of similar pairs.

    Args:
        data (list): The list from which duplicates need to be removed.
        similar_pairs (set): A set of indices representing similar pairs in the list.

    Returns:
        list: A list with duplicates removed based on the similar pairs.
    """
    return [pair for i, pair in enumerate(data) if i not in similar_pairs]


if __name__ == "__main__":
    main()