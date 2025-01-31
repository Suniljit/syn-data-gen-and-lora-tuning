from typing import List

from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter

# Define the path to the raw data directory
DATA_PATH = "raw_data"


class DataLoader:
    def __init__(self, data_path: str = DATA_PATH, chunk_size: int = 500, chunk_overlap: int = 20):
        self.data_path = data_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
    
    def prepare_data(self) -> List:
        """
        This function reads documents from the given data directory, splits them into chunks, 
        and returns the chunks as a list.
        
        Args:
            data_path (str): The path to the directory containing the data.
        
        Returns:
            List: A list of document chunks.
        """
        # Load documents from data directory
        documents = SimpleDirectoryReader(self.data_path).load_data()
        
        # Initialize sentence splitter
        splitter = SentenceSplitter(
            chunk_size=500,
            chunk_overlap=20
        )
        
        # Split documents into chunks
        chunks = splitter.get_nodes_from_documents(documents)
        
        return chunks