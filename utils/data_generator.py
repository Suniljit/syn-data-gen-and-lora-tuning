from typing import List

from openai import OpenAI
from pydantic import BaseModel


class DataGenerator:
    """
    A class to generate Q&A pairs using the OpenAI API.
    """
    def __init__(self, api_key: str = None, model: str = "gpt-4o-mini", system_prompt: str = None):
        """
        Initializes the data generator with the specified API key, model, and system prompt.
        
        Args:
            api_key (str): The API key for accessing the OpenAI service.
            model (str, optional): The model to use for generating data. Defaults to "gpt-4o-mini".
            system_prompt (str, optional): The system prompt to use for the model. Defaults to None.
        
        Raises:
            ValueError: If the API key is not provided.
        """
        if api_key is None:
            raise ValueError("OpenAI API key not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.system_prompt = system_prompt
    
    # Pydantic models to structure data
    class QAEntry(BaseModel):
        """
        QAEntry is a data model representing a question-answer pair.
        """
        input: str
        output: str
        
    class QAData(BaseModel):
        """
        QAData is a model representing a collection of QA entries.
        """
        entries: List["DataGenerator.QAEntry"]    
        
        
    def generate_data(self, chunks: List) -> List:
        """
        Generates a dataset by processing chunks of data.
        
        Args:
            chunks (List): A list of data chunks to be processed.
            
        Returns:
            List: A list of processed data in JSON format.
        """
        dataset = []
        for chunk in chunks:
            syn_data = self._generate_qa_pairs(chunk)
            dataset.append(syn_data.choices[0].message.parsed.entries)
        
        json_format = self._process_response(dataset)
        
        return json_format    
        
    
    def _generate_qa_pairs(self, chunk: str) -> QAData:
        """
        Generates question-answer pairs from a given text chunk using the chat completion API.

        Args:
            chunk (str): The text chunk from which to generate QA pairs.

        Returns:
            QAData: The generated question-answer pairs in a list.
        """
        response = self.client.beta.chat.completions.parse(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": chunk.text},
            ],
            response_format=self.QAData,
        )
        
        return response
    
    @staticmethod
    def _process_response(dataset: QAData) -> List:
        """
        Processes a QAData dataset and converts it into a list of JSON-serializable dictionaries.
        
        Args:
            dataset (QAData): The dataset containing QA entries to be processed.
            
        Returns:
            List: A list of dictionaries where each dictionary is the JSON-serializable 
                  representation of a QA entry in the dataset.
        """
        json_data = []
        for data_index in range(len(dataset)):
            for qa_entry in range(len(dataset[data_index])):
                json_data.append(dataset[data_index][qa_entry].model_dump())
        
        return json_data        
    