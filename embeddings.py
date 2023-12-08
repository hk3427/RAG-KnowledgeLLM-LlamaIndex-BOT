import logging
import os
from pathlib import Path
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index import OpenAIEmbedding 

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY', '')


class EmbeddingComponent:
    def __init__(self, embedding_mode, hf_model_name):
        self.embedding_mode = embedding_mode
        self.hf_model_name = hf_model_name
        self.embedding_model = None  # Initialize to None

        logger.info("Initializing the embedding model in mode=%s", embedding_mode)
        if embedding_mode == "local":
            self._initialize_local_embedding()
        elif embedding_mode == "openai":
            self._initialize_openai_embedding()

    def _initialize_local_embedding(self):
        from llama_index.embeddings import HuggingFaceEmbedding
        self.embedding_model = HuggingFaceEmbedding(model_name=self.hf_model_name)
    def _initialize_openai_embedding(self):
        from llama_index import OpenAIEmbedding
        self.embedding_model = OpenAIEmbedding(api_key=OPENAI_API_KEY)