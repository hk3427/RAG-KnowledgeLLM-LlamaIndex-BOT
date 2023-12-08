import logging
from llama_index import SimpleDirectoryReader
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index.storage.storage_context import StorageContext
from IPython.display import Markdown, display
from llama_index.node_parser import SentenceSplitter
from embeddings import EmbeddingComponent
import chromadb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileReader:
    def __init__(self, file_paths):
        self.file_paths = file_paths

    def read_data(self):
        logger.info("Reading data from files: %s", self.file_paths)
        file_reader = SimpleDirectoryReader(input_files=self.file_paths)
        return file_reader.load_data()

    def parse_data(self, data):
        logger.info("Parsing data")
        node_parser = SentenceSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separator=" ",
            paragraph_separator="\n\n\n",
            secondary_chunking_regex="[^,.;。]+[,.;。]?"
        )
        return node_parser.get_nodes_from_documents(data)


class DatabaseManager:
    def __init__(self, db_path, collection_name):
        self.db_path = db_path
        self.collection_name = collection_name

    def initialize_db(self):
        logger.info("Initializing the database at path: %s", self.db_path)
        db = chromadb.PersistentClient(path=self.db_path)
        return db.get_or_create_collection(self.collection_name)


class VectorIndexer:
    def __init__(self, nodes, vector_store, embedding_model):
        self.nodes = nodes
        self.vector_store = vector_store
        self.embedding_model = embedding_model

    def create_index(self):
        logger.info("Creating the vector index")
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        service_context = ServiceContext.from_defaults(embed_model=self.embedding_model, llm=None)
        return VectorStoreIndex(
            self.nodes, storage_context=storage_context, service_context=service_context
        )



def main():
    # User Inputs
    file_path = input("Enter the path to the file: ")
    db_path = input("Enter the path to the database: ")
    collection_name = input("Enter the database collection name: ")

    # File Reading
    file_reader = FileReader(file_paths=[file_path])
    file_data = file_reader.read_data()
    file_nodes = file_reader.parse_data(file_data)

    # Database Initialization
    db_manager = DatabaseManager(db_path=db_path, collection_name=collection_name)
    db_collection = db_manager.initialize_db()

    # Embedding Model Initialization
    embedding_mode = input("Enter the embedding mode ('local' or 'openai'): ")
    hf_model_name = input("Enter the Hugging Face model name: ")
    embedding_model = EmbeddingComponent(embedding_mode, hf_model_name)

    # Vector Store and Index Creation
    vector_store = ChromaVectorStore(chroma_collection=db_collection)
    indexer = VectorIndexer(nodes=file_nodes, vector_store=vector_store, embedding_model=embedding_model.embedding_model)
    index = indexer.create_index()

if __name__ == "__main__":
    main()
