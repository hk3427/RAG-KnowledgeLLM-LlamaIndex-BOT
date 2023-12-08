import chromadb
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.vector_stores import ChromaVectorStore
from llama_index import ServiceContext
from embeddings import EmbeddingComponent
from llm import LLMComponent 

def initialize_chromadb(db_path, collection_name):
    db = chromadb.PersistentClient(path=db_path)
    return db.get_or_create_collection(collection_name)

def initialize_vector_store(chroma_collection):
    return ChromaVectorStore(chroma_collection=chroma_collection)

def initialize_service_context(embedding_model, llm_instance):
    return ServiceContext.from_defaults(embed_model=embedding_model, llm=llm_instance)

def initialize_vector_store_index(vector_store, service_context):
    return VectorStoreIndex.from_vector_store(vector_store, service_context=service_context)

def main():
    # User Inputs
    db_path = input("Enter the path to ChromaDB: ")
    collection_name = input("Enter the ChromaDB collection name that you provided during ingestion: ")
    llm_mode = input("Enter the LLM mode ('local', 'openai', or 'mock'): ")
    llm_model_path = input("Enter the path to the LLM model: ")

    # Initialize ChromaDB
    chroma_collection = initialize_chromadb(db_path, collection_name)

    # Initialize Vector Store
    vector_store = initialize_vector_store(chroma_collection)

    # Initialize LLM
    llm_component = LLMComponent(llm_mode, llm_model_path)
    llm_instance = llm_component.llm

    # Initialize Embedding Model
    embedding_mode = input("Enter the embedding mode ('local' or 'openai') that you provided during ingestion: ")
    hf_model_name = input("Enter the Hugging Face model name that you provided during ingestion: ")
    embedding_component = EmbeddingComponent(embedding_mode, hf_model_name)
    embedding_model = embedding_component.embedding_model

    # Initialize Service Context
    service_context = initialize_service_context(embedding_model, llm_instance)

    # Initialize Vector Store Index
    index = initialize_vector_store_index(vector_store, service_context)

    # Query Engine
    query_engine = index.as_query_engine(streaming=True)

    # Query and Print Response
    query = input("Enter your query: ")
    response = query_engine.query(query)
    response.print_response_stream()

if __name__ == "__main__":
    main()