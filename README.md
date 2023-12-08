## Prerequisites

- Python 3.7 or higher
- Dependencies mentioned in requirements.txt
- ChromaDB instance running (if applicable)

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/hk3427/PDF-Knowledge-LLM-Bot)
    ```

2. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Set up environment variables (if needed):

    ```bash
    export OPENAI_API_KEY=your_openai_api_key
    ```
4. Donwload the LLAMA model locally

   Link : https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/blob/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf

## Usage

### Ingest Documents
Use `ingest.py` to ingest documents into the system.

```bash
python ingest.py
```

The system will prompt you to enter the required information:

File Path : Enter the path to the file to be ingested. 

Path to ChromaDB: Enter the path to ChromaDB.

ChromaDB Collection Name: Enter the ChromaDB collection name.

Embedding Mode ('local' or 'openai'): Enter the embedding mode. ()

Hugging Face Model Name (For 'local' embedding mode): Enter the Hugging Face model name. (Try:BAAI/bge-small-en-v1.5)


### Run the system
Use main.py to run the system, perform queries, and retrieve results.

```bash
python main.py
```

The system will prompt you to enter the required information:

Path to ChromaDB: Enter the path to ChromaDB that you used during ingestion process.

ChromaDB Collection Name: Enter the ChromaDB collection name that you used during ingestion process.

LLM Mode ('local', 'openai', or 'mock'): Enter the LLM mode.

Path to the LLM Model (For local mode): Enter the path to the LLM model.

Embedding Mode ('local' or 'openai'): Enter the embedding mode that you used during ingestion process.

Hugging Face Model Name (For 'local' embedding mode): Enter the Hugging Face model name that you used during ingestion process. (Try:BAAI/bge-small-en-v1.5)

Enter your query: Enter your query for the system.

### Notes

1. This is a working pipeline, and the execution speed may vary depending on your system configuration.
  
2. More experiments are planned to optimize and improve the speed locally in the future.

