# Mini Wikipedia - Local RAG Chatbot

A simple Retrieval-Augmented Generation (RAG) chatbot that answers questions using a Wikipedia Q&A dataset. The project runs completely offline using Ollama for both embeddings and text generation.

## Features

- Local embeddings with BGE Base
- Local text generation with Llama 3.2 1B
- Cosine similarity retrieval
- Cached vector database for faster startup
- Short-term conversation memory
- Retrieval evaluation using Hit Rate@3

## Requirements

- Python 3.9+
- Ollama

Install dependencies:

```bash
pip install ollama datasets
```

Pull the required models:

```bash
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf
ollama pull hf.co/bartowski/Llama-3.2-1B-Instruct-GGUF
```

## Run

Start the chatbot:

```bash
python rag.py
```

Evaluate retrieval:

```bash
python evaluate.py
```

## Project Structure

```
rag.py           # Chatbot
evaluate.py      # Retrieval evaluation
vector_db.pkl    # Cached embeddings
```

## Notes

- The vector database is created automatically on the first run.
- Everything runs locally—no API keys or cloud services required.
- Built using the `rag-mini-wikipedia` dataset for learning and demonstration purposes.