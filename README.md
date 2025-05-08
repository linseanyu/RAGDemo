# RAG Demo

A Retrieval-Augmented Generation (RAG) demonstration project using LangChain, FAISS, and Hugging Face models.

## Overview

This project demonstrates a RAG pipeline that:
1. Processes product and conversation data
2. Chunks and embeds the data using Hugging Face models
3. Stores embeddings in a FAISS vector database
4. Retrieves relevant context for queries
5. Generates responses using a language model

## Files

- `rag.py`: Main implementation of the RAG pipeline
- `generate_data.py`: Script to generate sample data
- `requirements.txt`: Project dependencies

## Setup

1. Create a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Run the RAG pipeline:
```bash
python rag.py
``` 