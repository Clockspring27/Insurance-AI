# Insurance Helper

An intelligent RAG (Retrieval-Augmented Generation) application that helps users understand complex insurance documents through natural language queries.

## Overview

Insurance Helper uses advanced AI to analyze insurance policy documents and answer questions in plain language. It automatically detects the intent of your questions and searches the most relevant sections of your documents.

## Features

- **Smart Query Detection**: Automatically identifies whether you're asking about exclusions, coverage, add-ons, or general terms
- **PDF Processing**: Upload and process insurance policy documents
- **Semantic Search**: Uses vector embeddings for accurate information retrieval
- **Natural Language Interface**: Ask questions in plain English
- **Source Citations**: Every answer includes references to source documents with page numbers
- **Section-Specific Search**: Intelligently routes queries to relevant document sections (exclusions, coverage, add-ons, etc.)

## Tech Stack

- **Frontend**: Streamlit
- **LLM**: Google Gemini (gemini-2.5-flash)
- **Embeddings**: Google Gemini (models/text-embedding-004")
- **Vector Database**: Qdrant Cloud
- **Framework**: LangChain
- **PDF Processing**: PyPDF

## Project Structure

```
insurance-helper/
├── app.py                     # Streamlit application
├── config.py                  # Configuration settings
├── .env                       # Environment variables
├── requirements.txt           # Python dependencies
├── PDF/                       # PDF Folder
    ├── 60_GEN893.pdf                         
└── utils/
    ├── pdf_processor.py       # PDF loading and chunking
    ├── vector_store.py        # Qdrant vector store operations
    └── rag_chain.py           # RAG pipeline implementation
```
**Web Demo**

check out the live web app click [here](https://huggingface.co/spaces/Clocksp/Insurance-AI)

Example questions are present in the left hand side in **How Smart Mode Works**
## Features Breakdown

### Automatic Section Detection
Documents are automatically analyzed and tagged:
- **Exclusions**: Clauses about what's NOT covered
- **Coverage**: Details about what IS covered
- **Claims**: Claim procedures and requirements
- **Pricing**: Premium and cost information

### Source Attribution
Every answer includes:
- Source document filename
- Page number
- Section type
- Content preview (first 200 characters)

⚠️ **NOTE**
---
This tool is for informational and educational purposes only. It should NOT be used as the sole basis for making insurance decisions. Always:

- Consult with licensed insurance professionals for advice
- Read your complete policy documents carefully
- Verify all information with your insurance provider
- Seek professional guidance before making coverage decisions

The AI-generated responses may contain errors or misinterpretations. This tool does not replace professional insurance advice.

## License
MIT License

