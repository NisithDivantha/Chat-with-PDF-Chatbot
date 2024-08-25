# Chat-with-PDF-Chatbot 🤖

## Overview

This project implements a chatbot capable of querying PDF documents using Retrieval-Augmented Generation (RAG) to provide optimized responses from a Large Language Model (LLM). The chatbot also includes a user-friendly interface that allows users to upload PDFs and interact with the system through chat.

## Features

- **PDF Upload**: Users can upload PDFs through the interface.
- **RAG-based Querying**: Utilize Retrieval-Augmented Generation to generate accurate and context-aware responses.
- **Optimized LLM Outputs**: The system is designed to provide the most relevant answers using state-of-the-art LLMs.

## Used Technologies

- **LangChain**: For text splitting, document processing, and managing interactions with LLMs.
- **Google Generative AI**: For generating embeddings and processing user queries.
- **Chroma**: As the vector database for storing and retrieving document embeddings.
- **Streamlit**: For creating the user interface, allowing PDF uploads and chat interactions.
- **Python**: The core programming language used for backend development.

## Setup ⚙️

To get started, follow these steps to set up your environment and install the necessary dependencies:

### 1. Create a Virtual Environment

It is recommended to create a virtual environment to manage your dependencies.

```bash
python3 -m venv llm_env
source llm_env/bin/activate
```
### 2. Install Dependencies
Once the virtual environment is activated, install the required dependencies:
```bash
pip install -qU langchain-text-splitters
pip install -q -U google-generativeai
pip install --upgrade --quiet langchain-google-genai
pip install chromadb
pip install -U langchain-chroma
```
After installing all the dependencies use below command to run the application.
```bash
streamlit run app.py
```
### 3. How to Use
- Upload PDFs: Open the application and upload the PDF documents you want to query.
- Interact with the Chatbot: Use the chat interface to ask questions about the content in the uploaded PDFs.
- Receive Optimized Responses: The chatbot will process your query using RAG and provide responses based on the relevant sections of the PDFs.