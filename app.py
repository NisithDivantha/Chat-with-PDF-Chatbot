import streamlit as st

import os
import argparse

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from fill_database import process_pdf
from langchain.schema import Document 
from env import API_KEY
os.environ['GOOGLE_API_KEY'] = API_KEY

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
You are a helpful assistant. Answer the following question based on the provided context:

{context}

Answer the question based on the above context: {question}
"""


import hashlib

def generate_pdf_identifier(pdf_file):
    file_hash = hashlib.md5(pdf_file.getbuffer()).hexdigest()
    return file_hash

def process_pdf_once(uploaded_file):
    CHROMA_PATH = "chroma"
    pdf_identifier = generate_pdf_identifier(uploaded_file)
    
    # Check if the database exists
    if os.path.exists(CHROMA_PATH):
        db = Chroma(persist_directory=CHROMA_PATH)
        
        # Check if the PDF identifier is already in the database
        all_docs = db._collection.get()["documents"]
        if pdf_identifier in all_docs:
            print("PDF already processed. Skipping.")
            return
    else:
        os.makedirs(CHROMA_PATH)
    
    # If not found, process the PDF
    process_pdf(uploaded_file)
    
    # Save the identifier to the database
    # db.add_documents([Document(page_content="PDF processed", metadata={"identifier": pdf_identifier})])
    # print("PDF processed and saved.")


st.title("Chat with PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.write("Processing the uploaded file...")
    process_pdf_once(uploaded_file)
    st.success("File processed and saved to the Chroma database.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def response_generator(prompt):
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    normalized_prompt = prompt.lower()

    print("prompt", prompt)
    results = db.similarity_search_with_score(normalized_prompt,  k = 10)
    print(results)

    if len(results) == 0 or results[0][1] < 0.5:
        yield "No results found."


    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    gemini_prompt = prompt_template.format(context=context_text, question=prompt)
    # print(prompt)

    model = GoogleGenerativeAI(model="gemini-pro", temperature= 0.7)
    response = model.predict(gemini_prompt)
    yield response


if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        response = st.write_stream(response_generator(prompt))
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

