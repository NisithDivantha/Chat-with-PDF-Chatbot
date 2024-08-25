import streamlit as st

import os
import argparse

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from fill_database import process_pdf

from env import API_KEY
os.environ['GOOGLE_API_KEY'] = API_KEY

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

st.title("Chat with PDF")
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    st.write("Processing the uploaded file...")
    process_pdf(uploaded_file)
    st.success("File processed and saved to the Chroma database.")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def response_generator(prompt):
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_score(prompt,  k = 3)
    # print(results)

    if len(results) == 0:
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

