import getpass
import os
import shutil
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma




from env import API_KEY
os.environ['GOOGLE_API_KEY'] = API_KEY

# def main():
#     documents = load_documents()
#     chunks = text_splitter(documents)
#     fill_chroma(chunks)

# def load_documents():

#     pdf_loader = PyPDFDirectoryLoader(pdf_path)
#     return pdf_loader.load()

def process_pdf(pdf_file):
    # Save the uploaded file temporarily

    temp_pdf_path = 'temp_data'

    if os.path.exists(temp_pdf_path):
        shutil.rmtree(temp_pdf_path)
    os.makedirs(temp_pdf_path)
    
    pdf_path = os.path.join(temp_pdf_path, 'uploaded.pdf')
    with open(pdf_path, 'wb') as f:
        f.write(pdf_file.getbuffer())
    
    # Load, split, and process the PDF
    pdf_loader = PyPDFDirectoryLoader(temp_pdf_path)
    documents = pdf_loader.load()
    chunks = text_splitter(documents)
    fill_chroma(chunks)
    
    # # Clean up the temporary directory
    shutil.rmtree(temp_pdf_path)

def text_splitter(documents: list[Document]):
    splitter = RecursiveCharacterTextSplitter(
    chunk_size=600,
    chunk_overlap=60,
    length_function=len,
    is_separator_regex=False,
    )
    return splitter.split_documents(documents)

def fill_chroma(chunks):
    
    CHROMA_PATH = os.path.abspath('chroma')
    # if os.path.exists(CHROMA_PATH):
    #     shutil.rmtree(CHROMA_PATH)
    print("filling chroma")
    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    print("filled chroma")
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=embedding_function
    )
    
    db.add_documents(chunks)


    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")
    all_docs = db._collection.get()["documents"]
    num_docs = len(all_docs)
    print(f"Number of documents in the Chroma database: {num_docs}")


# if __name__ == '__main__':
#     main()