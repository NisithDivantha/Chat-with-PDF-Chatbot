import os
import argparse

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_google_genai import GoogleGenerativeAI

from env import API_KEY
os.environ['GOOGLE_API_KEY'] = API_KEY

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query = args.query_text

    embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
 

    results = db.similarity_search_with_score(query,  k = 3)
    # print(results)

    if len(results) == 0 or results[0][1] < 0.7:
        print("No results found.")


    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query)
    # print(prompt)

    model = GoogleGenerativeAI(model="gemini-pro", temperature= 0.7)
    response = model.predict(prompt)

    print(response)

if __name__ == '__main__':
    main()