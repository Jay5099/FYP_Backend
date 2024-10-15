import google.generativeai as gemini_ai
import os
from dotenv import load_dotenv
load_dotenv()
gemini_api_key = os.environ.get('API_KEY_GEMINI')
openai_api_key = os.environ.get('API_KEY_OPENAI')
gemini_ai.configure(api_key=gemini_api_key)


def generate_response(prompt):
    model = gemini_ai.GenerativeModel('gemini-1.5-pro')
    response = model.generate_content(prompt)

# print("Response:", response)
    
    # Check if candidates are available
    if not response.candidates:
        return "No response available."

    # Check if content and parts are available
    if not response.candidates[0].content.parts:
        return "No content parts available."

    return response.candidates[0].content.parts[0].text

# from openai import AsyncOpenAI

# # Configure OpenAI API key
# client = AsyncOpenAI(
#     api_key=openai_api_key
# )
# async def generate_response (prompt):
#     try:
#         print("here")
#         # Call OpenAI's GPT-3.5-turbo model using the new API
#         response = await client.chat.completions.create(model="gpt-3.5-turbo",messages=[{"role": "user", "content": prompt}],max_tokens=500)
        
#         # Print and check the response structure
#         print("Response:", response.choices)
        
#         # Extract and return the text from the response
#         if response.choices:
#             return response.choices[0].message.content
#         else:
#             return "No response choices available."
    
#     except Exception as e:
#         return f"An error occurred: {e}"

# Example usage
# prompt = "Describe the significance of the Eiffel Tower."
# result = generate_response(prompt)
# print("Result:", result)

from langchain.embeddings import HuggingFaceEmbeddings

# Initialize the embeddings model
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

import numpy as np
import faiss
from langchain.vectorstores.faiss import FAISS
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import json
from langchain.embeddings.huggingface import HuggingFaceEmbeddings

# Load FAISS index from disk
index = faiss.read_index("faiss_index.idx")

# Load index to docstore ID mapping
with open("index_to_docstore_id.json", "r") as f:
    index_to_docstore_id = json.load(f) 

# Load document texts
with open("docstore.json", "r") as f:
    docstore_dict = json.load(f)

# Reinitialize the InMemoryDocstore
docstore = InMemoryDocstore(docstore_dict)

# Define the embedding function
embedding_function = embeddings.embed_query

# Reinitialize the FAISS vector store
vectorstore = FAISS(
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
    embedding_function=embedding_function
)


# from langchain.chains import RAG

class GoogleGenerativeAI:
    async def __call__(self, prompt):
        return await generate_response(prompt)

# Create a custom LLM using google.generativeai
# google_llm = GoogleGenerativeAI()
# rag_chain = RAG(retriever=vectorstore.as_retriever(), generator=google_llm)

# # Example query
# query = "Tell me about LangChain and Google Generative AI."
# result = rag_chain.run(query)
# print(result)

# Convert the vector store to a retriever
retriever = vectorstore.as_retriever()

def generate_rag_response(query):
    # Generate the embedding for the query
    query_embedding = np.array(embedding_function(query)).astype('float32').reshape(1, -1)  # Ensure it's a 2D array
    # print("query em :",query_embedding)
    # Retrieve relevant document indices based on the query
    distances, retrieved_docs_indices = vectorstore.index.search(query_embedding, k=5)
    print("retrieved documents :", retrieved_docs_indices)
    # Retrieve the actual documents from the docstore using the indices
    retrieved_docs = [docstore.search(str(idx)) for idx in retrieved_docs_indices.flatten()]

    print("res : ",retrieved_docs[0])
    
    # Combine the retrieved documents into a single context
    combined_context = " ".join(retrieved_docs[0])  # Directly join strings
    # print("combined_context : ",combined_context)
    # Generate a response using the generative model
    prompt = f"{combined_context}\n\n give answer in simple way\n Question:\n {query}\nAnswer:"
    print("PROMPT :",prompt)
    response =  generate_response(prompt)
    print("resp : ",response)
    return response

#     llm = OpenAI()

# # Set up the RAG chain with the retriever
#     rag_chain = RetrievalQA.from_chain_type(llm, chain_type="rag", retriever=retriever)

# Example query
# query = "What year was Sardar Vallabhbhai National Institute of Technology (SVNIT) established? and What was SVNIT originally called when it was founded? "
def queryprocess(query):
    # query = "What year was Sardar Vallabhbhai National Institute of Technology (SVNIT) established?"
    # query="In what ways has SVNIT's research initiatives contributed to solving real-world problems, and how does the institute leverage industry-academic partnerships to drive innovation?"
    # result = generate_rag_response(query)
    # print("result : ",result)
    print(query)
    return f"{query}!"
