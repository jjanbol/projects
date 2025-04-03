import chromadb
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_vertexai import VertexAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document 


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./key.json"
persist_directory = "./chromaDB_v2"
embedding_model = VertexAIEmbeddings(model_name="text-embedding-004")
vector_store = Chroma(
    collection_name="code_langs",
    embedding_function=embedding_model,
    persist_directory=persist_directory
)


def retrieval_augmented_generation(query):
    results = vector_store.similarity_search(query, k=1)  
    meta_key = []
    meta_value = []
    for i, result in enumerate(results):
        for key, value in result.metadata.items():
            meta_key.append(key)
            meta_value.append(value)

    return results, meta_key, meta_value

