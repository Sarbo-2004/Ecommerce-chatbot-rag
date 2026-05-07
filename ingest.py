# 1. Import Library
from langchain.document_loaders import JSONLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import AzureOpenAIEmbeddings
import os

from dotenv import load_dotenv

import sys
import csv

import warnings
warnings.filterwarnings("ignore")
 
load_dotenv()

walmart_csv_loader = CSVLoader(r"E-commerce\data\walmart-products.csv",encoding="utf-8")
walmart_csv_docs = walmart_csv_loader.load()

final_medical_docs = walmart_csv_docs
# final_medical_docs = amazon_csv_docs+lazada_csv_docs+shein_csv_docs+shopee_csv_docs+walmart_csv_docs
 
# 3. RecursiveCharacter Text Splitter
recursive_splitter = RecursiveCharacterTextSplitter(chunk_size = 2000, chunk_overlap = 500,
                                                    separators=["\n\n", "\n", " ", "", ".",",", ";"])
 
recursive_tokens = recursive_splitter.split_documents(final_medical_docs)
 
# 4. Create embeddings using HFEmbeddings

hf_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

 
# 5. Create Vector Store
faiss_store = FAISS.from_documents(documents = recursive_tokens, embedding=hf_embeddings)
 
# persist the vector store
faiss_store.save_local("faiss_index_ecommerce")
 
print("FAISS faiss_index_ecommerce created successfully!")
 
