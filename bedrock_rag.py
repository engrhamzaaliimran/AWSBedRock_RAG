import os
import getpass
from langchain.document_loaders import WebBaseLoader, UnstructuredURLLoader, NewsURLLoader, SeleniumURLLoader

import tiktoken
import matplotlib.pyplot as plt
import pandas as pd

from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA


from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter

import boto3
import json
import os
import sys
import numpy as np


from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper

# We will be using the Titan Embeddings Model to generate our Embeddings.
from langchain.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock

website = "https://emumba.com/"


module_path = ".."
sys.path.append(os.path.abspath(module_path))
from utils import bedrock, print_ww

bedrock_client = bedrock.get_bedrock_client(
    assumed_role=os.environ.get("BEDROCK_ASSUME_ROLE", None),
    region=os.environ.get("AWS_DEFAULT_REGION", None),
    runtime=True # Default. Needed for invoke_model() from the data plane
)

from utils.TokenCounterHandler import TokenCounterHandler
token_counter = TokenCounterHandler()

def load_document(loader_class, website_url):
    loader = loader_class([website_url])
    return loader.load()


selenium_loader_doc = load_document(SeleniumURLLoader, website)

# print("selenum = ", selenium_loader_doc)

#Chunking => Text Splitter into smaller tokens
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(selenium_loader_doc)


# - create the Anthropic Model
llm = Bedrock(model_id="anthropic.claude-v2", 
              client=bedrock_client, 
              model_kwargs={
                  'max_tokens_to_sample': 200
              }, 
              callbacks=[token_counter])

# - create the Titan Embeddings Model
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1",
                                       client=bedrock_client)

vectorstore_faiss = FAISS.from_documents(
    docs,
    bedrock_embeddings,
)

wrapper_store_faiss = VectorStoreIndexWrapper(vectorstore=vectorstore_faiss)

query = "What is Emumba?"

#creating an embedding of the query such that it could be compared with the documents
query_embedding = vectorstore_faiss.embedding_function(query)
np.array(query_embedding)


relevant_documents = vectorstore_faiss.similarity_search_by_vector(query_embedding)

answer = wrapper_store_faiss.query(question=query, llm=llm)
print_ww(answer)
