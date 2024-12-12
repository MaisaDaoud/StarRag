
import argparse
import os
from dotenv import load_dotenv, find_dotenv
import openai

from dataclasses import dataclass
from typing import List
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

import json

_ = load_dotenv(find_dotenv()) 
api_key  = os.environ['OPENAI_API_KEY'] 
base_url = os.environ['OPENAI_BASE_URL']

os.environ['TOKENIZERS_PARALLELISM'] = 'true'



MESSAGE = [
        (
            "system",
            "You are a helpful assistant that answers question about Star Wars movie based of the following context \
                    {context_text}. Only give answer from the context. Adopt  Yoda’s speaking style",
        ),
        ("human", "{user_prompt}"),
    ]

class VectorStore:
    def __init__(self, txt_file:str='star_wars_corpus.txt'):
        self.txt_file = txt_file
        self.vector_store = self.create_embeddings()

    def create_embeddings(self):
        documents = []
        id = 1
        with open(self.txt_file, 'r') as file:
            # Read each line in the file
            for line in file:
                document = Document(id=id, page_content=line)
                documents.append(document)
                id+=1
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Equivalent to SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        vector_store = InMemoryVectorStore(embeddings)
        vector_store.add_documents(documents=documents)
        
        print("✅ Vector data store ceated")
        return vector_store

        
    
# class SubmitQueryRequest(BaseModel):
#     query_text:str
  
@dataclass
class QueryResponse:
    user_prompt: str
    retrieved_context:str
    system_response:str


def rag(user_prompt:str, vector_store):
    #     client = openai.OpenAI(
    #     api_key=api_key,
    #     base_url=base_url,
    # )
    # define open client
    llm = ChatOpenAI(
    model= "Gpt4o", # "gpt-3.5-turbo-instruct",
    temperature=0,
    max_retries=2,
    api_key=api_key,
    base_url=base_url,
    )

    # Searching for similar texts
    results = vector_store.similarity_search(query=user_prompt, k=1)
    for doc in results:
        print(f"* {doc.page_content} [{doc.metadata}]")

    context_text = results[0].page_content

    # define template
    prompt_template = ChatPromptTemplate.from_messages(MESSAGE)
    prompt = prompt_template.format(context_text=context_text, user_prompt=user_prompt)

    response = llm.invoke(prompt)
    response_text = response.content

    # return QueryResponse(
    #     user_prompt=user_prompt, retrieved_context=context_text, system_response=response_text
    # )
    return_val = {
        'user_prompt':user_prompt, 
        'retrieved_context':context_text, 
        'system_response':response_text
    }
    return json.dumps(return_val)

if __name__== "__main__":

    parser = argparse.ArgumentParser()
    #TODO check for question mark?
    parser.add_argument('--user_prompt', dest='user_prompt',
                        default= 'Who is Luke Skywalker’s father?', type=str, required=True, help='enter user prompt')  
    parser.add_argument('--txt_file', dest='txt_file',
                        default= 'data/star_wars_corpus.txt', type=str,  help='corpus')                      

    args = parser.parse_args()
    vector_store_obj = VectorStore(args.txt_file)
    response = rag(args.user_prompt, vector_store_obj.vector_store)
    print(response)