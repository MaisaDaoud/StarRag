
import argparse
import os
from dotenv import load_dotenv, find_dotenv
import openai

from dataclasses import dataclass
from typing import List
from langchain_openai import OpenAI
from langchain.prompts import ChatPromptTemplate
from sentence_transformers import SentenceTransformer
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

_ = load_dotenv(find_dotenv()) 
api_key  = os.environ['OPENAI_API_KEY'] 
base_url = os.environ['OPENAI_BASE_URL']

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


from sentence_transformers import SentenceTransformer, util
from PIL import Image, ImageFile
import requests
import torch

     
    
# class SubmitQueryRequest(BaseModel):
#     query_text:str
  
@dataclass
class QueryResponse:
    user_prompt: str
    retrieved_context:str
    system_response:str


def rag(user_prompt:str, txt_file:str,openai_model:str="Gpt4o"):
    try:
        client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
    except:
         print('error in creating OpenAI client, provide or verify os environment: OPENAIKEY and the BASE_URL')
  
    
    text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

    # Now we encode our text:
    texts = []
    with open(txt_file, 'r') as file:
            # Read each line in the file
            for line in file:
                texts.append(line)
    text_embeddings = text_model.encode(texts)

    query_embed = text_model.encode(user_prompt)
    # Compute cosine similarities:
    cos_sim = util.cos_sim(text_embeddings, query_embed)
    max_sim_idx = torch.argmax(cos_sim)
    context_text = texts[max_sim_idx]

    try:
        chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": [
                    {
                    "type": "text",
                    "text": f"You are a helpful assistant that answers question about Star Wars movie based of the following context \
                    {context_text}. Adopt adopts Yoda’s speaking style"
                    
                    }
                ]
                },
            
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        model=openai_model,
    )
    except:
        print('Error generating response, use Gpt4o model')
        raise
    response = chat_completion.choices[0].message.content
    assert response is not None, 'response was not generated'
    assert response !='', 'response was not generated'

    return QueryResponse(
        user_prompt=user_prompt, retrieved_context=context_text, system_response=response

    )

if __name__== "__main__":

    parser = argparse.ArgumentParser()
    #TODO check for question mark?
    parser.add_argument('--user_prompt', dest='user_prompt',
                        default= 'Who is Luke Skywalker’s father?', type=str, required=True, help='enter user prompt')  
    parser.add_argument('--txt_file', dest='txt_file',
                        default= 'star_wars_corpus.txt', type=str,  help='corpus')                      

    args = parser.parse_args()
    response = rag(args.user_prompt, args.txt_file)
    print(response)