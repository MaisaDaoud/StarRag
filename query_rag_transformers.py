
import argparse
import os
from dotenv import load_dotenv, find_dotenv
import re
import openai

from dataclasses import dataclass
from typing import List
from sentence_transformers import SentenceTransformer

from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.evaluation import FaithfulnessEvaluator
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

import json
from sentence_transformers import SentenceTransformer, util

import torch
from pydantic import BaseModel


_ = load_dotenv(find_dotenv()) 
api_key  = os.environ['OPENAI_API_KEY'] 
base_url = os.environ['OPENAI_BASE_URL']

os.environ['TOKENIZERS_PARALLELISM'] = 'true'
   
     
class QueryResponse(BaseModel):
    user_prompt: str
    retrieved_context:str
    system_response:str

def evaluate_response(user_prompt, response):
    # evaluates if the answer is faithful to the retrieved contexts
    llm = OpenAI(
    model=  "gpt-3.5-turbo-instruct",
    temperature=0.0,
    max_retries=2,
    api_key=api_key,
    base_url=base_url,
    )

    evaluator = FaithfulnessEvaluator(llm=llm)
    documents = SimpleDirectoryReader('data').load_data()

    # Create an index from the documents
    index = VectorStoreIndex.from_documents(documents)
    vector_index = index.as_query_engine()
    # query index
    query_engine = vector_index.as_query_engine()
    response = query_engine.query(
      user_prompt
    )
    eval_result = evaluator.evaluate_response(response=response)
    print(f'[***]  faithful evaluation', str(eval_result.passing))

def generate_embeddings(text_model, text):
    return text_model.encode(text)

def rag(user_prompt:str, txt_file:str,openai_model:str="Gpt4o",api:bool=False):
 
    # create the client
    client = openai.OpenAI(
        api_key=api_key,
        base_url=base_url,
    )
    
    text_model = SentenceTransformer('sentence-transformers/clip-ViT-B-32-multilingual-v1')

    # encode our text:
    texts = []
    try:
        with open(txt_file, 'r') as file:
                # Read each line in the file
                for line in file:
                    texts.append(line)
    except:
        raise Exception(f"File not found!") from None
    
    # generate text and query embeddings
    text_embeddings = generate_embeddings(text_model,texts)

    query_embed = generate_embeddings(text_model,user_prompt)

    # compute cosine similarities:
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
                    {context_text}. Only give answer from the context. Adopt  Yoda’s speaking style"
                    
                    }
                ]
                },
            
            {
                "role": "user",
                "content": user_prompt,
            }
        ],
        model=openai_model,
        temperature=0.0 # to strictly return from context
    )   
        
        response = chat_completion.choices[0].message.content
        
        assert len(response)>1,  "OpenAI response is empty or invalid."

    except AssertionError as e:
        raise Exception(f"AssertionError!") from None
    except openai.OpenAIError as e: 
        raise Exception(f"OpenAI API error, check for the key and the base_url") from None
    except Exception as e:
        print(f"Unexpected error: {e}")

    ''' 
    # other detailed open ai exceptions
    # except openai.AuthenticationError as e:
    #     # Handle authentication-related issues (e.g., invalid API key)
    #     raise Exception(f"Authentication failed!") from None
    # except openai.APIConnectionError as e:
    #     # Handle issues connecting to the API (e.g., network problems)
    #     raise Exception(f"Failed to connect to the OpenAI API") from None #{e}
    # except openai.InvalidRequestError as e:
    #     # Handle invalid requests (e.g., malformed API requests)
    #     raise Exception(f"Invalid request to OpenAI API") from None
    # except openai.OpenAIError as e:
    #     # Handle generic OpenAI errors
    #     raise Exception(f"An error occurred with the OpenAI API") from None
    # except Exception as e:
    #     # Handle unexpected errors
    #     raise Exception(f"An unexpected error occurred: {e}, Check for model=Gpt4o")
    '''

    # evaluate the response: this method doesnt work as the API key is old. should work with new one
    #evaluate_response(user_prompt, response)
    result = QueryResponse(
        user_prompt=user_prompt, retrieved_context=context_text, system_response=response)
    if api:
        return result
    # return json.dumps(result.model_json_schema())
    return_val = {
        'user_prompt':user_prompt, 
        'retrieved_context':context_text, 
        'system_response':response
    }
    return json.dumps(return_val)

if __name__== "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--user_prompt', dest='user_prompt',
                        default= 'Who is Luke Skywalker’s father?', type=str, required=True, help='enter user prompt')  
    parser.add_argument('--txt_file', dest='txt_file',
                        default= 'data/star_wars_corpus.txt', type=str, required=True, help='corpus')                      

    args = parser.parse_args()
    response = rag(args.user_prompt, args.txt_file)
    print(response)