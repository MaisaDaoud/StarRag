
import argparse
import os
from dotenv import load_dotenv, find_dotenv
import openai
from pydantic import BaseModel, Field

_ = load_dotenv(find_dotenv()) 
openai.api_key  = os.environ['OPENAI_API_KEY'] 
openai.base_url = os.environ['OPENAI_BASE_URL']

PROMPT_TEMPLATE = """ <s>[INST] <<SYS>>
Your are Senior Rep a senior medical represintative aasisting   junior medical rep in introducing medicin to doctors based on the provided {context} which shows what  a doctor is interested to hear. Reply with short answers. Do not use following characters in response: '{{' '}}' '(' ')' '[' ']' ':'.\n"
<</SYS>>
---

Answer the question based on the above context: {question}  [/INST]
"""


# class SubmitQueryRequest(BaseModel):
#     query_text:str
  
@dataclass
class QueryResponse:
    query_text: str
    response_text: str
    sources: List[str]

def rag(user_prompt:str):

    return response

if __name__== "__main__":

    parser = argparse.ArgumentParser()
    #TODO check for question mark?
    parser.add_argument('--user_prompt', dest='user_prompt',
                        default= 'Who is Luke Skywalker’s father?', type=str, required=True, help='enter user prompt')                        

    args = parser.parse_args()
    response = rag(args.user_prompt)