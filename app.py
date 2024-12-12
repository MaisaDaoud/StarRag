from fastapi import FastAPI
from query_rag_transformers import rag, QueryResponse
import uvicorn


app = FastAPI()
   

@app.get("/")
def index():
    return {'Hello':'Datacom'}

@app.post("/submit_query")
def submit_query_endpoint(query:str, file_path:str)->QueryResponse:
    query_response = rag(query, file_path, api=True) 
    return query_response 


if __name__== "__main__":
    port = 7000
    print(f'running fastapi on {port}')
    uvicorn.run("app:app", host='0.0.0.0', port=port)
