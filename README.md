# StarRag

# How to Run
Create a file call it ```.env``` and add the following variables to it

```
OPENAI_BASE_URL=https://url
OPENAI_API_KEY=sk-key

```

Create conda environment

```
conda create --name rag python=3.10 
```

To run this code use:
```
conda activate rag
python query_rag_transformers.py
```
Or simply run the api on your localhost with 
```
python app.py
```
Note: ```query_rag_transformers.py``` has a method called ```evaluate_response``` that tests the degree of relevance between the response and the documents using llama_index. the call has been commented as it needs a recent API key to run.

### Langchain and InMemoryVectorStore
This repo provides equivelent code but using langchain, semantic search and vectorstore. Please see/run ```query_rag.py``` for reference 
