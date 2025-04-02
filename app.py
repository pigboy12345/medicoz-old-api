# app.py
import os
import time
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import time as time_module
import requests
from cachetools import TTLCache  # For caching

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables directly
HF_API_KEY = os.getenv("HF_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY_JER")
INDEX_NAME = "medicoz-embeddings"

# Preload embedding model
embedding_function = None
def get_embedding_function():
    global embedding_function
    if embedding_function is None:
        logger.info("Loading embedding model...")
        embedding_function = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",  # We’ll switch to bge-base later
            model_kwargs={"device": "cpu"},
        )
    return embedding_function

# FastAPI app
app = FastAPI(title="Medical Assistant API")

# Pinecone setup
def initialize_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(INDEX_NAME)

# In-memory cache for query responses (TTL = 1 hour)
cache = TTLCache(maxsize=100, ttl=3600)

# Hugging Face Inference Client with Retry Logic and Timeout
class HuggingFaceLLM:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.2", api_key=HF_API_KEY):
        self.model_id = model_id
        self.client = InferenceClient(token=api_key)
    
    def invoke(self, prompt):
        max_retries = 3
        retry_delay = 2  # Seconds
        timeout = 10  # Timeout for each API call (seconds)
        for attempt in range(max_retries):
            try:
                response = self.client.text_generation(
                    prompt,
                    model=self.model_id,
                    max_new_tokens=200,  # Reduced for faster response
                    temperature=0.1,
                    repetition_penalty=1.1,
                    timeout=timeout
                )
                return response
            except requests.exceptions.Timeout:
                logger.warning(f"Attempt {attempt + 1} timed out after {timeout} seconds.")
                if attempt == max_retries - 1:
                    return "Error: Hugging Face API timed out after multiple attempts."
                time_module.sleep(retry_delay)
            except Exception as e:
                if attempt == max_retries - 1:
                    return f"Error when calling Hugging Face API: {str(e)}"
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {retry_delay} seconds...")
                time_module.sleep(retry_delay)

# Prompt template
CUSTOM_PROMPT_TEMPLATE = """<s>[INST] You are a medical assistant specialized in providing accurate information based on medical documents. Use the following context to answer the question:

{context}

---

Question: {question}

Answer based solely on the provided context. If the answer is not clear from the context, state: "I cannot determine the answer based on the available context." [/INST]"""

# Request model
class QueryRequest(BaseModel):
    question: str

# Inference logic
@app.post("/query")
async def query_rag(request: QueryRequest):
    if not HF_API_KEY or not PINECONE_API_KEY:
        raise HTTPException(status_code=500, detail="Missing API keys in environment variables")
    try:
        query_text = request.question
        logger.info(f"Received query: {query_text}")
        
        # Check cache
        if query_text in cache:
            logger.info(f"Cache hit for query: {query_text}")
            return cache[query_text]
        
        start_time = time.time()
        embedding_function = get_embedding_function()
        query_embedding = embedding_function.embed_query(query_text)
        logger.info(f"Embedding generation took {time.time() - start_time:.2f} seconds")
        
        start_time = time.time()
        index = initialize_pinecone()
        results = index.query(
            vector=query_embedding,
            top_k=3,  # Reduced for faster search
            include_metadata=True
        )
        logger.info(f"Pinecone query took {time.time() - start_time:.2f} seconds")
        
        context_text = "\n\n---\n\n".join([match["metadata"]["text"] for match in results["matches"]])
        prompt = CUSTOM_PROMPT_TEMPLATE.format(context=context_text, question=query_text)
        
        start_time = time.time()
        model = HuggingFaceLLM()
        response_text = model.invoke(prompt)
        logger.info(f"Hugging Face API call took {time.time() - start_time:.2f} seconds")
        
        # Fallback if Hugging Face API fails
        if "Error when calling Hugging Face API" in response_text or "timed out" in response_text:
            response_text = f"Unable to generate a response due to API issues. Here is the raw context:\n\n{context_text}"
        
        sources = [match["id"] for match in results["matches"]]
        response = {"response": response_text, "sources": sources}
        
        # Cache the response
        cache[query_text] = response
        logger.info(f"Response generated successfully for query: {query_text}")
        return response
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)