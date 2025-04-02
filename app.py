# app.py
import os
import time
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables directly
HF_API_KEY = os.getenv("HF_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY_JER")
INDEX_NAME = "medicoz-embeddings"  # Matches your new index

# Preload embedding model
embedding_function = None
def get_embedding_function():
    global embedding_function
    if embedding_function is None:
        logger.info("Loading embedding model...")
        embedding_function = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",  # 1024 dimensions to match the index
            model_kwargs={"device": "cpu"},
        )
    return embedding_function

# FastAPI app
app = FastAPI(title="Medical Assistant API")

# Pinecone setup
def initialize_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(INDEX_NAME)

# Hugging Face Inference Client
class HuggingFaceLLM:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.2", api_key=HF_API_KEY):
        self.model_id = model_id
        self.client = InferenceClient(token=api_key)
    
    def invoke(self, prompt):
        try:
            response = self.client.text_generation(
                prompt,
                model=self.model_id,
                max_new_tokens=200,
                temperature=0.1,
                repetition_penalty=1.1,
            )
            return response
        except Exception as e:
            return f"Error when calling Hugging Face API: {str(e)}"

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
        
        # Generate embedding for the query
        start_time = time.time()
        embedding_function = get_embedding_function()
        query_embedding = embedding_function.embed_query(query_text)
        logger.info(f"Embedding generation took {time.time() - start_time:.2f} seconds")
        
        # Query Pinecone with the embedding vector
        start_time = time.time()
        index = initialize_pinecone()
        results = index.query(
            vector=query_embedding,  # Pass the embedding vector
            top_k=3,
            include_metadata=True
        )
        logger.info(f"Pinecone query took {time.time() - start_time:.2f} seconds")
        
        context_text = "\n\n---\n\n".join([match["metadata"]["text"] for match in results["matches"]])
        prompt = CUSTOM_PROMPT_TEMPLATE.format(context=context_text, question=query_text)
        
        start_time = time.time()
        model = HuggingFaceLLM()
        response_text = model.invoke(prompt)
        logger.info(f"Hugging Face API call took {time.time() - start_time:.2f} seconds")
        
        sources = [match["id"] for match in results["matches"]]
        logger.info(f"Response generated successfully for query: {query_text}")
        return {"response": response_text, "sources": sources}
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
    port = int(os.getenv("PORT", 8000))  # Use Railway's PORT if available, else 8000
    uvicorn.run(app, host="0.0.0.0", port=port)