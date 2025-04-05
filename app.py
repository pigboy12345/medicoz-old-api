import os
import asyncio
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import logging
from fastapi.middleware.cors import CORSMiddleware
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables directly
HF_API_KEY = os.getenv("HF_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY_JER")
INDEX_NAME = "medicoz-embeddings"

# Global variables for performance optimization
embeddings = None
pc_index = None

# FastAPI app
app = FastAPI(title="Medical Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "*", "https://3000-idx-medicozgit-1743525220848.cluster-bec2e4635ng44w7ed22sa22hes.cloudworkstations.dev/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Embedding function - initialize once
def get_embedding_function():
    global embeddings
    if embeddings is None:
        embeddings = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": "cpu"},
        )
    return embeddings

# Pinecone setup - initialize once
def initialize_pinecone():
    global pc_index
    if pc_index is None:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pc_index = pc.Index(INDEX_NAME)
    return pc_index

# Hugging Face Inference Client with tight timeout
class HuggingFaceLLM:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.2", api_key=HF_API_KEY):
        self.model_id = model_id
        self.client = InferenceClient(token=api_key)
    
    async def invoke_async(self, prompt, timeout=10):
        try:
            # Use a short timeout to prevent hanging
            response = await asyncio.wait_for(
                self.async_text_generation(prompt),
                timeout=timeout
            )
            return response
        except asyncio.TimeoutError:
            logger.warning(f"Timeout calling model {self.model_id}")
            return None
        except Exception as e:
            logger.error(f"Error calling model {self.model_id}: {str(e)}")
            return None
    
    async def async_text_generation(self, prompt):
        # Convert synchronous call to async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, 
            lambda: self.client.text_generation(
                prompt,
                model=self.model_id,
                max_new_tokens=256,  # Reduced token count
                temperature=0.1,
                repetition_penalty=1.1,
            )
        )

# Simple prompt template
CUSTOM_PROMPT_TEMPLATE = """<s>[INST] Medical assistant: Answer based on context:
{context}
Question: {question} [/INST]"""

# Simpler fallback prompt
FALLBACK_PROMPT = """<s>[INST] Answer this medical question briefly: {question} [/INST]"""

# Request model
class QueryRequest(BaseModel):
    question: str

# Response cache to improve performance
response_cache = {}

# Health check endpoint - respond quickly
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Inference logic
@app.post("/query")
async def query_rag(request: QueryRequest, background_tasks: BackgroundTasks):
    # Quick validation
    if not request.question:
        return JSONResponse(
            status_code=400,
            content={"response": "Please provide a question", "sources": []}
        )
    
    # Check cache first
    cache_key = request.question.strip().lower()
    if cache_key in response_cache:
        logger.info(f"Cache hit for query: {cache_key[:30]}...")
        return response_cache[cache_key]
    
    # Start processing but return quickly
    background_tasks.add_task(process_query, request.question)
    
    # Return immediate response to prevent timeout
    return JSONResponse(
        content={
            "response": "Your medical question is being processed. Please check back in a few seconds.",
            "processing": True,
            "sources": []
        }
    )

# Status endpoint to check if query is ready
@app.get("/query_status/{query_text}")
async def query_status(query_text: str):
    cache_key = query_text.strip().lower()
    if cache_key in response_cache:
        return response_cache[cache_key]
    else:
        return {"processing": True, "response": "Still processing your request..."}

# Background processing function
async def process_query(query_text: str):
    cache_key = query_text.strip().lower()
    
    try:
        # Initialize resources
        embedding_function = get_embedding_function()
        index = initialize_pinecone()
        
        # Generate query embedding
        query_embedding = embedding_function.embed_query(query_text)
        
        # Query Pinecone with short timeout
        try:
            results = index.query(
                vector=query_embedding,
                top_k=3,  # Reduced for speed
                include_metadata=True
            )
        except Exception as e:
            logger.error(f"Pinecone query error: {str(e)}")
            results = {"matches": []}
        
        # Extract context safely
        context_text = ""
        sources = []
        try:
            if results and "matches" in results and results["matches"]:
                context_chunks = []
                for match in results["matches"]:
                    if "metadata" in match and "text" in match["metadata"]:
                        context_chunks.append(match["metadata"]["text"])
                    if "id" in match:
                        sources.append(match["id"])
                
                context_text = "\n---\n".join(context_chunks)
            
            if not context_text:
                context_text = "No relevant context found."
        except Exception as e:
            logger.error(f"Context extraction error: {str(e)}")
            context_text = "Context extraction error."
        
        # Prepare prompt
        prompt = CUSTOM_PROMPT_TEMPLATE.format(
            context=context_text[:4000],  # Limit context size
            question=query_text
        )
        
        # Try primary model with short timeout
        model = HuggingFaceLLM(model_id="google/flan-t5-base")  # Using smaller model first
        response_text = await model.invoke_async(prompt, timeout=10)
        
        # Quick fallback if primary fails
        if not response_text or len(str(response_text).strip()) < 10:
            fallback_prompt = FALLBACK_PROMPT.format(question=query_text)
            model = HuggingFaceLLM(model_id="google/flan-t5-small")  # Even smaller model
            response_text = await model.invoke_async(fallback_prompt, timeout=5)
            
            if not response_text:
                response_text = "Unable to process your medical question at this time."
        
        # Store in cache
        response_cache[cache_key] = {
            "response": str(response_text),
            "sources": sources,
            "processing": False
        }
        
        return response_text
    except Exception as e:
        logger.error(f"Background processing error: {str(e)}")
        response_cache[cache_key] = {
            "response": "An error occurred while processing your request.",
            "sources": [],
            "processing": False
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)