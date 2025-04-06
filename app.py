import os
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
HF_API_KEY = os.getenv("HF_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY_JER")
INDEX_NAME = "medicoz-embeddings"

# FastAPI app
app = FastAPI(title="Medical Assistant API")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global variables
embedding_function = None
pinecone_index = None
model = None

@app.on_event("startup")
async def startup_event():
    global embedding_function, pinecone_index, model
    try:
        logger.info("Initializing embedding model...")
        embedding_function = HuggingFaceEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            model_kwargs={"device": "cpu"},
        )
        
        logger.info("Initializing Pinecone index...")
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(INDEX_NAME)
        
        logger.info("Initializing HuggingFace LLM...")
        model = HuggingFaceLLM()
        
        logger.info("All services initialized successfully.")
    except Exception as e:
        logger.error(f"Error during initialization: {str(e)}")
        # Don't raise exception here to allow app to start even if initialization fails


# Hugging Face LLM Wrapper
class HuggingFaceLLM:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.2", api_key=HF_API_KEY):
        self.model_id = model_id
        self.client = InferenceClient(token=api_key)
    
    def invoke(self, prompt):
        try:
            response = self.client.text_generation(
                prompt,
                model=self.model_id,
                max_new_tokens=512,
                temperature=0.1,
                repetition_penalty=1.1,
            )
            return response
        except Exception as e:
            logger.error(f"Error calling Hugging Face API: {str(e)}")
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


# Query endpoint
@app.post("/query")
async def query_rag(request: QueryRequest):
    global embedding_function, pinecone_index, model
    
    # Verify initialization status
    if not embedding_function or not pinecone_index or not model:
        logger.error("Services not properly initialized")
        raise HTTPException(status_code=500, detail="Services not properly initialized. Please try again later.")
        
    if not HF_API_KEY or not PINECONE_API_KEY:
        logger.error("Missing API keys in environment variables")
        raise HTTPException(status_code=500, detail="Missing API keys in environment variables")

    try:
        query_text = request.question
        logger.info(f"Processing query: {query_text}")

        # Generate query embedding
        query_embedding = embedding_function.embed_query(query_text)
        logger.info("Query embedding generated")

        # Query Pinecone
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        logger.info(f"Retrieved {len(results['matches'])} results from Pinecone")

        # Extract context
        context_text = "\n\n---\n\n".join([match["metadata"]["text"] for match in results["matches"]])
        prompt = CUSTOM_PROMPT_TEMPLATE.format(context=context_text, question=query_text)

        # Call Hugging Face API
        response_text = model.invoke(prompt)
        logger.info("Response generated from Hugging Face API")

        # Extract sources
        sources = [match["id"] for match in results["matches"]]

        logger.info(f"Successfully processed query: {query_text}")
        return {"response": response_text, "sources": sources}

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


# Health check endpoint
@app.get("/health")
async def health_check():
    global embedding_function, pinecone_index, model
    status = {
        "status": "healthy",
        "components": {
            "embedding_model": "initialized" if embedding_function else "not initialized",
            "pinecone_index": "initialized" if pinecone_index else "not initialized",
            "huggingface_model": "initialized" if model else "not initialized"
        }
    }
    logger.info(f"Health check: {status}")
    return status


# Run the app with correct port for Railway
if __name__ == "__main__":
    import uvicorn
    # Railway sets PORT environment variable automatically
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)