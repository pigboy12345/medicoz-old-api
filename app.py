import os
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings
from huggingface_hub import InferenceClient
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # Import CORS middleware
from pydantic import BaseModel
import logging
# from dotenv import load_dotenv
# load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

HF_API_KEY = os.getenv("HF_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY_JER")
INDEX_NAME = "medicoz-embeddings" # Your new index name

# FastAPI app
app = FastAPI(title="Medical Assistant API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow your Next.js app's origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Embedding function
def get_embedding_function():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},  # Change to "cuda" if GPU is available
    )
    return embeddings

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
                max_new_tokens=512,
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
        embedding_function = get_embedding_function()
        index = initialize_pinecone()
        
        # Generate query embedding
        query_embedding = embedding_function.embed_query(query_text)
        
        # Query Pinecone
        results = index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )
        
        # Extract context
        context_text = "\n\n---\n\n".join([match["metadata"]["text"] for match in results["matches"]])
        prompt = CUSTOM_PROMPT_TEMPLATE.format(context=context_text, question=query_text)
        
        # Call Hugging Face API
        model = HuggingFaceLLM()
        response_text = model.invoke(prompt)
        
        # Extract sources
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
    # logger.info(f"api keys: {PINECONE_API_KEY} and {HF_API_KEY}")
    uvicorn.run(app, host="0.0.0.0", port=8080)