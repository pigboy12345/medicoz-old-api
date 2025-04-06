import os
import asyncio
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
    allow_origins=["*"],  # Change to specific frontend domain if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load embedding model and Pinecone index at startup
embedding_function = None
pinecone_index = None

@app.on_event("startup")
async def startup_event():
    global embedding_function, pinecone_index
    logger.info("Initializing embedding model and Pinecone index...")
    embedding_function = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",
        model_kwargs={"device": "cpu"},
    )
    pc = Pinecone(api_key=PINECONE_API_KEY)
    pinecone_index = pc.Index(INDEX_NAME)
    logger.info("Initialization complete.")


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
    if not HF_API_KEY or not PINECONE_API_KEY:
        raise HTTPException(status_code=500, detail="Missing API keys in environment variables")

    try:
        query_text = request.question
        logger.info(f"Received query: {query_text}")

        # Generate query embedding
        query_embedding = embedding_function.embed_query(query_text)

        # Query Pinecone
        results = pinecone_index.query(
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


# Run the app if executed directly
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
