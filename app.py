import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from pinecone import Pinecone
from langchain_huggingface import HuggingFaceEmbeddings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# Load environment variables
HF_API_KEY = os.getenv("HF_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY_JER")
INDEX_NAME = "medicoz-embeddings"

# FastAPI app
app = FastAPI(title="Medical Assistant API")

# Global variables
embedding_function = None
pinecone_index = None

# Hugging Face Inference Wrapper
class HuggingFaceLLM:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.2", api_key=HF_API_KEY):
        self.model_id = model_id
        self.client = InferenceClient(token=api_key)

    def invoke(self, prompt: str) -> str:
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
            logger.error(f"HuggingFace API error: {e}")
            return f"Error calling Hugging Face API: {str(e)}"

# Prompt template
CUSTOM_PROMPT_TEMPLATE = """<s>[INST] You are a medical assistant specialized in providing accurate information based on medical documents. Use the following context to answer the question:

{context}

---

Question: {question}

Answer based solely on the provided context. If the answer is not clear from the context, state: "I cannot determine the answer based on the available context." [/INST]"""

# Request schema
class QueryRequest(BaseModel):
    question: str

# Startup initialization
@app.on_event("startup")
def startup_event():
    global embedding_function, pinecone_index

    logger.info("Initializing embedding model and Pinecone index...")

    if not PINECONE_API_KEY or not HF_API_KEY:
        logger.error("Missing required API keys")
        raise RuntimeError("Missing required environment variables (HF_API_KEY or PINECONE_API_KEY_JER)")

    try:
        embedding_function = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"},  # Set to 'cuda' if GPU is available
        )

        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(INDEX_NAME)

        if embedding_function is None or pinecone_index is None:
            raise RuntimeError("Embedding model or Pinecone index failed to initialize")

        logger.info("✅ Initialization complete.")
    except Exception as e:
        logger.exception("Startup failed")
        raise RuntimeError(f"Startup error: {e}")

# Query endpoint
@app.post("/query")
async def query_rag(request: QueryRequest):
    if embedding_function is None or pinecone_index is None:
        raise HTTPException(status_code=500, detail="Server not ready. Try again shortly.")

    try:
        query_text = request.question
        logger.info(f"Received query: {query_text}")

        # Embed query
        query_embedding = embedding_function.embed_query(query_text)

        # Query Pinecone
        results = pinecone_index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True,
        )

        # Extract context from matched documents
        context_text = "\n\n---\n\n".join([match["metadata"]["text"] for match in results["matches"]])
        prompt = CUSTOM_PROMPT_TEMPLATE.format(context=context_text, question=query_text)

        # Get response from HuggingFace model
        model = HuggingFaceLLM()
        response_text = model.invoke(prompt)

        # Return result
        sources = [match["id"] for match in results["matches"]]
        return {"response": response_text, "sources": sources}
    except Exception as e:
        logger.exception("Query processing error")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Readiness check
@app.get("/ready")
async def readiness_check():
    if embedding_function and pinecone_index:
        return {"status": "ready"}
    else:
        return {"status": "initializing"}

# Run locally
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
