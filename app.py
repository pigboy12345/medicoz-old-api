import os
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# Environment variables
HF_API_KEY = os.getenv("HF_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY_JER")
INDEX_NAME = "medicoz-embeddings"

# FastAPI app
app = FastAPI(title="Medical Assistant API")

# Global vars
embedding_function = None
pinecone_index = None

# Prompt template
CUSTOM_PROMPT_TEMPLATE = """<s>[INST] You are a medical assistant specialized in providing accurate information based on medical documents. Use the following context to answer the question:

{context}

---

Question: {question}

Answer based solely on the provided context. If the answer is not clear from the context, state: "I cannot determine the answer based on the available context." [/INST]"""

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
            logger.error(f"Hugging Face API error: {str(e)}")
            return f"Error when calling Hugging Face API: {str(e)}"

# Request model
class QueryRequest(BaseModel):
    question: str

# Startup: load embedding model and Pinecone index
@app.on_event("startup")
def startup_event():
    global embedding_function, pinecone_index
    logger.info("Initializing embedding model and Pinecone index...")

    try:
        embedding_function = HuggingFaceEmbeddings(
            model_name="BAAI/bge-small-en-v1.5",  # Use small model to avoid memory issues
            model_kwargs={"device": "cpu"},
        )
        pc = Pinecone(api_key=PINECONE_API_KEY)
        pinecone_index = pc.Index(INDEX_NAME)

        logger.info("Initialization complete.")
    except Exception as e:
        logger.error(f"Startup initialization failed: {str(e)}")

# Inference endpoint
@app.post("/query")
async def query_rag(request: QueryRequest):
    if not HF_API_KEY or not PINECONE_API_KEY:
        raise HTTPException(status_code=500, detail="Missing API keys")

    try:
        query_text = request.question
        logger.info(f"Received query: {query_text}")

        query_embedding = embedding_function.embed_query(query_text)

        results = pinecone_index.query(
            vector=query_embedding,
            top_k=5,
            include_metadata=True
        )

        context_text = "\n\n---\n\n".join([match["metadata"]["text"] for match in results["matches"]])
        prompt = CUSTOM_PROMPT_TEMPLATE.format(context=context_text, question=query_text)

        model = HuggingFaceLLM()
        response_text = model.invoke(prompt)

        sources = [match["id"] for match in results["matches"]]
        return {"response": response_text, "sources": sources}

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

# Health check endpoint
@app.get("/health")
async def health_check():
    logger.info("Health check requested")
    return {"status": "healthy"}

# Entry point for local dev
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
