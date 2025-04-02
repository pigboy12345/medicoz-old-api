# chat.py
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
from langchain.prompts import ChatPromptTemplate
from huggingface_hub import InferenceClient
# from dotenv import load_dotenv
import os

# load_dotenv()

HF_API_KEY = os.getenv("HF_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY_JER")
INDEX_NAME = "medicoz-embeddings" # Your new index name

CUSTOM_PROMPT_TEMPLATE = """<s>[INST] You are a medical assistant specialized in providing accurate information based on medical documents. Use the following context to answer the question:

{context}

---

Question: {question}

Answer based solely on the provided context. If the answer is not clear from the context, state: "I cannot determine the answer based on the available context." [/INST]
"""

# Embedding function using Hugging Face
def get_embedding_function():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-large-en-v1.5",  # Outputs 1024 dimensions
        model_kwargs={"device": "cpu"},  # Use "cuda" if you have a GPU
    )
    return embeddings

# Pinecone setup
def initialize_pinecone():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    return pc.Index(INDEX_NAME)

# Hugging Face API class
class HuggingFaceLLM:
    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.2", api_key=None):
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

def query_rag(query_text, model_id="mistralai/Mistral-7B-Instruct-v0.2", api_key=None):
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
    
    # Extract context from results
    context_text = "\n\n---\n\n".join([match["metadata"]["text"] for match in results["matches"]])
    prompt_template = ChatPromptTemplate.from_template(CUSTOM_PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    
    # Call Hugging Face API
    model = HuggingFaceLLM(model_id=model_id, api_key=api_key)
    response_text = model.invoke(prompt)
    
    # Extract sources
    sources = [match["id"] for match in results["matches"]]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)
    return response_text

if __name__ == "__main__":
    model_id = input("Enter model ID (default: mistralai/Mistral-7B-Instruct-v0.2): ") or "mistralai/Mistral-7B-Instruct-v0.2"
    while True:
        query = input("\nYour question (type 'exit' to quit): ")
        if query.lower() in ['exit', 'quit']:
            break
        query_rag(query, model_id=model_id, api_key=HF_API_KEY)