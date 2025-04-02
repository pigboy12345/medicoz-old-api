import os
import shutil
import time
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY_JER")
DATA_PATH = "data-jer"
INDEX_NAME = "medicoz-embeddings"

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
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1024,  # Match BAAI/bge-large-en-v1.5
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pc.Index(INDEX_NAME)

# Database functions
def reset_database(index):
    try:
        print("✨ Clearing Pinecone Index")
        index.delete(delete_all=True)
        print("✅ Index cleared successfully")
    except Exception as e:
        print(f"Warning: Could not reset index (possibly empty or no namespace): {str(e)}")

def load_documents(): 
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks):
    last_page_id = None
    current_chunk_index = 0
    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id
    return chunks

def add_to_pinecone(chunks, index, embedding_function):
    chunks_with_ids = calculate_chunk_ids(chunks)
    total_chunks = len(chunks_with_ids)
    
    if total_chunks == 0:
        print("✅ No documents to add")
        return
    
    print(f"📊 Total chunks to process: {total_chunks}")
    
    # Process and upload each chunk individually
    start_time = time.time()
    successful_uploads = 0
    failed_uploads = 0
    
    for i, chunk in enumerate(chunks_with_ids):
        chunk_id = chunk.metadata["id"]
        source_file = os.path.basename(chunk.metadata["source"])
        page_num = chunk.metadata["page"]
        
        try:
            # Generate embedding for the current chunk
            embedding = embedding_function.embed_query(chunk.page_content)
            
            # Prepare vector for upsert
            vector = {
                "id": chunk_id,
                "values": embedding,
                "metadata": {
                    "text": chunk.page_content, 
                    "source": chunk.metadata["source"], 
                    "page": chunk.metadata["page"]
                }
            }
            
            # Upload single vector immediately
            index.upsert(vectors=[vector])
            
            successful_uploads += 1
            
            # Calculate and display progress
            progress = (i + 1) / total_chunks * 100
            elapsed_time = time.time() - start_time
            avg_time_per_chunk = elapsed_time / (i + 1)
            est_remaining = avg_time_per_chunk * (total_chunks - i - 1)
            
            print(f"✅ [{i+1}/{total_chunks}] ({progress:.1f}%) Uploaded chunk from {source_file} page {page_num}")
            
            # Every 10 chunks, show time stats
            if (i + 1) % 10 == 0 or i == total_chunks - 1:
                print(f"⏱️  Progress: {successful_uploads} uploaded, {failed_uploads} failed - Est. remaining time: {est_remaining:.1f}s")
                
        except Exception as e:
            failed_uploads += 1
            print(f"❌ Failed to upload chunk {i+1}/{total_chunks} from {source_file} page {page_num}: {str(e)}")
    
    # Final statistics
    total_time = time.time() - start_time
    print(f"\n📈 Upload Statistics:")
    print(f"   ✅ Successfully uploaded: {successful_uploads}/{total_chunks} chunks")
    print(f"   ❌ Failed uploads: {failed_uploads}")
    print(f"   ⏱️  Total processing time: {total_time:.2f} seconds")
    print(f"   🚀 Average processing time: {total_time/total_chunks:.2f} seconds per chunk")
    
    if successful_uploads == total_chunks:
        print("\n🎉 All chunks successfully uploaded to Pinecone!")
    else:
        print(f"\n⚠️  Completed with {failed_uploads} failed uploads")

def populate_database(pdf_paths, reset=False):
    print("\n🚀 Initializing Pinecone connection...")
    index = initialize_pinecone()
    print("✅ Pinecone connection established")
    
    if reset:
        reset_database(index)
    else:
        print("ℹ️ Appending to existing Pinecone index")
    
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
        print(f"📁 Created directory: {DATA_PATH}")
    
    # Process multiple PDF files
    valid_pdf_paths = []
    for pdf_path in pdf_paths:
        if os.path.exists(pdf_path) and pdf_path.lower().endswith('.pdf'):
            valid_pdf_paths.append(pdf_path)
        else:
            print(f"⚠️ Skipping {pdf_path}: File does not exist or is not a PDF")
    
    if not valid_pdf_paths:
        print("❌ No valid PDF files provided.")
        return
    
    # Copy all valid PDFs to the data directory
    copied_files = 0
    skipped_files = 0
    
    print("\n📋 Preparing files:")
    for pdf_path in valid_pdf_paths:
        dest_path = os.path.join(DATA_PATH, os.path.basename(pdf_path))
        if not os.path.exists(dest_path):
            shutil.copy(pdf_path, dest_path)
            copied_files += 1
            print(f"   📄 Copied: {os.path.basename(pdf_path)}")
        else:
            skipped_files += 1
            print(f"   ⏭️  Skipped copy: {os.path.basename(pdf_path)} (already exists)")
    
    print(f"📊 Files processed: {copied_files} copied, {skipped_files} skipped")
    
    # Load, split, and add to Pinecone
    print("\n📚 Loading documents...")
    documents = load_documents()
    print(f"✅ Loaded {len(documents)} document(s)")
    
    print("\n✂️  Splitting documents into chunks...")
    chunks = split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")
    
    print("\n🧠 Initializing embedding model...")
    embedding_function = get_embedding_function()
    print("✅ Embedding model ready")
    
    print("\n🔄 Starting upload process...")
    add_to_pinecone(chunks, index, embedding_function)

if __name__ == "__main__":
    print("=" * 60)
    print("🔍 PINECONE PDF UPLOADER")
    print("=" * 60)
    
    print("\nEnter the paths to your PDF files (separated by commas) or drag-and-drop multiple files:")
    print("Example: C:\\path\\to\\file1.pdf, C:\\path\\to\\file2.pdf")
    
    reset_choice = input("\n🗑️  Do you want to reset the Pinecone index? (y/n, default n): ").lower() == 'y'
    user_input = input("\n📄 PDF paths: ")
    
    pdf_paths = [path.strip() for path in user_input.split(',') if path.strip()]
    
    if pdf_paths:
        populate_database(pdf_paths, reset=reset_choice)
    else:
        print("❌ No PDF paths provided.")