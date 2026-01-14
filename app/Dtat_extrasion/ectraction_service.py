import os
import shutil
from llama_parse import LlamaParse
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_chroma import Chroma
from langchain_cohere import CohereEmbeddings
from config import settings

# Initialize Cohere (Best for Compliance RAG)
embeddings = CohereEmbeddings(
    model="embed-english-v3.0",
    cohere_api_key=settings.COHERE_API_KEY
)

async def process_and_embed_rules(file_path: str):
    """
    1. Delete old DB (fresh start).
    2. Parse PDF with LlamaParse -> Markdown.
    3. Split text specifically by # (Heading 1).
    4. Embed & Save to Local ChromaDB.
    """
    
    # --- 1. Clean up old Database ---
    if os.path.exists(settings.CHROMA_DB_DIR):
        try:
            shutil.rmtree(settings.CHROMA_DB_DIR)
            print("--- Old Database Deleted ---")
        except Exception as e:
            print(f"Warning: Could not delete old DB: {e}")

    # --- 2. LlamaParse (Extract Markdown) ---
    print("--- Starting LlamaParse ---")
    parser = LlamaParse(
        api_key=settings.LLAMA_CLOUD_API_KEY,
        result_type="markdown", # Crucial for header detection
        verbose=True
    )
    documents = await parser.aload_data(file_path)
    full_markdown_text = "\n\n".join([doc.text for doc in documents])

    # --- 3. Chunking (Rule: Split by # Heading 1) ---
    # We map '#' to the metadata key 'Section'
    headers_to_split_on = [
        ("#", "Section"), 
    ]
    
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False # Keep the header text in the chunk for context
    )
    
    splits = splitter.split_text(full_markdown_text)
    print(f"--- Created {len(splits)} Section Chunks ---")

    # --- 4. Embed & Persist ---
    # This automatically creates the folder and saves data there
    Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=settings.CHROMA_DB_DIR,
        collection_name="pilot_rules"
    )
    
    return len(splits)