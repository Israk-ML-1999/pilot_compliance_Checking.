import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    COHERE_API_KEY = os.getenv("COHERE_API_KEY")
    LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
    
    # Local Storage Paths
    CHROMA_DB_DIR = "./chroma_db_data"
    TEMP_DATA_DIR = "./temp_uploads"

    # Create temp folder if it doesn't exist
    os.makedirs(TEMP_DATA_DIR, exist_ok=True)

settings = Settings()