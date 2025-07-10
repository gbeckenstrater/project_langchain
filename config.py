import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Ollama settings
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    
    # Models to use
    EXTRACTION_MODEL = "llama3.2"  # For data extraction
    ANALYSIS_MODEL = "llama3.2"    # For analysis
    
    # File paths
    INPUT_DIR = "data/input"
    OUTPUT_DIR = "data/output"
    
    # Processing settings
    MAX_CHUNK_SIZE = 4000
    CHUNK_OVERLAP = 200