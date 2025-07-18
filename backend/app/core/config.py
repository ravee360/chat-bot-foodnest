# backend/app/core/config.py

print("--- Starting config.py execution ---") # DEBUGGING

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional, Dict, Any
from dotenv import load_dotenv
load_dotenv()

print(f"Current working directory: {os.getcwd()}") # DEBUGGING
print(f"Script __file__: {__file__}") # DEBUGGING

# Determine project root directory for correct data paths
# __file__ is .../doc_theme_bot/backend/app/core/config.py
# .parent is .../doc_theme_bot/backend/app/core/
# .parents[0] is .../doc_theme_bot/backend/app/core/
# .parents[1] is .../doc_theme_bot/backend/app/
# .parents[2] is .../doc_theme_bot/backend/
# .parents[3] is .../doc_theme_bot/
PROJECT_ROOT_DIR = Path(__file__).resolve().parents[3]
ENV_FILE_PATH = PROJECT_ROOT_DIR / "backend" / ".env" # .env is in backend/

print(f"Calculated PROJECT_ROOT_DIR: {PROJECT_ROOT_DIR}") # DEBUGGING
print(f"Calculated ENV_FILE_PATH: {ENV_FILE_PATH}") # DEBUGGING
print(f"Does .env file exist at calculated path? {ENV_FILE_PATH.exists()}") # DEBUGGING


class Settings(BaseSettings):
    """
    Application settings are loaded from environment variables and/or an .env file.
    Focus is on using OpenRouter as the primary LLM provider.
    """
    print("--- Inside Settings class definition ---") # DEBUGGING
    # --- Primary API Key ---
    OPENROUTER_API_KEY: Optional[str] = os.environ.get("OPENROUTER_API_KEY")

    # --- LLM Configuration ---
    DEFAULT_LLM_PROVIDER: str = "openrouter"
    OPENROUTER_API_BASE: str = "https://openrouter.ai/api/v1"
    DEFAULT_LLM_MODEL: str = "meta-llama/llama-3.3-8b-instruct:free"

    # --- Embedding Model Configuration ---
    EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2" # This is a SentenceTransformer model

    # --- Vector Store (ChromaDB) Configuration ---
    CHROMA_DB_PATH: str = str(PROJECT_ROOT_DIR / "data" / "db")
    CHROMA_COLLECTION_NAME: str = "doc_embeddings_v1"

    # --- Document Processing ---
    UPLOAD_DIR: str = str(PROJECT_ROOT_DIR / "data" / "uploads")

    # --- API Configuration ---
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Document Theme Bot"

    # --- Project Root Directory ---
    PROJECT_ROOT_DIR: str = str(PROJECT_ROOT_DIR)

    model_config = SettingsConfigDict(
        env_file=ENV_FILE_PATH,
        env_file_encoding='utf-8',
        extra='ignore'
    )
    print("--- Finished Settings class definition ---") # DEBUGGING

print("--- About to instantiate Settings() ---") # DEBUGGING
try:
    settings = Settings()
    print("--- Successfully instantiated Settings() ---") # DEBUGGING
except Exception as e:
    print(f"!!! ERROR during Settings() instantiation: {e} !!!") 
    raise

if __name__ == "__main__":
    print("--- Inside if __name__ == '__main__' block of config.py ---") 
    print("Current Settings Loaded (OpenRouter Focus):")
    if 'settings' in locals() or 'settings' in globals():
        print(f"  OpenRouter API Key Loaded: {'Yes' if settings.OPENROUTER_API_KEY else 'No'}")
        print(f"  Default LLM Model (for OpenRouter): {settings.DEFAULT_LLM_MODEL}")
        print(f"  Embedding Model Name: {settings.EMBEDDING_MODEL_NAME}")
        print(f"  ChromaDB Path: {settings.CHROMA_DB_PATH}")
        print(f"  Upload Directory: {settings.UPLOAD_DIR}")
        print(f"  .env file path used: {ENV_FILE_PATH}")

        try:
            Path(settings.CHROMA_DB_PATH).mkdir(parents=True, exist_ok=True)
            Path(settings.UPLOAD_DIR).mkdir(parents=True, exist_ok=True)
            print(f"  Ensured ChromaDB directory exists: {Path(settings.CHROMA_DB_PATH).exists()}")
            print(f"  Ensured Upload directory exists: {Path(settings.UPLOAD_DIR).exists()}")
        except Exception as e:
            print(f"!!! ERROR creating directories in config.py __main__: {e} !!!") 
    else:
        print("!!! 'settings' object not found in config.py __main__. Instantiation likely failed. !!!")

else:
    # This block will execute when config.py is imported by another module (like vstore_svc.py)
    print(f"--- Script 'config.py' is imported, not run directly. __name__ is: {__name__} ---")
    print(f"--- Settings instantiated in 'config.py' global scope. ChromaDB path: {settings.CHROMA_DB_PATH if 'settings' in globals() else 'settings not yet defined globally'} ---")

print("--- End of config.py execution ---") # DEBUGGING
