# config.py
import os
from dotenv import load_dotenv

load_dotenv("../.env")


class Settings:
    """Central configuration loaded from environment variables."""

    # OpenAI API Key for DSPy
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # Qdrant Database Configurations
    QDRANT_URL = os.getenv("QDRANT_URL", "")
    QDRANT_ENTITY_COLLECTION = os.getenv("QDRANT_ENTITY_COLLECTION", "")
    QDRANT_CHUNK_COLLECTION = os.getenv("QDRANT_CHUNK_COLLECTION", "")

    # --- Neo4j Database Configurations for TWO separate instances ---    
    NEO4J_PDF_KG_URI = os.getenv("NEO4J_PDF_KG_URI","")
    NEO4J_PDF_KG_USER = os.getenv("NEO4J_PDF_USERNAME","")
    NEO4J_PDF_KG_PASSWORD = os.getenv("NEO4J_PDF_KG_PASSWORD","")

    # Pipeline Behavior
    SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT", "5"))
    NEO4J_HOP_DEPTH = int(os.getenv("NEO4J_HOP_DEPTH", "2"))


settings = Settings()
