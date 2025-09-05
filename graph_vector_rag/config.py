# config.py
import os
from dotenv import load_dotenv

load_dotenv("../.env")


class Settings:
    """Central configuration loaded from environment variables."""

    # OpenAI API Key for DSPy
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

    # Qdrant Database Configurations
    QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_ENTITY_COLLECTION = "opt3001_entities_crew_ai"
    QDRANT_CHUNK_COLLECTION = "data-sheet"

    # --- Neo4j Database Configurations for TWO separate instances ---    
    NEO4J_PDF_KG_URI = "neo4j+s://6140a227.databases.neo4j.io"
    NEO4J_PDF_KG_USER = "neo4j"
    NEO4J_PDF_KG_PASSWORD = "hicibQq67Vnx1B9kugrj5JE68lj-I4Znoh5rTeWto80"

    # Pipeline Behavior
    SEARCH_LIMIT = int(os.getenv("SEARCH_LIMIT", "5"))
    NEO4J_HOP_DEPTH = int(os.getenv("NEO4J_HOP_DEPTH", "2"))


settings = Settings()
