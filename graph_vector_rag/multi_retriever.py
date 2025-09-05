import logging
from qdrant_client import QdrantClient
from neo4j import GraphDatabase, Driver
from neo4j import Query  # Import for type safety
from graph_vector_rag.config import settings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class MultiRAGRetriever:
    """
    This class is an adaptation of your HybridRAGRetriever, modified to support
    a multi-hop reasoning agent. It can execute different query types as directed.
    """

    def __init__(self):
        self.qdrant_client = QdrantClient(url=settings.QDRANT_URL)

        self.model = SentenceTransformer("ibm-granite/granite-embedding-278m-multilingual")
        self.dense_vector_size = 768
        self.pdf_kg_driver = self._create_driver(
            uri=settings.NEO4J_PDF_KG_URI,
            user=settings.NEO4J_PDF_KG_USER,
            password=settings.NEO4J_PDF_KG_PASSWORD,
            graph_name="PDF-Specific KG",
        )

    def _create_driver(
        self, uri: str, user: str, password: str, graph_name: str
    ) -> Driver | None:
        try:
            driver = GraphDatabase.driver(uri, auth=(user, password))
            driver.verify_connectivity()
            logger.info(f"✅ Connected to {graph_name} at {uri}")
            return driver
        except Exception as e:
            logger.error(f"❌ Failed to connect to {graph_name}: {e}")
            return None

    def _get_embedding(self, text: str) -> list[float]:
        try:
            embedding = self.model.encode(text, convert_to_numpy=True).tolist()
            return embedding
        except Exception:
            return [0.0] * self.dense_vector_size

    def retrieve(self, query: str, query_type: str) -> str:
        """Executes a specific retrieval task based on the agent's request."""
        logger.info(f"Executing retrieval of type '{query_type}'...")

        if query_type == "semantic_search_chunks":
            return self._semantic_search_chunks(query)
        elif query_type == "semantic_search_entities":
            return self._semantic_search_entities(query)
        elif query_type == "graph_cypher_query":
            pdf_context = self._cypher_query(self.pdf_kg_driver, query)
            return f"\nPDF-Specific KG Results:\n{pdf_context}"
        else:
            logger.warning(f"Unknown query type received: {query_type}")
            return "An unknown query type was requested."

    def _semantic_search_chunks(self, query_text: str) -> str:
        """Performs a semantic search on the PDF Chunks Qdrant DB."""
        query_vector = self._get_embedding(query_text)
        chunk_hits = self.qdrant_client.search(
            collection_name=settings.QDRANT_CHUNK_COLLECTION,
            query_vector=("dense", query_vector),
            limit=settings.SEARCH_LIMIT,
        )
        evidence = [
            str(hit.payload.get("text_chunk", hit.payload.get("body", ""))) for hit in chunk_hits if hit.payload
        ]      
        return (
            "\n".join(f"- {e}" for e in evidence)
            if evidence
            else "No relevant text passages found."
        )

    def _semantic_search_entities(self, query_text: str) -> str:
        """Performs a semantic search on the Entities Qdrant DB."""
        query_vector = self._get_embedding(query_text)
        entity_hits = self.qdrant_client.search(
            collection_name=settings.QDRANT_ENTITY_COLLECTION,
            query_vector=("dense", query_vector),
            limit=settings.SEARCH_LIMIT,
        )
        entities = [
            str(hit.payload.get("entity_name", ""))
            for hit in entity_hits
            if hit.payload
        ]
        return (
            f"Found potentially relevant entities: {', '.join(entities)}"
            if entities
            else "No matching entities found."
        )

    def _cypher_query(self, driver: Driver | None, cypher_query: str) -> str:
        """Executes a provided Cypher query against a specific Neo4j instance."""
        if driver is None:
            return "Graph connection is not available."

        logger.info(f"Executing Cypher: {cypher_query[:100]}...")
        try:
            with driver.session() as session:
                query = Query(cypher_query)  # type: ignore
                result = session.run(query)
                records = [str(dict(r)) for r in result]
                return "\n".join(records) if records else "Query returned no results."
        except Exception as e:
            return f"Cypher query failed: {e}. Query was: {cypher_query}"

    def close(self):
        """Closes all database connections."""
        if self.pdf_kg_driver:
            self.pdf_kg_driver.close()
            logger.info("PDF-Specific KG (Neo4j Instance 2) connection closed.")
