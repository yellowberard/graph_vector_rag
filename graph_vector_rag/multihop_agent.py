import dspy
import logging
from graph_vector_rag.multi_retriever import MultiRAGRetriever

logger = logging.getLogger(__name__)


class GenerateSearchQuery(dspy.Signature):
    """
    You have to think step-by-step to answer complex questions that require
    multi-hop reasoning across various sources of information to provide best possible answer to complex questions.

    Given a complex question and the history of previous search steps, generate
    the search query which would help find the any missing information to get the correct answer.

    Choose ONE of the following query types:
    - 'semantic_search_chunks': A natural language string to find relevant text passages.
    - 'semantic_search_entities': A natural language string to identify core entities.
    - 'graph_cypher_query': A Cypher query to retrieve structured data from the knowledge graphs.

    to generate cypher query first get the semantic entities using semantic_search_entities and then use those entities to form a cypher query.
    Analyze the scratchpad to see what is already known and what is still needed
    to answer the original question.

    Formulate a query to address a specific missing part. If you need to know about a specific entity's connections or you think a Cypher query would be more effective, use a Cypher query. If you need general context or factual information, use a semantic search. All of this text, the scratchpad and the question are related to the embedded engineering systems.
    """

    question: str = dspy.InputField(desc="The original user question.")
    scratchpad: str = dspy.InputField(desc="The history of previous search results.")

    query_type: str = dspy.OutputField(
        desc="One of: 'semantic_search_chunks', 'semantic_search_entities', 'graph_cypher_query'."
    )
    query: str = dspy.OutputField(desc="The search query to execute.")


class SynthesizeFinalAnswer(dspy.Signature):
    """
    After multiple search steps, synthesize a final, comprehensive answer based
    on the original question and the accumulated information in the scratchpad.
    Answer based on the information in the scratchpad. If the scratchpad
    does not contain enough information, state that clearly.
    """

    question: str = dspy.InputField(desc="The original user question.")
    scratchpad: str = dspy.InputField(
        desc="All retrieved context from multiple search hops."
    )
    answer: str = dspy.OutputField(desc="The final, synthesized answer.")


class MultiHopQASystem(dspy.Module):
    """An agent that performs multi-hop reasoning."""

    def __init__(self, num_hops: int = 3):
        super().__init__()
        self.num_hops = num_hops
        self.retriever = MultiRAGRetriever()
        self.generate_query = dspy.ChainOfThought(GenerateSearchQuery)
        self.synthesize_answer = dspy.ChainOfThought(SynthesizeFinalAnswer)

    def forward(self, question: str) -> dspy.Prediction:
        scratchpad = f"User Question: {question}\n\n"
        logger.info(f"Starting multi-hop search with {self.num_hops} hops...")

        for hop in range(self.num_hops):
            logger.info(f"--- Hop {hop + 1}/{self.num_hops} ---")

            query_result = self.generate_query(question=question, scratchpad=scratchpad)

            retrieved_context = self.retriever.retrieve(
                query=query_result.query, query_type=query_result.query_type
            )

            scratchpad += (
                f"\n--- Hop {hop + 1} ---\n"
                f"Search Query (type: {query_result.query_type}): {query_result.query}\n"
                f"Retrieved Context:\n{retrieved_context}\n"
            )
            logger.info("Scratchpad updated for the next hop.")


        logger.info("--- Synthesizing Final Answer ---")
        final_result = self.synthesize_answer(question=question, scratchpad=scratchpad)

        return dspy.Prediction(answer=final_result.answer, full_scratchpad=scratchpad)

    def close(self):
        """Closes the retriever's connections."""
        self.retriever.close()
