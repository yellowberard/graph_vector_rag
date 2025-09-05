# main.py
import dspy
import logging
from graph_vector_rag.config import settings
from graph_vector_rag.multihop_agent import MultiHopQASystem


def setup_dspy_and_logging():
    """Configures DSPy and Python's logging system."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    if not settings.OPENAI_API_KEY:
        raise ValueError("Please set OPENAI_API_KEY")

    lm = dspy.LM(
        model="openai/gpt-4o",  # Use a more powerful model for reasoning
        api_key=settings.OPENAI_API_KEY,
        temperature=0.1,
        max_tokens=8192,  # Increased for more complex reasoning
    )
    dspy.configure(lm=lm)
    logging.info("✅ DSPy and logging configured.")


def main():
    """Runs an interactive chat loop for the multi-hop QA system."""
    setup_dspy_and_logging()

    # Initialize our multi-hop agent
    agent = MultiHopQASystem(num_hops=3)

    print("\n✅ Multi-Hop Graph RAG Chat Prototype Initialized")
    print("   This agent can answer complex, multi-step questions.")
    print("   Type 'exit' to end the session.")

    try:
        while True:
            question = input("\nAsk a complex question: ")
            if question.lower() == "exit":
                break

            result = agent(question=question)

            print("\n--- Final Answer ---")
            print(result.answer)

            print("\n--- Full Reasoning Scratchpad ---")
            print(result.full_scratchpad)

    except KeyboardInterrupt:
        print("\nSession ended by user.")
    finally:
        agent.close()  # Close the retriever connections
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
