"""Stored output chunks demo.

Shows store_output_chunks=True: long assistant replies are chunked by paragraph
and stored; only chunks relevant to the current query are retrieved and added
to context. Keeps context lean when prior answers were long.

Run:
    python examples/11_context/context_output_chunks_demo.py
"""

from __future__ import annotations

from syrin import Agent, AgentConfig, Context
from syrin.memory import Memory
from syrin.model import Model

_model = Model.Almock()


def _main() -> None:
    print("=== Stored output chunks (long answers -> relevant chunks only) ===\n")

    agent = Agent(
        model=_model,
        system_prompt=(
            "You are helpful. When explaining, use multiple paragraphs separated by blank lines. "
            "Keep each paragraph focused."
        ),
        memory=Memory(),
        config=AgentConfig(
            context=Context(
                max_tokens=8000,
                store_output_chunks=True,
                output_chunk_top_k=5,
                output_chunk_threshold=0.0,
            )
        ),
    )

    # First turn: long answer about Syrin memory (multiple paragraphs)
    agent.response("Explain Syrin's memory system in 3 short paragraphs.")

    # Second turn: ask about relevance -- only relevant paragraphs should be in context
    result = agent.response("What about the relevance threshold?")
    snap = agent.context.snapshot()
    print(f"Response: {result.content}")
    print(f"output_chunks count: {len(snap.output_chunks)}")
    print(f"output_chunk_scores: {snap.output_chunk_scores}")
    for i, oc in enumerate(snap.output_chunks):
        content = oc.get("content", "")
        snippet = content
        print(f"  [{i}] score={oc.get('score', 0):.2f} content={snippet}")


if __name__ == "__main__":
    _main()
