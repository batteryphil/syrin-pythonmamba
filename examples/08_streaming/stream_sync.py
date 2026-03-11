"""Synchronous Streaming -- Token-by-token output with stream().

Demonstrates:
- agent.stream(input) for synchronous streaming
- Collecting chunks into full text
- StreamChunk properties (index, text, is_final)

Run: python examples/08_streaming/stream_sync.py
"""

from syrin import Agent, Model

# --- Define a streaming agent ---


class StreamAgent(Agent):
    _agent_name = "stream-agent"
    _agent_description = "Streams token-by-token output"
    model = Model.Almock()
    system_prompt = "You are a helpful assistant."


# --- Stream and collect output ---

if __name__ == "__main__":
    agent = StreamAgent()

    print("Streaming response:")
    print("-" * 40)

    chunks = list(agent.stream("Tell me a short story"))
    full_text = "".join(c.text for c in chunks)

    print(full_text[:200])
    print("-" * 40)
    print(f"Total chunks: {len(chunks)}")

    # Optional: serve with playground UI
    # agent.serve(port=8000, enable_playground=True, debug=True)
