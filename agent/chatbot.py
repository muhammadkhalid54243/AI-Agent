class Chatbot:
    """Orchestrates a conversation — talks to an LLM client, doesn't know which provider."""

    def __init__(self, llm_client):
        self._llm_client = llm_client

    def ask(self, user_message):
        messages = [{"role": "user", "content": user_message}]
        return self._llm_client.send(messages)
