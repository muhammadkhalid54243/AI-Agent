from agent.llms.base import BaseLLM

PROVIDERS = ("groq", "openroute", "anthropic", "google", "openai")


def get_llm(provider: str) -> BaseLLM:
    """Build the right LLM adapter from a provider name string."""
    provider = provider.lower()

    if provider == "groq":
        from agent.llms.groq.config import GroqConfig
        from agent.llms.groq.llm import GroqLLM
        config = GroqConfig()
        config.validate()
        return GroqLLM(config)

    if provider == "openroute":
        from agent.llms.openroute.config import OpenrouteConfig
        from agent.llms.openroute.llm import OpenrouteLLM
        config = OpenrouteConfig()
        config.validate()
        return OpenrouteLLM(config)

    if provider == "anthropic":
        from agent.llms.anthropic.config import AnthropicConfig
        from agent.llms.anthropic.llm import AnthropicLLM
        config = AnthropicConfig()
        config.validate()
        return AnthropicLLM(config)

    if provider == "google":
        from agent.llms.google.config import GoogleConfig
        from agent.llms.google.llm import GoogleLLM
        config = GoogleConfig()
        config.validate()
        return GoogleLLM(config)

    if provider == "openai":
        from agent.llms.openai.config import OpenAIConfig
        from agent.llms.openai.llm import OpenAILLM
        config = OpenAIConfig()
        config.validate()
        return OpenAILLM(config)

    raise ValueError(f"Unknown provider: {provider!r}. Choose from: {', '.join(PROVIDERS)}")
