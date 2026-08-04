"""A resilient wrapper around the LLM adapters: retries + provider fallback.

Failure handling for the network boundary. Wraps one or more BaseLLM adapters:
  - retries a transient failure a few times with exponential backoff
  - if the primary provider keeps failing, falls back to the next one

The rest of the app keeps calling send_with_tools() — it doesn't know or care
that under the hood we might have retried twice and switched providers.
"""

import time


class AllProvidersFailed(Exception):
    pass


class ResilientLLM:
    def __init__(self, providers: list, max_retries: int = 2, base_delay: float = 0.5, logger=None):
        """providers: ordered list of (name, BaseLLM) — first is primary."""
        if not providers:
            raise ValueError("Need at least one provider.")
        self._providers = providers
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._logger = logger

    def send_with_tools(self, messages, tools):
        return self._call("send_with_tools", messages, tools)

    def send(self, messages):
        return self._call("send", messages)

    def _call(self, method, *args):
        last_error = None
        for name, llm in self._providers:
            for attempt in range(self._max_retries + 1):
                try:
                    result = getattr(llm, method)(*args)
                    if self._logger and attempt > 0:
                        self._logger.log("retry_success", provider=name, attempt=attempt)
                    return result
                except Exception as e:
                    last_error = e
                    if self._logger:
                        self._logger.log(
                            "provider_error", provider=name, attempt=attempt, error=repr(e)
                        )
                    if attempt < self._max_retries:
                        time.sleep(self._base_delay * (2 ** attempt))  # exponential backoff
                    # else: fall through to next provider
        raise AllProvidersFailed(f"All providers failed. Last error: {last_error!r}")
