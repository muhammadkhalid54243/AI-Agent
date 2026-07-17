# AI Agent

A hands-on learning project: building generative & agentic AI systems from a single raw
LLM call up to a production-grade agent — one milestone at a time.

See [`ROADMAP.md`](ROADMAP.md) for the full 11-milestone build ladder and
[`PROGRESS.md`](PROGRESS.md) for where the project currently stands.

## Project scope

Each milestone is a small, working proof-of-concept, built and understood before moving on
— no skipping ahead. Current focus: **Milestone 1 — multi-turn chatbot with memory.**

## Providers

Multiple LLM providers are supported side by side: Groq, Google AI Studio, Claude
(Anthropic), and OpenRouter. Only Groq is wired up so far; the others get added the same
way as the roadmap reaches them.

## Structure

```
agent/
  chatbot.py          Chatbot — owns the conversation, doesn't know which provider it's using
  llms/
    groq/
      config.py        GroqConfig — loads GROQ_API_KEY / GROQ_MODEL from .env
      llm.py            GroqLLM — the only place that calls the Groq SDK
    <provider>/         same shape (config.py + llm.py) for each provider as it's added
main.py                 wires a provider's config -> its llm client -> Chatbot
```

Every provider folder exposes the same `send(messages) -> str` shape on its LLM class, so
`Chatbot` can hold any of them interchangeably without caring which SDK is underneath.

## Setup

```bash
uv sync
```

Create a `.env` file in the project root (never committed — see `.gitignore`):

```
GROQ_API_KEY=your-groq-api-key
```

## Run

```bash
uv run python main.py
```
