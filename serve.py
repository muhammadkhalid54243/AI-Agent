"""Run the framework's chat server.

    uv run python serve.py     # then open http://127.0.0.1:8000
"""

import os

if __name__ == "__main__":
    import uvicorn
    from framework.serve.app import app, get_agent

    # Warm up BEFORE serving: load the embedding model + build the SHR index once,
    # upfront, so the first real request is fast instead of blocking for minutes.
    # (Within a running server the model stays loaded — it never reloads per request.)
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    print("Warming up: loading models + building the document index (one-time; "
          "the server is NOT accepting requests yet)...", flush=True)
    get_agent()
    print(f"Ready — server starting. Open http://{host}:{port}", flush=True)
    uvicorn.run(app, host=host, port=port)
