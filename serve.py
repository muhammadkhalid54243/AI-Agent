"""Run the framework's chat server.

    uv run python serve.py     # then open http://127.0.0.1:8000
"""

import os

if __name__ == "__main__":
    import uvicorn
    from framework.serve.app import app
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
