"""CI never downloads MiniLM. Live `python app.py` still auto-detects .[rag]."""

from __future__ import annotations

import os

os.environ["AGENTSHIELD_RAG_HASH"] = "1"
