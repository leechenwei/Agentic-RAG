"""Package init — must run BEFORE any submodule imports chromadb.

Streamlit Community Cloud (and other older Linux distros) ships a
sqlite3 too old for ChromaDB (which needs >= 3.35). We pull in
pysqlite3-binary and swap it into sys.modules under the 'sqlite3'
name so any later `import sqlite3` (including ChromaDB's) picks up
the modern version. No-op on systems whose stdlib sqlite3 is fine.
"""
try:
    __import__("pysqlite3")
    import sys
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    # pysqlite3-binary not installed (e.g. local dev on macOS where
    # the system sqlite3 is already new enough) — leave stdlib sqlite3 in place.
    pass
