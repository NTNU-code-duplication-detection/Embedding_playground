"""
API server entry point.

Run with:
    python api/server.py

Listens on http://127.0.0.100:3000
"""

import os
import sys

# Ensure project root is on the path regardless of how this script is invoked
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# pylint: disable=wrong-import-position
from flask import Flask

from api.handlers.ast_handler import ast_bp
from api.settings import HOST, PORT
# pylint: enable=wrong-import-position

app = Flask(__name__)

# Register handlers — add new blueprints here as the API grows
app.register_blueprint(ast_bp)


if __name__ == "__main__":
    print(f"Starting API server on http://{HOST}:{PORT}")
    app.run(host=HOST, port=PORT, debug=False)
