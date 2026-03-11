# pylint: disable=invalid-name
"""Handler for UniXcoder-based plagiarism detection endpoint.

Accepts two Java code snippets, chunks them with TreeSitterChunker,
embeds each chunk with UniXcoder, then computes pairwise cosine
similarity to identify and return suspicious chunk pairs.

The model and chunker are lazy-loaded once and reused across requests.
"""

import threading

import torch
import torch.nn.functional as F
from flask import Blueprint, jsonify, request

from chunk_gnn.data.chunker import Chunk, TreeSitterChunker
from chunk_gnn.data.embedder import ChunkEmbedder

unixcoder_bp = Blueprint("unixcoder", __name__)

# ---------------------------------------------------------------------------
# Lazy singleton — model is loaded once on first request
# ---------------------------------------------------------------------------

_models: dict = {}
_model_lock = threading.Lock()

DEFAULT_THRESHOLD = 0.8


def _get_models() -> tuple[TreeSitterChunker, ChunkEmbedder]:
    """Return (chunker, embedder), loading them on first call."""
    if "chunker" not in _models:
        with _model_lock:
            if "chunker" not in _models:
                _models["chunker"] = TreeSitterChunker()
                _models["embedder"] = ChunkEmbedder()
    return _models["chunker"], _models["embedder"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk_to_dict(chunk: Chunk) -> dict:
    return {
        "text": chunk.text,
        "kind": chunk.kind.value,
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "depth": chunk.depth,
    }


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def _validate_request(body: dict | None) -> tuple[str | None, str, str, float]:
    """Validate and extract fields from the request body.

    Returns (error_message, code1, code2, threshold).
    error_message is None when inputs are valid.
    """
    if not body:
        return "Request body must be JSON.", "", "", 0.0

    code1 = body.get("code1", "")
    code2 = body.get("code2", "")
    threshold = body.get("threshold", DEFAULT_THRESHOLD)

    if not isinstance(code1, str) or not code1.strip():
        return "'code1' must be a non-empty string.", "", "", 0.0
    if not isinstance(code2, str) or not code2.strip():
        return "'code2' must be a non-empty string.", "", "", 0.0
    if not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0:
        return "'threshold' must be a float between 0 and 1.", "", "", 0.0

    return None, code1, code2, float(threshold)


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@unixcoder_bp.route("/UnixCoderSimilarity", methods=["POST"])
def check_similarity():
    """
    POST /UnixCoderSimilarity

    Request body (JSON):
        {
            "code1":     "<Java method source>",
            "code2":     "<Java method source>",
            "threshold":  0.8          (optional, default 0.8)
        }

    Both code1 and code2 should be single Java methods (no class wrapper
    needed). The threshold controls what cosine similarity counts as
    a suspicious chunk match.

    Response (JSON):
        {
            "is_plagiarism":      bool,
            "overall_similarity": float,
            "threshold_used":     float,
            "suspicious_chunks":  [
                {
                    "similarity":  float,
                    "code1_chunk": { "text", "kind", "start_line",
                                     "end_line", "depth" },
                    "code2_chunk": { "text", "kind", "start_line",
                                     "end_line", "depth" }
                },
                ...
            ]
        }
    """
    error, code1, code2, threshold = _validate_request(request.get_json(silent=True))
    if error:
        return jsonify({"error": error}), 400

    chunker, embedder = _get_models()
    chunks1 = chunker.chunk_function(code1)
    chunks2 = chunker.chunk_function(code2)

    if not chunks1:
        return jsonify({
            "error": "No chunks extracted from code1. Provide a valid Java method."
        }), 400
    if not chunks2:
        return jsonify({
            "error": "No chunks extracted from code2. Provide a valid Java method."
        }), 400

    try:
        emb1 = embedder.embed_chunks(chunks1)   # (n1, 768)
        emb2 = embedder.embed_chunks(chunks2)   # (n2, 768)
    except RuntimeError as exc:
        return jsonify({"error": f"Embedding failed: {exc}"}), 500

    # Pairwise cosine similarity matrix  (n1 x n2)
    sim_matrix = torch.mm(
        F.normalize(emb1, dim=1),
        F.normalize(emb2, dim=1).T,
    )

    # Collect all chunk pairs above threshold, sorted by similarity
    suspicious = sorted(
        [
            {
                "similarity": round(sim_matrix[i, j].item(), 4),
                "code1_chunk": _chunk_to_dict(chunks1[i]),
                "code2_chunk": _chunk_to_dict(chunks2[j]),
            }
            for i in range(len(chunks1))
            for j in range(len(chunks2))
            if sim_matrix[i, j].item() >= threshold
        ],
        key=lambda x: x["similarity"],
        reverse=True,
    )

    # Overall similarity: mean of each code1 chunk's best match in code2
    overall_similarity = round(sim_matrix.max(dim=1).values.mean().item(), 4)

    return jsonify({
        "is_plagiarism": overall_similarity >= threshold,
        "overall_similarity": overall_similarity,
        "threshold_used": threshold,
        "suspicious_chunks": suspicious,
    }), 200
