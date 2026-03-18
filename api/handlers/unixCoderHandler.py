# pylint: disable=invalid-name
"""Handler for UniXcoder-based plagiarism detection endpoint.

Accepts two Java project folders (lists of files), chunks every file with
TreeSitterChunker, embeds each chunk with UniXcoder, then computes pairwise
cosine similarity to identify and return suspicious chunk pairs.

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


def _validate_folder_items(folder: list, name: str) -> str | None:
    """Return an error string if any file object in folder is invalid, else None."""
    for i, f in enumerate(folder):
        content = f.get("content") if isinstance(f, dict) else None
        if not isinstance(content, str) or not content.strip():
            return f"{name}[{i}] must have a non-empty 'content' string."
    return None


def _validate_request(body: dict | None):
    """Validate and extract fields from the request body.

    Returns (error_message, folder1, folder2, threshold, model_name, pipeline).
    error_message is None when inputs are valid.
    """
    if not body:
        return "Request body must be JSON.", None, None, 0.0, None, None

    folder1 = body.get("folder1")
    folder2 = body.get("folder2")
    threshold = body.get("threshold", DEFAULT_THRESHOLD)
    model_name = body.get("model_name")
    pipeline = body.get("pipeline")

    if not isinstance(folder1, list) or not folder1:
        return (
            "'folder1' must be a non-empty list of file objects.",
            None, None, 0.0, None, None,
        )
    if not isinstance(folder2, list) or not folder2:
        return (
            "'folder2' must be a non-empty list of file objects.",
            None, None, 0.0, None, None,
        )

    error = _validate_folder_items(folder1, "folder1") or \
            _validate_folder_items(folder2, "folder2")
    if error:
        return error, None, None, 0.0, None, None

    if not isinstance(threshold, (int, float)) or not 0.0 <= float(threshold) <= 1.0:
        return "'threshold' must be a float between 0 and 1.", None, None, 0.0, None, None

    return None, folder1, folder2, float(threshold), model_name, pipeline


def _chunks_from_folder(chunker: TreeSitterChunker, folder: list[dict]) -> list[Chunk]:
    """Chunk all files in a folder and return a flat list of chunks."""
    all_chunks = []
    for file in folder:
        all_chunks.extend(chunker.chunk_function(file["content"]))
    return all_chunks


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@unixcoder_bp.route("/UnixCoderSimilarity", methods=["POST"])
def check_similarity():
    """
    POST /UnixCoderSimilarity

    Request body (JSON):
        {
            "folder1":    [{"name": "<filename>", "content": "<Java source>"}, ...],
            "folder2":    [{"name": "<filename>", "content": "<Java source>"}, ...],
            "threshold":   0.8,           (optional, default 0.8)
            "model_name": "<model id>",   (optional)
            "pipeline":   "<pipeline>",   (optional)
        }

    Response (JSON):
        {
            "is_plagiarism":      bool,
            "overall_similarity": float,
            "threshold_used":     float,
            "suspicious_chunks":  [...]
        }
    """
    error, folder1, folder2, threshold, _, _ = _validate_request(request.get_json(silent=True))
    if error:
        return jsonify({"error": error}), 400

    chunker, embedder = _get_models()

    chunks1 = _chunks_from_folder(chunker, folder1)
    chunks2 = _chunks_from_folder(chunker, folder2)

    if not chunks1:
        return jsonify({
            "error": "No chunks extracted from folder1. Provide valid Java files."
        }), 400
    if not chunks2:
        return jsonify({
            "error": "No chunks extracted from folder2. Provide valid Java files."
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

    # Overall similarity: mean of each folder1 chunk's best match in folder2
    overall_similarity = round(sim_matrix.max(dim=1).values.mean().item(), 4)

    return jsonify({
        "is_plagiarism": overall_similarity >= threshold,
        "overall_similarity": overall_similarity,
        "threshold_used": threshold,
        "suspicious_chunks": suspicious,
    }), 200
