"""Handler for the /train endpoint.

Receives a JSON training configuration from the frontend and returns
the resolved model name upon success.
"""
from flask import Blueprint, request, jsonify

train_bp = Blueprint("train", __name__)


@train_bp.route("/train", methods=["POST"])
def train_model():
    """Accept a training config JSON and return the model name."""
    body = request.get_json(force=True, silent=True)
    if body is None:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    model_name = body.get("model_name")
    if not model_name:
        return jsonify({"error": "model_name is required"}), 400

    # Launch training using the parsed config.
    # Config fields available:
    #   backend, model_name, pipelines, objective, schedule,
    #   max_epochs, max_steps, batch_size, eval_batch_size,
    #   search_backend, trials, dataset, statement_embedding, gnn_encoder

    return jsonify({"model_name": model_name}), 200
