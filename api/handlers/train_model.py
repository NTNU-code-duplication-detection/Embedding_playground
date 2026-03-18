"""Handler for the /train endpoint.

Receives a JSON training configuration from the frontend and returns
the resolved model name upon success.
"""
from flask import Blueprint, request, jsonify

train_bp = Blueprint("train", __name__)


@train_bp.route("/train", methods=["POST"])
def train_model():
    """Accept a training config JSON and return the model name.
       receive body (JSON):
        {
            "backend":        "huggingface" | "sentence_transformers",
            "model_name":     "<hf model id>",
            "pipelines":      ["embedding_baseline", "homogeneous_gnn", "relational_gnn"],
            "objective":      "pairwise" | "triplet",
            "schedule":       "epochs" | "steps",
            "max_epochs":     5,
            "max_steps":      1000,
            "batch_size":     32,
            "eval_batch_size": 64,
            "search_backend": "none" | "optuna",
            "trials":         10,
            "dataset": {
                "source":                "custom_llm_pairs",
                "clone_types":           ["t3", "t4"],
                "include_negative_pairs": true,
                "split": {
                    "train": 0.70,
                    "val":   0.15,
                    "test":  0.15,
                    "stratified_by": "label",
                    "seed":  11
                }
            },
            "statement_embedding": {
                "batch_size":           16,
                "chunk_weighting_mode": "depth_aware" | "uniform"
            },
            "gnn_encoder": {
                "num_layers":    2,
                "hidden_dim":    256,
                "output_dim":    256,
                "edge_type_dim": 32
            }
        }
    """
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
