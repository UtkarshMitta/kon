"""
HuggingFace Webhook listener for auto-triggering retraining.
When the dataset repo on HF Hub is updated, this listener triggers retraining.

This can be deployed as a HuggingFace Space or run locally.
"""

import json
import sys
import os
import hmac
import hashlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import HF_DATASET_ID, HF_TOKEN

# ──────────────────────────────────────────────
# Webhook listener using Flask (lightweight)
# ──────────────────────────────────────────────

try:
    from flask import Flask, request, jsonify
except ImportError:
    print("Flask not installed. Run: pip install flask")
    print("This is an optional component for automated retraining.")
    sys.exit(0)

app = Flask(__name__)

# Webhook secret for verifying HF webhooks
WEBHOOK_SECRET = os.getenv("HF_WEBHOOK_SECRET", "")


def verify_webhook_signature(payload: bytes, signature: str) -> bool:
    """Verify the webhook signature from HuggingFace."""
    if not WEBHOOK_SECRET:
        return True  # No secret configured, accept all

    expected = hmac.new(
        WEBHOOK_SECRET.encode(), payload, hashlib.sha256
    ).hexdigest()
    return hmac.compare_digest(f"sha256={expected}", signature)


@app.route("/webhook", methods=["POST"])
def handle_webhook():
    """Handle incoming HuggingFace webhook."""
    # Verify signature
    signature = request.headers.get("X-Webhook-Secret", "")
    if not verify_webhook_signature(request.data, signature):
        return jsonify({"error": "Invalid signature"}), 401

    payload = request.json
    event = payload.get("event", {})
    repo = payload.get("repo", {})

    # Log the event
    event_type = event.get("action", "unknown")
    repo_name = repo.get("name", "unknown")
    print(f"📬 Webhook received: {event_type} on {repo_name}")

    # Only trigger on dataset repo updates
    if repo.get("type") != "dataset":
        return jsonify({"status": "ignored", "reason": "not a dataset repo"}), 200

    if repo_name != HF_DATASET_ID:
        return jsonify({"status": "ignored", "reason": f"not our dataset ({HF_DATASET_ID})"}), 200

    if event_type != "update":
        return jsonify({"status": "ignored", "reason": "not an update event"}), 200

    # Trigger retraining
    print(f"🚀 Dataset updated! Triggering retraining...")
    try:
        from feedback.retrain_trigger import trigger_retrain
        trigger_retrain(method="transformers")
        return jsonify({"status": "retrain_triggered"}), 200
    except Exception as e:
        print(f"❌ Retrain failed: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "dataset_id": HF_DATASET_ID,
        "webhook_secret_configured": bool(WEBHOOK_SECRET),
    })


@app.route("/status", methods=["GET"])
def status():
    """Get current feedback and model status."""
    from feedback.collect import FeedbackCollector

    collector = FeedbackCollector()
    stats = collector.get_stats()

    return jsonify({
        "feedback": stats,
        "dataset_id": HF_DATASET_ID,
    })


def main():
    import argparse

    parser = argparse.ArgumentParser(description="HF Webhook listener for auto-retraining")
    parser.add_argument("--port", type=int, default=7860, help="Port to listen on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    print(f"🌐 Starting webhook listener on {args.host}:{args.port}")
    print(f"   Dataset: {HF_DATASET_ID}")
    print(f"   Webhook secret: {'configured' if WEBHOOK_SECRET else 'not configured'}")
    print(f"\n   Endpoints:")
    print(f"     POST /webhook  — HF webhook handler")
    print(f"     GET  /health   — Health check")
    print(f"     GET  /status   — Feedback status")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
