"""Flask JSON-only API — no templates, no Jinja.

Run with::

    python -m backend.app          # direct
    flask --app backend.app run    # flask CLI
"""

from __future__ import annotations

import logging
import os
import uuid

import matplotlib
matplotlib.use("Agg")  # noqa: E402 — must be before any pyplot import

from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename

from backend.config import (
    UPLOAD_DIR,
    OUTPUT_DIR,
    SECRET_KEY,
    MAX_CONTENT_LENGTH,
    FLASK_DEBUG,
    apply_to_env,
)
from backend.pipeline_runner import start_run, get_run_status
from pipeline.storage import init_db, get_run, list_runs, get_chat_history
from pipeline.chat import handle_chat

logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {".dcm", ".dicom", ".DCM", ".DICOM"}


def create_app() -> Flask:
    """Application factory."""
    apply_to_env()

    app = Flask(__name__)
    app.config["SECRET_KEY"] = SECRET_KEY
    app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

    CORS(app)

    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    init_db()

    # ------------------------------------------------------------------ #
    # Routes                                                              #
    # ------------------------------------------------------------------ #

    # -- Root / Health Check ---------------------------------------------

    @app.route("/")
    def index():
        """API welcome page."""
        return jsonify({
            "name": "Medical Imaging QA - Backend API",
            "version": "1.0.0",
            "status": "running",
            "endpoints": {
                "upload": "POST /api/upload",
                "run": "POST /api/run",
                "runs": "GET /api/runs",
                "run_detail": "GET /api/runs/<id>",
                "run_status": "GET /api/runs/<id>/status",
                "report": "GET /api/runs/<id>/report",
                "before_after": "GET /api/runs/<id>/before_after",
                "chat": "POST /api/runs/<id>/chat",
                "logs": "GET /api/runs/<id>/logs"
            },
            "docs": "See README.md for full API documentation"
        })

    # -- Upload ----------------------------------------------------------

    @app.route("/api/upload", methods=["POST"])
    def api_upload():
        if "file" not in request.files:
            return jsonify({"error": "No file part in request"}), 400
        f = request.files["file"]
        if f.filename == "" or f.filename is None:
            return jsonify({"error": "No file selected"}), 400

        ext = os.path.splitext(f.filename)[1].lower()
        if ext not in {".dcm", ".dicom"}:
            return jsonify({"error": f"Unsupported file type: {ext}"}), 400

        file_id = uuid.uuid4().hex[:12]
        safe_name = secure_filename(f.filename) or f"{file_id}.dcm"
        stored_name = f"{file_id}_{safe_name}"
        dest = os.path.join(UPLOAD_DIR, stored_name)
        f.save(dest)
        return jsonify({"file_id": file_id, "filename": safe_name, "stored_name": stored_name}), 200

    # -- Run -------------------------------------------------------------

    @app.route("/api/run", methods=["POST"])
    def api_run():
        data = request.get_json(silent=True) or {}
        file_id = data.get("file_id")
        if not file_id:
            return jsonify({"error": "file_id is required"}), 400

        # Find the uploaded file by prefix
        stored = _find_upload(file_id)
        if stored is None:
            return jsonify({"error": f"Upload {file_id} not found"}), 404

        genai = bool(data.get("genai", False))
        model = data.get("model") or None
        max_iters = data.get("max_iters") or None
        if max_iters is not None:
            max_iters = int(max_iters)

        run_id = start_run(
            file_path=os.path.join(UPLOAD_DIR, stored),
            genai=genai,
            model=model,
            max_iters=max_iters,
        )
        return jsonify({"run_id": run_id, "status": "pending"}), 202

    # -- Runs list -------------------------------------------------------

    @app.route("/api/runs", methods=["GET"])
    def api_runs():
        limit = request.args.get("limit", 100, type=int)
        offset = request.args.get("offset", 0, type=int)
        runs = list_runs(limit=limit, offset=offset)
        # Return summary — strip heavy fields
        summaries = []
        for r in runs:
            summaries.append({
                "run_id": r["run_id"],
                "timestamp": r.get("timestamp", ""),
                "input_filename": r.get("input_filename", ""),
                "status": r.get("status", "unknown"),
                "issues": r.get("issues", []),
                "genai_model": r.get("genai_model", ""),
            })
        return jsonify({"runs": summaries}), 200

    # -- Single run detail ------------------------------------------------

    @app.route("/api/runs/<run_id>", methods=["GET"])
    def api_run_detail(run_id: str):
        data = get_run(run_id)
        if data is None:
            return jsonify({"error": "Run not found"}), 404
        # Include chat history
        data["chat_history"] = get_chat_history(run_id)
        return jsonify(data), 200

    # -- Run status (lightweight polling) ---------------------------------

    @app.route("/api/runs/<run_id>/status", methods=["GET"])
    def api_run_status(run_id: str):
        status = get_run_status(run_id)
        if status == "not_found":
            return jsonify({"error": "Run not found"}), 404
        return jsonify({"run_id": run_id, "status": status}), 200

    # -- Report markdown --------------------------------------------------

    @app.route("/api/runs/<run_id>/report", methods=["GET"])
    def api_report(run_id: str):
        data = get_run(run_id)
        if data is None:
            return jsonify({"error": "Run not found"}), 404

        report_path = data.get("report_path", "")
        if report_path and os.path.isfile(report_path):
            with open(report_path, encoding="utf-8") as f:
                md = f.read()
            return jsonify({"markdown": md}), 200

        return jsonify({"markdown": "", "note": "Report file not found"}), 200

    # -- Before/after image -----------------------------------------------

    @app.route("/api/runs/<run_id>/before_after", methods=["GET"])
    def api_before_after(run_id: str):
        data = get_run(run_id)
        if data is None:
            return jsonify({"error": "Run not found"}), 404

        ba_path = data.get("before_after_path", "")
        if ba_path and os.path.isfile(ba_path):
            return send_file(ba_path, mimetype="image/png")

        return jsonify({"error": "Image not found"}), 404

    # -- Chat -------------------------------------------------------------

    @app.route("/api/runs/<run_id>/chat", methods=["POST"])
    def api_chat(run_id: str):
        data_json = request.get_json(silent=True) or {}
        message = (data_json.get("message") or "").strip()
        if not message:
            return jsonify({"error": "message is required"}), 400
        if len(message) > 2000:
            return jsonify({"error": "Message too long (max 2000 chars)"}), 400

        model = data_json.get("model") or None
        try:
            reply = handle_chat(run_id, message, model=model)
            return jsonify({"reply": reply}), 200
        except Exception as exc:
            logger.exception("Chat error for run %s: %s", run_id, exc)
            return jsonify({"error": "Chat processing failed"}), 500

    # -- Logs -------------------------------------------------------------

    @app.route("/api/runs/<run_id>/logs", methods=["GET"])
    def api_logs(run_id: str):
        data = get_run(run_id)
        if data is None:
            return jsonify({"error": "Run not found"}), 404
        logs = data.get("agent_logs", [])
        return jsonify({"logs": logs}), 200

    # -- Serve output/upload files ----------------------------------------

    @app.route("/api/files/outputs/<path:filename>")
    def api_serve_output(filename: str):
        full = os.path.join(OUTPUT_DIR, filename)
        if os.path.isfile(full):
            return send_file(full)
        return jsonify({"error": "File not found"}), 404

    @app.route("/api/files/uploads/<path:filename>")
    def api_serve_upload(filename: str):
        full = os.path.join(UPLOAD_DIR, filename)
        if os.path.isfile(full):
            return send_file(full)
        return jsonify({"error": "File not found"}), 404

    # -- Error handlers ---------------------------------------------------

    @app.errorhandler(404)
    def not_found(e):
        return jsonify({"error": "Not found"}), 404

    @app.errorhandler(413)
    def too_large(e):
        return jsonify({"error": "File too large (max 50 MB)"}), 413

    @app.errorhandler(500)
    def internal_error(e):
        return jsonify({"error": "Internal server error"}), 500

    return app


def _find_upload(file_id: str) -> str | None:
    """Find an uploaded file by its file_id prefix."""
    for name in os.listdir(UPLOAD_DIR):
        if name.startswith(file_id):
            return name
    return None


# ====================================================================== #
# Direct run support: python -m backend.app                               #
# ====================================================================== #

app = create_app()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    app.run(host="0.0.0.0", port=5000, debug=FLASK_DEBUG)
