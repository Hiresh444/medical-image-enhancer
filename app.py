"""Flask web application for the Medical Imaging QA pipeline.

Run with:
    python app.py

Or for production:
    gunicorn app:app
"""

from __future__ import annotations

import json
import logging
import os
import secrets
from datetime import datetime, timezone

import markdown
from flask import (
    Flask,
    abort,
    flash,
    jsonify,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

# Load dotenv early
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from pipeline.storage import init_db, get_run, list_runs, get_chat_history

logger = logging.getLogger("mdimg.web")

ALLOWED_EXTENSIONS = {".dcm", ".dicom", ".DCM", ".DICOM"}


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> Flask:
    """Application factory — returns a configured Flask instance."""
    _app = Flask(__name__)

    # --- Security ---
    _app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", secrets.token_hex(32))
    _app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50 MB max upload
    _app.config["UPLOAD_FOLDER"] = os.path.join(
        os.path.dirname(__file__), "uploads"
    )
    _app.config["OUTPUT_FOLDER"] = os.path.join(
        os.path.dirname(__file__), "outputs"
    )

    # CSRF protection
    try:
        from flask_wtf.csrf import CSRFProtect
        _csrf = CSRFProtect(_app)
    except ImportError:
        _csrf = None
        logger.warning("flask-wtf not installed — CSRF protection disabled.")

    # --- Helpers ---

    def _allowed_file(filename: str) -> bool:
        _, ext = os.path.splitext(filename)
        return ext.lower() in {e.lower() for e in ALLOWED_EXTENSIONS}

    def _ensure_dirs() -> None:
        os.makedirs(_app.config["UPLOAD_FOLDER"], exist_ok=True)
        os.makedirs(_app.config["OUTPUT_FOLDER"], exist_ok=True)

    # --- Routes ---

    @_app.route("/")
    def index():
        return render_template("upload.html")

    @_app.route("/run", methods=["POST"])
    def run():
        _ensure_dirs()

        if "file" not in request.files:
            flash("No file uploaded.", "danger")
            return redirect(url_for("index"))

        file = request.files["file"]
        if file.filename == "" or file.filename is None:
            flash("No file selected.", "danger")
            return redirect(url_for("index"))

        if not _allowed_file(file.filename):
            flash("Invalid file type. Only .dcm / .dicom files are allowed.", "danger")
            return redirect(url_for("index"))

        filename = secure_filename(file.filename)
        if not filename:
            flash("Invalid filename.", "danger")
            return redirect(url_for("index"))

        upload_path = os.path.join(_app.config["UPLOAD_FOLDER"], filename)
        file.save(upload_path)

        use_genai = request.form.get("genai", "off") == "on"
        model = request.form.get("model", os.environ.get("OPENAI_MODEL", "gpt-5-mini"))

        from pipeline.runner import run_pipeline

        try:
            context = run_pipeline(
                input_path=upload_path,
                output_dir=_app.config["OUTPUT_FOLDER"],
                genai=use_genai,
                model=model if use_genai else None,
                save_artifacts=True,
                no_show=True,
            )
        except Exception as exc:
            logger.exception("Pipeline failed")
            flash(f"Pipeline error: {exc}", "danger")
            return redirect(url_for("index"))

        run_id = context.get("run_id", "unknown")
        return redirect(url_for("result", run_id=run_id))

    @_app.route("/result/<run_id>")
    def result(run_id: str):
        run_data = get_run(run_id)
        if run_data is None:
            abort(404)
        chat_history = get_chat_history(run_id)
        return render_template("result.html", run=run_data, chat_history=chat_history)

    @_app.route("/reports")
    def reports():
        runs = list_runs(limit=200)
        return render_template("reports.html", runs=runs)

    @_app.route("/reports/<run_id>")
    def report_detail(run_id: str):
        run_data = get_run(run_id)
        if run_data is None:
            abort(404)

        report_path = run_data.get("report_path", "")
        report_html = ""
        if report_path and os.path.isfile(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                report_md_raw = f.read()
            report_html = markdown.markdown(
                report_md_raw,
                extensions=["tables", "fenced_code", "codehilite"],
            )

        return render_template(
            "report_detail.html",
            run_id=run_id,
            run=run_data,
            report_html=report_html,
        )

    @_app.route("/reports/<run_id>/download")
    def download_report(run_id: str):
        run_data = get_run(run_id)
        if run_data is None:
            abort(404)
        report_path = run_data.get("report_path", "")
        if not report_path or not os.path.isfile(report_path):
            abort(404)
        directory = os.path.dirname(report_path)
        fname = os.path.basename(report_path)
        return send_from_directory(directory, fname, as_attachment=True)

    @_app.route("/logs/<run_id>")
    def logs(run_id: str):
        run_data = get_run(run_id)
        if run_data is None:
            abort(404)
        traces = run_data.get("agent_traces", [])
        return render_template("logs.html", run_id=run_id, traces=traces)

    @_app.route("/api/chat", methods=["POST"])
    def api_chat():
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid JSON"}), 400

        run_id = data.get("run_id", "")
        user_message = data.get("message", "").strip()

        if not run_id:
            return jsonify({"error": "run_id is required"}), 400
        if not user_message:
            return jsonify({"error": "message is required"}), 400
        if len(user_message) > 2000:
            return jsonify({"error": "Message too long (max 2000 chars)"}), 400

        run_data = get_run(run_id)
        if run_data is None:
            return jsonify({"error": f"Run '{run_id}' not found"}), 404

        from pipeline.chat import handle_chat

        try:
            response = handle_chat(run_id=run_id, user_message=user_message)
        except Exception as exc:
            logger.exception("Chat failed")
            response = f"Error processing your question: {exc}"

        return jsonify({"reply": response})

    # Exempt chat from CSRF
    if _csrf is not None:
        _csrf.exempt(api_chat)

    @_app.route("/outputs/<path:filename>")
    def serve_output(filename: str):
        safe = secure_filename(filename)
        if not safe:
            abort(404)
        return send_from_directory(_app.config["OUTPUT_FOLDER"], safe)

    @_app.route("/uploads/<path:filename>")
    def serve_upload(filename: str):
        safe = secure_filename(filename)
        if not safe:
            abort(404)
        return send_from_directory(_app.config["UPLOAD_FOLDER"], safe)

    # --- Error handlers ---
    @_app.errorhandler(404)
    def not_found(e):
        return render_template("error.html", message="Page not found", code=404), 404

    @_app.errorhandler(413)
    def too_large(e):
        flash("File too large. Maximum upload size is 50 MB.", "danger")
        return redirect(url_for("index"))

    @_app.errorhandler(500)
    def server_error(e):
        return render_template("error.html", message="Internal server error", code=500), 500

    return _app


# Module-level app instance for gunicorn / direct run
app = create_app()


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    init_db()
    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    port = int(os.environ.get("FLASK_PORT", "5000"))

    print(f"\n  Medical Imaging QA Web UI")
    print(f"  http://127.0.0.1:{port}\n")

    app.run(host="0.0.0.0", port=port, debug=debug)
