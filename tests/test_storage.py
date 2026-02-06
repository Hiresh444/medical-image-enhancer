"""Tests for pipeline.storage – SQLite persistence."""

from __future__ import annotations

import os
import tempfile

import pytest

# Override DB path before importing storage
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
os.environ["MDIMG_DB_PATH"] = _tmp.name

from pipeline.storage import (
    init_db,
    generate_run_id,
    save_run,
    get_run,
    list_runs,
    save_chat_message,
    get_chat_history,
)


@pytest.fixture(autouse=True)
def _fresh_db():
    """Re-create the DB before each test."""
    init_db()
    yield
    # clean up
    try:
        os.unlink(os.environ["MDIMG_DB_PATH"])
    except OSError:
        pass
    _new = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    _new.close()
    os.environ["MDIMG_DB_PATH"] = _new.name


class TestRunPersistence:
    def test_generate_run_id(self):
        rid = generate_run_id()
        assert isinstance(rid, str)
        assert len(rid) > 10

    def test_save_and_get_run(self):
        rid = generate_run_id()
        save_run(
            run_id=rid,
            input_filename="test.dcm",
            metadata_summary={"Modality": "CT"},
            issues=["noise"],
            metrics_before={"sigma": 0.12},
            metrics_after={"sigma": 0.05},
            plan_json='{"ops": ["denoise"]}',
            validation={"ssim": 0.92, "psnr": 30.0, "passes": True},
            applied_ops=["denoise"],
            explainability={"detected_issues": "Noise"},
            report_path="outputs/test_report.md",
            before_after_path="outputs/test_before_after.png",
            agent_logs=[{"phase": "detection"}],
            status="PASS",
            genai_model="gpt-5-mini",
        )
        row = get_run(rid)
        assert row is not None
        assert row["input_filename"] == "test.dcm"
        assert row["status"] == "PASS"
        assert isinstance(row["issues"], list)
        assert row["issues"][0] == "noise"

    def test_list_runs(self):
        for i in range(3):
            rid = generate_run_id()
            save_run(
                run_id=rid,
                input_filename=f"file{i}.dcm",
                metadata_summary={},
                issues=[],
                metrics_before={},
                metrics_after={},
                plan_json="",
                validation={},
                applied_ops=[],
                explainability="",
                report_path="",
                before_after_path="",
                agent_logs=[],
                status="PASS",
            )
        runs = list_runs()
        assert len(runs) >= 3


class TestChat:
    def test_save_and_get_chat(self):
        rid = generate_run_id()
        save_run(
            run_id=rid, input_filename="test.dcm",
            metadata_summary={}, issues=[], metrics_before={},
            metrics_after={}, plan_json="", validation={},
            applied_ops=[], explainability="", report_path="",
            before_after_path="", agent_logs=[], status="PASS",
        )
        save_chat_message(rid, "user", "What is SSIM?")
        save_chat_message(rid, "assistant", "SSIM is...")
        history = get_chat_history(rid)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"
