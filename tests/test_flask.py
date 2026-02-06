"""Tests for the Flask web app (app.py)."""

from __future__ import annotations

import pytest

from app import create_app
from pipeline.storage import init_db


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    init_db()
    with app.test_client() as c:
        yield c


class TestRoutes:
    def test_upload_page(self, client):
        rv = client.get("/")
        assert rv.status_code == 200
        assert b"Upload" in rv.data or b"upload" in rv.data

    def test_reports_page(self, client):
        rv = client.get("/reports")
        assert rv.status_code == 200

    def test_404(self, client):
        rv = client.get("/nonexistent")
        assert rv.status_code == 404

    def test_upload_no_file(self, client):
        rv = client.post("/run", data={})
        # Should redirect back or show error
        assert rv.status_code in (302, 400, 200)
