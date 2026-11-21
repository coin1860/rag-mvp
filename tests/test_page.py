"""The page contract: HTML, self-contained, wired to /ask."""

from fastapi.testclient import TestClient

import app as app_module


def test_root_is_html_not_json():
    client = TestClient(app_module.app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/html")
    assert not resp.text.lstrip().startswith('"<'), "response is a JSON-encoded string"
    assert resp.text.lstrip().startswith("<!doctype html")


def test_page_is_self_contained_and_posts_to_ask():
    html = TestClient(app_module.app).get("/").text
    assert "fetch('/ask'" in html
    assert "http://" not in html and "https://" not in html, "page must not fetch external assets"
    assert "<style>" in html and "<script>" in html
