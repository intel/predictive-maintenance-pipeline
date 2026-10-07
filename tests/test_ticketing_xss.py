import re

import pytest

import src.agents.ticketing_agent as ticketing
import src.utility.sqlite_client as sqlite_client
import web_app.app as web_app_module
from src.agents.ticketing_agent import (
    create_ticket,
    generate_ticket_html,
    validate_frame_id,
)

XSS = '<img src=x onerror=alert(1)>'
FILENAME_RE = re.compile(r'^ticket_[A-Za-z0-9_-]+\.html$')


class FakeDB:
    def __init__(self, rows=None, **_kwargs):
        self.rows = rows or []

    def execute_query(self, _sql, _params=()):
        return self.rows

    def close(self):
        pass


@pytest.fixture
def out_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ticketing, "_get_out_dir", lambda: tmp_path)
    monkeypatch.setattr(web_app_module, "_get_out_dir", lambda: tmp_path)
    return tmp_path


@pytest.fixture
def client(out_dir, monkeypatch):
    monkeypatch.setattr(web_app_module, "_load_full_config", lambda: {})
    monkeypatch.setattr(sqlite_client, "SQLiteClient", FakeDB)
    web_app_module.app.config["TESTING"] = True
    return web_app_module.app.test_client()


@pytest.mark.parametrize("value", [0, 5, 123456, "5", "img_001.jpg", "row_12", "a-b.png", "dir/img.jpg"])
def test_validate_frame_id_accepts_legitimate_ids(value):
    assert validate_frame_id(value) == value


@pytest.mark.parametrize("value", [
    XSS, "", "../etc", ".hidden", "a b", "a'b", 'a"b', "a<b", "x" * 129,
    True, False, -1, 1.5, [1], {"a": 1}, None,
])
def test_validate_frame_id_rejects_unsafe_values(value):
    with pytest.raises(ValueError):
        validate_frame_id(value)


def test_generate_ticket_html_escapes_all_interpolated_values():
    detections = [
        {"label": XSS, "confidence": 0.9, "x": XSS, "y": 1, "width": 2, "height": XSS},
    ]
    html = generate_ticket_html(XSS, detections, XSS, XSS, None, include_image=True)
    assert "<img src=x" not in html
    assert "&lt;img src=x onerror=alert(1)&gt;" in html

    classification = [{"source": XSS, "label": XSS, "confidence": 0.5}]
    html = generate_ticket_html('a"b', classification, "", None, None, include_image=True)
    assert "<img src=x" not in html
    assert 'alt="a"b"' not in html


def test_create_ticket_rejects_malicious_frame_id(out_dir):
    with pytest.raises(ValueError):
        create_ticket(FakeDB(), XSS, include_image=False)
    assert not (out_dir / "tickets").exists()


@pytest.mark.parametrize("frame_id,expected", [
    (5, "ticket_5.html"),
    ("5", "ticket_5.html"),
    ("img_001.jpg", "ticket_img_001_jpg.html"),
    ("dir/img.png", "ticket_dir_img_png.html"),
])
def test_create_ticket_filenames_are_safe(out_dir, frame_id, expected):
    result = create_ticket(FakeDB(), frame_id, include_image=False)
    assert result["ok"] is True
    assert result["filename"] == expected
    assert FILENAME_RE.fullmatch(result["filename"])
    assert (out_dir / "tickets" / expected).is_file()


@pytest.mark.parametrize("payload", [
    {"frame_id": XSS},
    {"frame_id": "<img src=x onerror=eval(atob('aW1wb3J0'))>"},
    {"frame_id": True},
    {"frame_id": [1]},
    {"frame_id": "x" * 300},
    {},
    [1, 2],
])
def test_ticket_create_endpoint_rejects_bad_input(client, out_dir, payload):
    resp = client.post("/api/ticket/create", json=payload)
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False
    assert not (out_dir / "tickets").exists()


def test_ticket_create_list_and_serve_roundtrip(client, monkeypatch):
    rows = [{"source": "img_001.jpg", "label": XSS, "confidence": 0.8}]
    monkeypatch.setattr(sqlite_client, "SQLiteClient", lambda **_kw: FakeDB(rows))

    resp = client.post("/api/ticket/create", json={"frame_id": "img_001.jpg"})
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert data["filename"] == "ticket_img_001_jpg.html"

    tickets = client.get("/api/tickets").get_json()["tickets"]
    assert [t["sample_id"] for t in tickets] == ["img_001.jpg"]

    served = client.get(f"/ticket/{data['filename']}")
    assert served.status_code == 200
    body = served.get_data(as_text=True)
    assert "<img src=x" not in body
    assert "sandbox" in served.headers["Content-Security-Policy"]
    assert served.headers["X-Content-Type-Options"] == "nosniff"


def test_security_headers_on_regular_responses(client):
    resp = client.get("/api/tickets")
    assert resp.headers["X-Content-Type-Options"] == "nosniff"
    assert "object-src 'none'" in resp.headers["Content-Security-Policy"]
