"""Where the gateway listens is read and validated once; a LAN address is not locked out by its own allowlist."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from aar_gateway import cli, listen, server


def test_defaults(monkeypatch):
    monkeypatch.delenv("AAR_GATEWAY_HOST", raising=False)
    monkeypatch.delenv("AAR_GATEWAY_PORT", raising=False)
    assert listen.listen_address() == ("127.0.0.1", 8787)
    assert listen.base_url() == "http://127.0.0.1:8787"


def test_host_and_port_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_HOST", "192.168.1.5")
    monkeypatch.setenv("AAR_GATEWAY_PORT", "9000")
    assert listen.listen_address() == ("192.168.1.5", 9000) and cli._base_url() == "http://192.168.1.5:9000"


@pytest.mark.parametrize("port", ["abc", "0", "70000", "-1", "80.5"])
def test_a_bad_port_is_a_clear_error_not_a_traceback(monkeypatch, port):
    monkeypatch.setenv("AAR_GATEWAY_PORT", port)
    with pytest.raises(ValueError, match="AAR_GATEWAY_PORT"):
        listen.listen_address()
    with pytest.raises(SystemExit, match="AAR_GATEWAY_PORT"):
        cli._base_url()


def test_the_address_the_gateway_listens_on_is_one_it_answers_to(monkeypatch):
    """Serving on a LAN address used to 403 every request unless AAR_GATEWAY_ALLOWED_HOSTS also named it."""
    monkeypatch.setenv("AAR_GATEWAY_HOST", "192.168.1.5")
    monkeypatch.delenv("AAR_GATEWAY_ALLOWED_HOSTS", raising=False)
    with TestClient(server.app, base_url="http://192.168.1.5:8787") as client:
        assert client.get("/health").status_code == 200
        assert client.get("/health", headers={"host": "attacker.example"}).status_code == 403


def test_a_wildcard_bind_does_not_open_the_allowlist(monkeypatch):
    monkeypatch.setenv("AAR_GATEWAY_HOST", "0.0.0.0")
    monkeypatch.delenv("AAR_GATEWAY_ALLOWED_HOSTS", raising=False)
    assert "0.0.0.0" not in server._allowed_hosts()
