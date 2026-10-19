"""/health reports the router the interceptor actually has, not a copy taken at startup."""

from __future__ import annotations

from fastapi.testclient import TestClient

from aar_gateway import interceptor, server
from aar_gateway.config import load_settings


def _picker(kind="knn"):
    def pick(body, ctx):
        return None

    pick.router_kind = kind
    pick.calibration_status = lambda: {"graded_prompts": 7, "offsets": {"claude-haiku-5-5": 0.1}}
    return pick


def test_health_follows_the_interceptors_state_when_it_is_reconfigured():
    settings = load_settings(None)
    with TestClient(server.app) as client:
        interceptor.configure(settings, _picker("nirt"), router_error=None)
        router = client.get("/health").json()["router"]
        assert router["enabled"] is True and router["kind"] == "nirt" and router["error"] is None
        assert router["calibration"]["graded_prompts"] == 7
        assert router["models"] == sorted(settings["models"])
        interceptor.configure(settings, None, router_error="FileNotFoundError: router.npz")
        router = client.get("/health").json()["router"]
        assert router["enabled"] is False and router["error"].startswith("FileNotFoundError")
        assert "calibration" not in router


def test_router_status_names_where_the_router_came_from():
    settings = load_settings(None)
    settings["router"].update(artifact=None)
    interceptor.configure(settings, _picker("knn"))
    assert interceptor.router_status()["source"] is None
    settings["router"].update(artifact="router.npz")
    assert interceptor.router_status()["source"] == "router.npz"
