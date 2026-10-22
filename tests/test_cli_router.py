"""``aar-gateway router [list|add|use]``: the router files on this machine, and
switching which one the gateway serves without touching the config."""

from __future__ import annotations

import json

import pytest

from aar_gateway import cli
from aar_gateway._core.scorers import load_router
from router_files import save_mlp
from test_router_file import mlp_payload

ENC = {"backend": "sentence_transformer", "model_name": "x"}


@pytest.fixture
def gateway(tmp_path, monkeypatch):
    """A home folder set up with the bundled MLP, and a gateway that isn't running."""
    monkeypatch.setattr(cli, "_doctor", lambda: 0)
    monkeypatch.setattr(cli, "_healthy", lambda url: False)
    monkeypatch.delenv("AAR_GATEWAY_ROUTER_URL", raising=False)
    assert cli.main(["setup", "--no-enable", "--no-share"]) == 0
    return tmp_path  # AAR_GATEWAY_LOG lives in tmp_path, so it is the gateway home (conftest)


def rows(capsys):
    capsys.readouterr()
    assert cli.main(["router", "list", "--json"]) == 0
    return {r["name"]: r for r in json.loads(capsys.readouterr().out)}


def test_list_shows_the_bundled_routers_and_marks_the_one_served(gateway, capsys):
    got = rows(capsys)
    assert got["llmrouterbench-mlp"]["active"] and not got["llmrouterbench-nirt"]["active"]
    assert got["llmrouterbench-nirt"]["kind"] == "mlp" and got["llmrouterbench-nirt"]["models"] == 13
    assert got["llmrouterbench-nirt"]["id"] == "nirt-lrb-nirt-projected"


def test_use_swaps_the_router_file_and_leaves_the_config_alone(gateway, capsys):
    config = (gateway / "config.yaml").read_bytes()
    assert cli.main(["router", "use", "llmrouterbench-nirt"]) == 0
    assert (gateway / "config.yaml").read_bytes() == config
    assert load_router(gateway / "router.npz").name == "nirt-lrb-nirt-projected"
    assert rows(capsys)["llmrouterbench-nirt"]["active"]
    assert cli.main(["router", "use", "llmrouterbench-mlp"]) == 0
    assert load_router(gateway / "router.npz").name == "mlp-llmrouterbench-h256"


def test_use_restarts_a_running_gateway_only(gateway, monkeypatch):
    restarts = []
    monkeypatch.setattr(cli, "_restart", lambda: restarts.append(1))
    cli.main(["router", "use", "llmrouterbench-nirt"])
    assert restarts == []
    monkeypatch.setattr(cli, "_healthy", lambda url: True)
    cli.main(["router", "use", "llmrouterbench-mlp"])
    assert restarts == [1]


def test_use_an_unknown_name_fails_and_lists_the_choices(gateway):
    before = (gateway / "router.npz").read_bytes()
    with pytest.raises(SystemExit, match="llmrouterbench-nirt"):
        cli.main(["router", "use", "nope"])
    assert (gateway / "router.npz").read_bytes() == before


def test_add_then_use_your_own_router_file(gateway, tmp_path, capsys):
    source = tmp_path / "mine.npz"
    save_mlp(mlp_payload(), source, encoder=ENC, meta={"artifact_id": "my-router"})
    assert cli.main(["router", "add", str(source), "--name", "toy"]) == 0
    assert (gateway / "routers" / "toy.npz").exists()
    assert rows(capsys)["toy"]["source"] == "added"
    assert cli.main(["router", "use", "toy"]) == 0
    assert load_router(gateway / "router.npz").name == "my-router"


def test_add_refuses_a_file_that_is_not_a_router(gateway, tmp_path):
    bad = tmp_path / "bad.npz"
    bad.write_bytes(b"nope")
    with pytest.raises(SystemExit):
        cli.main(["router", "add", str(bad)])
    assert not (gateway / "routers" / "bad.npz").exists()


def test_a_file_without_an_artifact_id_keeps_its_old_name(tmp_path):
    path = tmp_path / "plain.npz"
    save_mlp(mlp_payload(), path, encoder=ENC)
    assert load_router(path).name == "mlp:plain"
