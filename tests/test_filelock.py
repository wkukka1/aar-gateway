"""The share queue is written by the gateway and trimmed by the CLI: two processes."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from aar_gateway import share

SRC = str(Path(__file__).resolve().parents[3] / "src")

_COUNTER = """
import sys
sys.path.insert(0, {src!r})
from pathlib import Path
from aar_gateway.config import locked_file
p = Path({path!r})
for _ in range(60):
    with locked_file(p):
        n = int(p.read_text()) if p.exists() else 0
        p.write_text(str(n + 1))
"""


def test_locked_file_excludes_other_processes(tmp_path):
    target = tmp_path / "counter.txt"
    code = _COUNTER.format(src=SRC, path=str(target))
    procs = [subprocess.Popen([sys.executable, "-c", code]) for _ in range(3)]
    assert all(p.wait(timeout=60) == 0 for p in procs)
    assert int(target.read_text()) == 180  # no lost update


def test_a_record_queued_by_another_process_during_the_trim_survives(tmp_path, monkeypatch):
    queue = share.queue_path()
    sent = [json.dumps({"n": i}) + "\n" for i in range(3)]
    queue.parent.mkdir(parents=True, exist_ok=True)
    queue.write_text("".join(sent), encoding="utf-8")
    share.set_endpoint("https://collect.example.org/v1/records")
    child: list[subprocess.Popen] = []
    real_write = share.atomic_write_text

    def write_while_cli_appends(path, text):
        if Path(path) == queue:  # the trim: another process (the gateway) queues a record now
            code = (f"import sys; sys.path.insert(0, {SRC!r}); from aar_gateway import share; "
                    "share._queue_line('{\"late\": 1}\\n')")
            child.append(subprocess.Popen([sys.executable, "-c", code],
                                          env={**__import__("os").environ, "AAR_GATEWAY_LOG": str(tmp_path / "requests.jsonl")}))
            time.sleep(1.5)  # long enough for an unlocked append to land before the replace
        real_write(path, text)

    monkeypatch.setattr(share, "atomic_write_text", write_while_cli_appends)

    assert share.upload(post=lambda url, body, headers: 200)["sent"] == 3
    assert child[0].wait(timeout=60) == 0
    assert [json.loads(line) for line in queue.read_text(encoding="utf-8").splitlines()] == [{"late": 1}]
