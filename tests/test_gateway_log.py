"""gateway.log (the daemon's stdout/stderr) stays bounded, and `logs` reads only the end of it."""

from __future__ import annotations

from aar_gateway import cli


def test_tail_returns_the_last_lines_of_a_file_larger_than_its_blocks(tmp_path):
    path = tmp_path / "gateway.log"
    path.write_bytes(b"".join(f"line {i}\n".encode() for i in range(50_000)))
    got = cli._tail(path, 5)
    assert got == [f"line {i}\n".encode() for i in range(49_995, 50_000)]


def test_tail_edge_cases(tmp_path):
    path = tmp_path / "x.log"
    path.write_bytes(b"")
    assert cli._tail(path, 3) == []
    path.write_bytes(b"a\nb")  # no trailing newline
    assert cli._tail(path, 5) == [b"a\n", b"b"]
    assert cli._tail(path, 0) == []
    assert cli._tail(path, 1) == [b"b"]


def test_logs_prints_the_end_without_reading_the_whole_file(tmp_path, monkeypatch, capsys):
    log = tmp_path / "gateway.log"
    log.write_bytes(b"".join(f"line {i}\n".encode() for i in range(100_000)))
    monkeypatch.setattr(cli, "_log_file", lambda: log)
    real_open = type(log).open
    read_total = []

    def counting_open(self, *a, **kw):
        f = real_open(self, *a, **kw)
        real_read = f.read
        f.read = lambda n=-1: (lambda b: (read_total.append(len(b)), b)[1])(real_read(n))
        return f

    monkeypatch.setattr(type(log), "open", counting_open)
    assert cli._logs(3, follow=False) == 0
    assert capsys.readouterr().out.splitlines() == ["line 99997", "line 99998", "line 99999"]
    assert sum(read_total) < log.stat().st_size // 10


def test_an_oversize_gateway_log_is_set_aside_before_a_start(tmp_path):
    log = tmp_path / "gateway.log"
    log.write_bytes(b"x" * 2000)
    cli._rotate_log(log, max_bytes=1000)
    assert not log.exists() and (tmp_path / "gateway.log.1").stat().st_size == 2000
    log.write_bytes(b"y" * 10)
    cli._rotate_log(log, max_bytes=1000)  # small: left alone
    assert log.read_bytes() == b"y" * 10
