"""The user's graded turns as routing data: similarity-weighted grades for
the real tiers, or nothing when too few graded prompts are close enough."""

from __future__ import annotations

import json

import numpy as np

from aar_gateway.graded import graded_predictor, load_graded

TOPICS = {"sql": [1, 0, 0], "css": [0, 1, 0], "poem": [0, 0, 1]}


def encode(texts):
    vecs = np.array([[1.0 if t in text else 0.0 for t in TOPICS] for text in texts]) + 1e-3
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows))


def graded_rows(n_per_topic=4):
    rows = []
    for topic, (opus, haiku) in {"sql": (0.9, 0.3), "css": (0.8, 0.8)}.items():
        for i in range(n_per_topic):
            h = f"{topic}{i}"
            rows += [{"prompt_hash": h, "prompt": f"{topic} question {i}", "model": "claude-opus-5-5", "score": opus},
                     {"prompt_hash": h, "prompt": f"{topic} question {i}", "model": "claude-haiku-5-5", "score": haiku}]
    return rows


def test_load_graded_builds_a_prompt_by_model_matrix(tmp_path):
    p = tmp_path / "graded.jsonl"
    write(p, graded_rows(1) + [{"prompt_hash": "x", "prompt": "p", "model": "m", "score": None}])
    prompts, models, scores = load_graded(p)
    assert len(prompts) == 2 and models == ["claude-haiku-5-5", "claude-opus-5-5"] and not np.isnan(scores).any()


def test_similar_graded_prompts_predict_each_tier(tmp_path):
    p = tmp_path / "graded.jsonl"
    write(p, graded_rows())
    predict = graded_predictor(p, encode, min_prompts=4, min_neighbors=3, k=4)
    sql = predict("a new sql question")
    assert sql["quality"] == {"claude-haiku-5-5": 0.3, "claude-opus-5-5": 0.9}
    css = predict("css layout please")
    assert abs(css["quality"]["claude-haiku-5-5"] - 0.8) < 1e-9


def test_nothing_similar_or_too_little_data_means_no_prediction(tmp_path):
    p = tmp_path / "graded.jsonl"
    write(p, graded_rows())
    assert graded_predictor(p, encode, min_prompts=4)("write me a poem") is None
    assert graded_predictor(p, encode, min_prompts=100)("sql again") is None


def test_new_grades_are_picked_up_without_a_restart(tmp_path):
    p = tmp_path / "graded.jsonl"
    write(p, graded_rows(1))
    predict = graded_predictor(p, encode, min_prompts=4)
    assert predict("sql") is None
    import os
    import time
    write(p, graded_rows(4))
    os.utime(p, (time.time() + 5, time.time() + 5))
    assert predict("sql")["quality"]["claude-opus-5-5"] == 0.9


def _rows(n, model="claude-opus-5-5", start=0):
    return "".join(json.dumps({"prompt_hash": f"h{i}", "prompt": f"q{i}", "model": model, "score": 0.5}) + "\n"
                   for i in range(start, start + n))


def test_a_malformed_graded_row_is_skipped_not_fatal(tmp_path):
    path = tmp_path / "graded.jsonl"
    path.write_text(_rows(3) + json.dumps({"prompt": "no hash", "model": "m", "score": 1.0}) + "\n"
                    + json.dumps({"prompt_hash": "x", "prompt": "no model", "score": 1.0}) + "\n"
                    + json.dumps({"prompt_hash": "y", "prompt": "bad score", "model": "m", "score": "high"}) + "\n")
    prompts, models, scores = load_graded(path)
    assert len(prompts) == 3 and models == ["claude-opus-5-5"]


def test_the_graded_file_is_parsed_once_per_change_however_many_readers_ask(tmp_path, monkeypatch):
    from aar_gateway import graded

    path = tmp_path / "graded.jsonl"
    path.write_text(_rows(4))
    reads = []
    real = graded.read_jsonl
    monkeypatch.setattr(graded, "read_jsonl", lambda p: reads.append(p) or real(p))
    first = load_graded(path)
    second = load_graded(path)
    assert reads == [path] and first[0] == second[0]
    path.write_text(_rows(4) + _rows(2, start=4))  # the grader appended
    assert len(load_graded(path)[0]) == 6 and len(reads) == 2
