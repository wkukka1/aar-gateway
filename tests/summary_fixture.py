"""A deterministic synthetic request log with every shape /stats cares about."""

from __future__ import annotations

MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5", "local-coder"]


def make_records(n: int, *, start: int = 0) -> list[dict]:
    out = []
    for i in range(start, start + n):
        routed, asked = MODELS[i % 4], MODELS[(i // 3) % 3]
        sub = i % 5 == 0
        rec = {
            "ts": f"2026-10-{1 + (i // 40) % 28:02d}T{(i // 2) % 24:02d}:{i % 60:02d}:00+00:00",
            "claude_session": None if i % 11 == 0 else f"sess{(i // 13) % 6}",
            "conversation": f"conv{i // 4}",
            "new_turn": i % 4 == 0 and not sub,
            "subagent": sub,
            "agent_role": ["explore", "plan", None][i % 3] if sub else None,
            "role_source": "hook" if sub else None,
            "reason": "side request" if i % 17 == 0 else "cheapest within 0.05",
            "requested_model": asked, "routed_model": routed,
            "fallback": i % 19 == 0,
            "status": 404 if i % 23 == 0 else (500 if i % 29 == 0 else 200),
            "error": "No thread state for the given message id" if i % 23 == 0 else None,
            "input_tokens": 100 + i, "output_tokens": 20 + i % 7,
            "cache_read_input_tokens": 1000 + 3 * i, "cache_creation_input_tokens": 50 + i % 5,
            "cost_usd": None if i % 31 == 0 else round(0.001 * (1 + i % 9), 6),
            "cost_if_requested_usd": None if i % 37 == 0 else round(0.003 * (1 + i % 9), 6),
            "cost_if_strongest_usd": None if i % 41 == 0 else round(0.005 * (1 + i % 9), 6),
            "predicted_quality": None if i % 6 == 0 else round(0.5 + (i % 10) / 20, 3),
            "best_predicted_quality": None if i % 7 == 0 else round(0.6 + (i % 10) / 25, 3),
            "strongest_model": MODELS[0] if i % 8 else None,
            "latency_s": round(0.5 + (i % 13) / 10, 3),
            "report_tokens": 40 + i % 9 if sub and i % 2 == 0 else None,
        }
        out.append(rec)
    return out
