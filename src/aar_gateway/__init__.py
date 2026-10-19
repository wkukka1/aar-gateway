"""Local gateway that runs Claude Code through the router.

Claude Code points at it (``ANTHROPIC_BASE_URL``, or ``aar-gateway code``);
at every new human message the trained router picks the model for that turn
(:mod:`.routing`), Claude models are forwarded to Anthropic on the user's own
login and other models get the request translated with its tools
(:mod:`.translate`), and every request is logged with its cost against two
baselines (:mod:`.interceptor`, :mod:`.pricing`) for the dashboard.
See ``docs/gateway.md``.
"""
