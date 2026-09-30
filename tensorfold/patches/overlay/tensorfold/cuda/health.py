"""patches/0150: liveness for ``tensorfold.cuda.server``: ``GET /health`` that can say no, and ``GET /metrics``.

Upstream's ``/health`` answers ``{"ok": true}`` whatever the engine is doing. With two ranks that is not enough: a
CUDA error or an out-of-memory in the middle of a request leaves rank 1 waiting in a collective forever, the next
request hangs behind it, and ``/health`` still says ok (the MiaAI-Lab and vLLM-kit watchdogs found the same with
vLLM: ``/health`` stays 200 through a stuck NCCL collective, so they watch forward progress instead). This module
keeps, per server:

- ``fatal``: the first exception (other than ``ValueError``: a bad request) raised out of ``engine.generate``. The
  ranks are out of step after one, so nothing after it can be trusted.
- the requests in flight, each with its prompt length, start time and the time of its last token, so a request that
  stopped making progress is visible (a stall).
- counters for ``/metrics`` (Prometheus text): requests, errors, prompt / completion / cached tokens, decode rounds
  and seconds, prefill seconds. Tokens a round over an interval is the drafter's health (a broken or mismatched
  drafter decodes one token a round).

``GLM53_TF_HEALTH``:

- ``basic`` (default): ``/health`` stays 200 and adds the fields below (``fatal``, ``stalled``, ``inflight``, ...),
  so nothing that polls it today changes behaviour.
- ``strict``: ``/health`` answers 503 once ``fatal`` is set or while a request is stalled, and new completions get a
  503 right away once ``fatal`` is set (instead of queueing behind a hung collective). ``scripts/serve.sh watch``
  restarts both ranks on it.

A request is stalled when its last token (or its start, before the first token) is older than ``GLM53_TF_STALL_S``
(default 0: never) plus, before the first token, its prompt at ``GLM53_TF_STALL_PREFILL_TPS`` tokens a second
(default 200, well under the slowest measured prefill, so a 1M prompt gets ~83 min before it counts).
"""

from __future__ import annotations

import functools
import os
import threading
import time
from typing import Any

MODES = ("basic", "strict")


def env_mode() -> str:
    mode = os.environ.get("GLM53_TF_HEALTH", "basic").strip().lower() or "basic"
    if mode not in MODES:
        raise ValueError(f"GLM53_TF_HEALTH={mode!r}: expected one of {', '.join(MODES)}")
    return mode


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    value = float(raw)
    if value < 0:
        raise ValueError(f"{name}={raw}: must be >= 0")
    return value


class Health:
    """Thread-safe request bookkeeping for ``/health`` and ``/metrics``. ``clock``: for tests."""

    def __init__(self, *, mode: str | None = None, stall_s: float | None = None, prefill_tps: float | None = None,
                 clock=time.monotonic) -> None:
        self.mode = mode or env_mode()
        self.stall_s = _env_float("GLM53_TF_STALL_S", 0.0) if stall_s is None else float(stall_s)
        self.prefill_tps = _env_float("GLM53_TF_STALL_PREFILL_TPS", 200.0) if prefill_tps is None \
            else float(prefill_tps)
        self.clock = clock
        self.started = clock()
        self.lock = threading.Lock()
        self.fatal: str | None = None
        self.inflight: dict[int, dict[str, float]] = {}
        self.next_id = 0
        self.last_done = None
        self.counters: dict[str, float] = {k: 0 for k in (
            "requests", "errors", "rejected", "prompt_tokens", "completion_tokens", "cached_tokens",
            "decode_rounds", "decode_seconds", "prefill_seconds")}

    # -- request lifecycle -------------------------------------------------------------------------------------

    def begin(self, prompt_tokens: int) -> int:
        with self.lock:
            rid = self.next_id
            self.next_id += 1
            now = self.clock()
            self.inflight[rid] = {"prompt": float(prompt_tokens), "start": now, "last": now, "tokens": 0.0}
            return rid

    def progress(self, rid: int, n: int) -> None:
        with self.lock:
            entry = self.inflight.get(rid)
            if entry is not None:
                entry["last"] = self.clock()
                entry["tokens"] += n
                # patches/9003: counters move while the reply streams (a long reply no longer lands at once)
                self.counters["completion_tokens"] += n
                if not entry.get("prompt_counted"):
                    entry["prompt_counted"] = 1.0
                    self.counters["prompt_tokens"] += entry["prompt"]

    def end(self, rid: int, *, stats: dict[str, Any] | None = None, completion_tokens: int = 0,
            error: BaseException | None = None) -> None:
        with self.lock:
            entry = self.inflight.pop(rid, None)
            self.last_done = self.clock()
            c = self.counters
            c["requests"] += 1
            if error is not None:
                c["errors"] += 1
                if not isinstance(error, ValueError) and self.fatal is None:
                    self.fatal = f"{type(error).__name__}: {error}"[:500]
                return
            if entry is not None and not entry.get("prompt_counted"):
                c["prompt_tokens"] += entry["prompt"]
            c["completion_tokens"] += max(0.0, completion_tokens - (entry["tokens"] if entry else 0.0))   # 9003
            stats = stats or {}
            c["cached_tokens"] += float(stats.get("cached", 0) or 0)
            c["decode_rounds"] += float(stats.get("rounds", 0) or 0)
            c["decode_seconds"] += float(stats.get("decode_s", 0) or 0)
            c["prefill_seconds"] += float(stats.get("prefill_s", 0) or 0)

    def reject(self) -> str | None:
        """Why a new completion must not start (strict mode after a fatal error), or None."""

        if self.mode == "strict" and self.fatal is not None:
            with self.lock:
                self.counters["rejected"] += 1
            return f"the engine failed earlier ({self.fatal}); restart both ranks"
        return None

    def track(self, engine) -> "TrackedEngine":
        return TrackedEngine(engine, self)

    # -- views -------------------------------------------------------------------------------------------------

    def _stalled(self, now: float) -> list[dict[str, float]]:
        if self.stall_s <= 0:
            return []
        out = []
        for entry in self.inflight.values():
            allow = self.stall_s
            if entry["tokens"] == 0 and self.prefill_tps > 0:
                allow += entry["prompt"] / self.prefill_tps
            quiet = now - entry["last"]
            if quiet > allow:
                out.append({"prompt": entry["prompt"], "quiet_s": round(quiet, 1), "allowed_s": round(allow, 1)})
        return out

    def status(self) -> tuple[int, dict[str, Any]]:
        """(HTTP code, body) for ``GET /health``."""

        with self.lock:
            now = self.clock()
            stalled = self._stalled(now)
            body: dict[str, Any] = {
                "ok": self.fatal is None and not stalled,
                "mode": self.mode,
                "uptime_s": round(now - self.started, 1),
                "inflight": len(self.inflight),
                "oldest_s": round(max((now - e["start"] for e in self.inflight.values()), default=0.0), 1),
                "idle_s": round(now - self.last_done, 1) if self.last_done is not None and not self.inflight
                else None,
                "requests": int(self.counters["requests"]),
                "errors": int(self.counters["errors"]),
            }
            if self.fatal is not None:
                body["fatal"] = self.fatal
            if stalled:
                body["stalled"] = stalled
        code = 503 if self.mode == "strict" and not body["ok"] else 200
        return code, body

    def metrics(self, served: str = "") -> str:
        """Prometheus text exposition (counters and gauges, one ``model`` label)."""

        with self.lock:
            now = self.clock()
            c = dict(self.counters)
            inflight = len(self.inflight)
            stalled = len(self._stalled(now))
            fatal = int(self.fatal is not None)
            uptime = now - self.started
        label = '{model="%s"}' % served.replace("\\", "\\\\").replace('"', '\\"')
        rows = [
            ("tensorfold_requests_total", "counter", "completions finished (errors included)", c["requests"]),
            ("tensorfold_request_errors_total", "counter", "completions that raised", c["errors"]),
            ("tensorfold_requests_rejected_total", "counter", "completions refused after a fatal error",
             c["rejected"]),
            ("tensorfold_prompt_tokens_total", "counter", "prompt tokens of finished completions", c["prompt_tokens"]),
            ("tensorfold_cached_tokens_total", "counter", "prompt tokens resumed instead of prefilled",
             c["cached_tokens"]),
            ("tensorfold_completion_tokens_total", "counter", "generated tokens", c["completion_tokens"]),
            ("tensorfold_decode_rounds_total", "counter", "decode rounds (tokens / rounds: draft acceptance)",
             c["decode_rounds"]),
            ("tensorfold_decode_seconds_total", "counter", "seconds decoding", c["decode_seconds"]),
            ("tensorfold_prefill_seconds_total", "counter", "seconds prefilling", c["prefill_seconds"]),
            ("tensorfold_requests_inflight", "gauge", "completions running now", inflight),
            ("tensorfold_requests_stalled", "gauge", "running completions past GLM53_TF_STALL_S", stalled),
            ("tensorfold_engine_fatal", "gauge", "1 once the engine raised (restart both ranks)", fatal),
            ("tensorfold_uptime_seconds", "gauge", "seconds since the server started", uptime),
        ]
        out = []
        for name, kind, doc, value in rows:
            value = float(value)
            shown = str(int(value)) if value.is_integer() else f"{value:.6f}"
            out += [f"# HELP {name} {doc}", f"# TYPE {name} {kind}", f"{name}{label} {shown}"]
        return "\n".join(out) + "\n"


class TrackedEngine:
    """``engine`` with ``generate`` going through ``Health`` (begin, a progress mark per token callback, end with
    the stats or the exception). Every other attribute is the engine's own (read and written through)."""

    def __init__(self, engine, health: Health) -> None:
        object.__setattr__(self, "_engine", engine)
        object.__setattr__(self, "_health", health)
        inner = engine.generate

        @functools.wraps(inner)                 # keeps the signature: ``App.check`` inspects it for ``draft``
        def generate(prompt, max_tokens, sampling, on_tokens, *args, **kwargs):
            rid = health.begin(len(prompt))
            count = [0]

            def tracked(new):
                count[0] += len(new)
                health.progress(rid, len(new))
                return on_tokens(new)

            try:
                stats = inner(prompt, max_tokens, sampling, tracked, *args, **kwargs)
            except BaseException as exc:
                health.end(rid, error=exc)
                raise
            health.end(rid, stats=stats if isinstance(stats, dict) else None, completion_tokens=count[0])
            return stats

        object.__setattr__(self, "generate", generate)

    def __getattr__(self, name: str):
        return getattr(self._engine, name)

    def __setattr__(self, name: str, value) -> None:
        setattr(self._engine, name, value)
