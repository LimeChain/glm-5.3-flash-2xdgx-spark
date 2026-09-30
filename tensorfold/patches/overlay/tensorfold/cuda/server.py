"""OpenAI-compatible server for TensorFold's CUDA engines (DGX Spark and other NVIDIA GPUs).

One request decodes at a time, as one exact stream: drafted output is byte-identical to serial decoding on the
same engine. The chat template is the model's own ``chat_template.jinja``; Qwen tool calls are parsed from the
reply; with thinking on, text before ``</think>`` streams as ``reasoning_content``.

A family's CUDA engine (``cuda_engine`` in its package) provides:

    eos                                              # token ids that end a reply
    generate(prompt, max_tokens, sampling, on_tokens) -> dict
                                                     # decode; on_tokens(new_ids) returns True to stop early
    follow()                                         # two GPUs: rank 1 mirrors every request rank 0 serves

``tensorfold serve MODEL`` builds the engine and serves it here (see ``tensorfold.cli``).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
import uuid
from datetime import datetime
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Callable

_THINK_END = "</think>"
_CALL_OPEN, _CALL_CLOSE = "<tool_call>", "</tool_call>"
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_TOOL_FUNCTION_BLOCK_RE = re.compile(r"^\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*$", re.IGNORECASE | re.DOTALL)
_TOOL_PARAMETER_BLOCK_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.IGNORECASE | re.DOTALL)
# GLM-4.5 to 5.3: <tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value>...</tool_call>
_GLM_ARG_RE = re.compile(r"<arg_key>\s*(.*?)\s*</arg_key>\s*<arg_value>(.*?)</arg_value>", re.DOTALL)


# -- text helpers (same rules as the Mac lane server) ---------------------------------------

def _partial_tag(text: str, tag: str) -> int:
    for k in range(min(len(tag) - 1, len(text)), 0, -1):
        if text.endswith(tag[:k]):
            return k
    return 0


class StreamDecoder:
    """The text of a growing token list, decoding only a short window each time tokens arrive.

    Decoding the whole reply every round grows with its length. Each step decodes the tokens from
    ``prefix`` on twice, with and without the newest ones, and appends the difference, the way vLLM
    detokenizes: the shared window keeps a decoder's leading-space and byte-level rules the same on
    both sides. A trailing partial multi-byte character waits for the next tokens.
    """

    def __init__(self, tok, skip: tuple[int, ...] = ()):
        self.tok, self.skip = tok, frozenset(skip)
        self.ids: list[int] = []
        self.text = ""
        self.prefix = 0             # window start
        self.read = 0               # tokens already reflected in ``text``

    def _decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    def add(self, new: list[int]) -> str:
        self.ids.extend(t for t in new if t not in self.skip)
        before = self._decode(self.ids[self.prefix:self.read])
        after = self._decode(self.ids[self.prefix:])
        if len(after) > len(before) and not after.endswith("\ufffd"):
            self.text += after[len(before):]
            self.prefix, self.read = self.read, len(self.ids)
        return self.text

    def final(self) -> str:
        """Everything, including a trailing partial character (as decoding it all at once gives)."""

        before = self._decode(self.ids[self.prefix:self.read])
        return self.text + self._decode(self.ids[self.prefix:])[len(before):]


def split_thinking(text: str, *, finished: bool) -> tuple[str, str]:
    end = text.find(_THINK_END)
    if end < 0:
        return text[: len(text) - (0 if finished else _partial_tag(text, _THINK_END))], ""
    return text[:end], text[end + len(_THINK_END):].lstrip("\n")


def hide_tool_calls(text: str, *, finished: bool) -> str:
    out: list[str] = []
    pos = 0
    while True:
        start = text.find(_CALL_OPEN, pos)
        if start < 0:
            tail = text[pos:]
            out.append(tail[: len(tail) - (0 if finished else _partial_tag(tail, _CALL_OPEN))])
            return "".join(out)
        out.append(text[pos:start])
        end = text.find(_CALL_CLOSE, start)
        if end < 0:
            return "".join(out)
        pos = end + len(_CALL_CLOSE)


def _visible_spans(text: str, *, finished: bool) -> tuple[str, list[tuple[int, int]]]:
    """patches/0160: ``hide_tool_calls``'s text plus, for each kept piece, (its start in the result, its start in
    ``text``), so a position in the visible text maps back to the raw text."""

    out: list[str] = []
    spans: list[tuple[int, int]] = []
    pos = n = 0
    while True:
        start = text.find(_CALL_OPEN, pos)
        if start < 0:
            tail = text[pos:]
            tail = tail[: len(tail) - (0 if finished else _partial_tag(tail, _CALL_OPEN))]
            spans.append((n, pos))
            out.append(tail)
            return "".join(out), spans
        spans.append((n, pos))
        out.append(text[pos:start])
        n += start - pos
        end = text.find(_CALL_CLOSE, start)
        if end < 0:
            return "".join(out), spans
        pos = end + len(_CALL_CLOSE)


def _raw_index(spans: list[tuple[int, int]], index: int) -> int:
    """patches/0160: the raw position of visible position ``index`` (``_visible_spans``)."""

    vis, raw = 0, 0
    for v, r in spans:
        if v > index:
            break
        vis, raw = v, r
    return raw + index - vis


def parse_stop(value: Any) -> list[str] | None:
    """patches/0160: OpenAI's ``stop`` (a string or a list of strings) as a list without empty strings; raises
    ValueError for anything else."""

    if value is None:
        return []
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise ValueError("stop must be a string or a list of strings")
    return [v for v in value if v]


def find_stop(text: str, stops: list[str], start: int = 0) -> int:
    """patches/0160: the earliest position (from ``start``) where any stop string begins, or -1."""

    best = -1
    for s in stops:
        i = text.find(s, start)
        if i >= 0 and (best < 0 or i < best):
            best = i
    return best


def stop_holdback(text: str, stops: list[str]) -> int:
    """patches/0160: how many trailing characters could still begin a stop string (held back while streaming)."""

    return max((_partial_tag(text, s) for s in stops), default=0)


# patches/9004: per-round series in the engine's stats grow with the reply (one entry a decode round). They are
# benchmark data; a long reply's arrays exceed strict clients' JSON bounds and bloat every response.
STATS_SERIES = frozenset({"keeps", "depths", "drafters", "arms", "stats"})
STATS_MODES = ("summary", "full", "off")


def stats_mode(body: dict[str, Any]) -> str:
    """patches/9004: ``tensorfold_stats`` in the request, else GLM53_TF_STATS, else ``summary``."""

    mode = body.get("tensorfold_stats")
    if mode is None:
        mode = (os.environ.get("GLM53_TF_STATS") or "summary").strip().lower()
    return mode if mode in STATS_MODES else "summary"


def response_stats(stats: dict[str, Any] | None, mode: str) -> dict[str, Any] | None:
    """patches/9004: the ``tensorfold`` object for a response. ``summary`` drops the per-round series and keeps the
    scalars (rounds, decode_s, tokens_per_round, ...) the canary and dashboards read; ``full`` is the engine's stats;
    ``off`` omits the object."""

    if mode == "off" or stats is None:
        return None
    if mode == "full":
        return stats
    return {key: value for key, value in stats.items() if key not in STATS_SERIES}


def reasoning_fields(text: str) -> dict[str, str]:
    """patches/0160: the reply's reasoning under the field names GLM53_TF_REASONING_FIELDS asks for:
    ``both`` (default: ``reasoning_content`` and ``reasoning``), ``reasoning_content`` or ``reasoning``."""

    mode = (os.environ.get("GLM53_TF_REASONING_FIELDS") or "both").strip().lower()
    if mode == "reasoning_content":
        return {"reasoning_content": text}
    if mode == "reasoning":
        return {"reasoning": text}
    return {"reasoning_content": text, "reasoning": text}


def _tool_name(tool: dict[str, Any]) -> str:
    fn = tool.get("function") if isinstance(tool, dict) else None
    return str((fn or tool).get("name") or "").strip() if isinstance(tool, dict) else ""


def _glm_call(block: str, tools: list[dict[str, Any]]) -> tuple[str | None, dict[str, Any]]:
    """GLM's ``name<arg_key>k</arg_key><arg_value>v</arg_value>...`` body, the way vLLM's glm47 parser reads it:
    the name is the text before the first ``<arg_key>``; a value is kept as text when the tool's schema types the
    parameter as a string, and read as JSON otherwise (falling back to the text)."""

    cut = block.find("<arg_key>")
    name = (block if cut < 0 else block[:cut]).strip()
    if not name or "<" in name or "\n" in name:
        return None, {}
    schema: dict[str, Any] = {}
    for t in tools:
        if _tool_name(t).lower() == name.lower():
            fn = t.get("function") if isinstance(t.get("function"), dict) else t
            schema = ((fn.get("parameters") or {}).get("properties") or {}) if isinstance(fn, dict) else {}
    args: dict[str, Any] = {}
    for m in _GLM_ARG_RE.finditer(block):
        key, raw = m.group(1).strip(), m.group(2)
        kind = (schema.get(key) or {}).get("type") if isinstance(schema.get(key), dict) else None
        if kind == "string":
            args[key] = raw
            continue
        try:
            args[key] = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            args[key] = raw
    return name, args


def parse_tool_calls(text: str, tools: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]] | None]:
    """Qwen ``<tool_call><function=name><parameter=k>v</parameter></function></tool_call>``, GLM
    ``<tool_call>name<arg_key>k</arg_key><arg_value>v</arg_value></tool_call>``, or JSON bodies."""

    if not tools:
        return text, None
    known = {_tool_name(t).lower(): _tool_name(t) for t in tools}
    calls: list[dict[str, Any]] = []
    residue: list[str] = []
    cursor = 0
    for match in _TOOL_CALL_BLOCK_RE.finditer(text):
        residue.append(text[cursor:match.start()])
        cursor = match.end()
        block = match.group(1).strip()
        name, args = None, {}
        try:
            payload = json.loads(block)
            if isinstance(payload, dict):
                fn = payload.get("function") if isinstance(payload.get("function"), dict) else payload
                name = fn.get("name")
                args = fn.get("arguments", fn.get("parameters", {}))
                if isinstance(args, str):
                    args = json.loads(args) if args.strip() else {}
        except (json.JSONDecodeError, AttributeError):
            m = _TOOL_FUNCTION_BLOCK_RE.match(block)
            if m:
                name = m.group(1).strip()
                args = {p.group(1).strip(): p.group(2) for p in _TOOL_PARAMETER_BLOCK_RE.finditer(m.group(2))}
            else:
                name, args = _glm_call(block, tools)
        if not name or str(name).lower() not in known:
            residue.append(match.group(0))
            continue
        calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": known[str(name).lower()],
                                   "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":"))}})
    residue.append(text[cursor:])
    return "".join(residue).strip(), calls or None


# -- patches/0490: OpenAI-shaped request errors, the context limit, token-ID prompts ----------------------------------

class Problem(str):
    """patches/0490: a request error message that carries an OpenAI error ``code`` and ``param`` (``check`` may
    return one instead of a plain string; the handler puts both in the error body)."""

    code: str | None = None
    param: str | None = None

    def __new__(cls, message: str, code: str | None = None, param: str | None = None) -> "Problem":
        self = super().__new__(cls, message)
        self.code, self.param = code, param
        return self


def error_body(message: str, kind: str = "invalid_request_error") -> dict[str, Any]:
    """patches/0490: ``{"error": {...}}`` as OpenAI sends it: message, type, and a ``Problem``'s code / param."""

    err: dict[str, Any] = {"message": str(message), "type": kind}
    code, param = getattr(message, "code", None), getattr(message, "param", None)
    if code is not None:
        err["code"] = code
    if param is not None:
        err["param"] = param
    return {"error": err}


def context_problem(prompt: int, asked: int | None, limit: int, *, chat: bool = True) -> Problem | None:
    """patches/0490: None when ``prompt`` tokens plus ``asked`` reply tokens (1 without max_tokens) fit ``limit``;
    otherwise a ``context_length_exceeded`` error in the wording OpenAI and vLLM use (clients such as opencode and
    Hermes Agent read the limit and the split from it and compact the conversation), plus how to raise the limit."""

    need = prompt + (int(asked) if asked else 1)
    if need <= limit:
        return None
    where = "messages" if chat else "prompt"
    if asked:
        what = (f"However, you requested {need} tokens ({prompt} in the {where}, {int(asked)} in the completion). "
                f"Please reduce the length of the {where} or completion.")
    else:
        what = f"However, your {where} resulted in {prompt} tokens. Please reduce the length of the {where}."
    return Problem(f"This model's maximum context length is {limit} tokens. {what} (This TensorFold server was "
                   f"started with --context {limit} (CONTEXT); restart both ranks with CONTEXT={need} or more to "
                   "serve it.)", code="context_length_exceeded", param=where)


def token_ids_problem(prompt: Any, vocab: int | None) -> str | None:
    """patches/0490: why a ``/v1/completions`` token-ID ``prompt`` (one flat list of ints, as vLLM takes it) is not
    usable, or None."""

    if not prompt:
        return "prompt must not be empty"
    if all(isinstance(p, list) for p in prompt) or all(isinstance(p, str) for p in prompt):
        return "one prompt a request: send a string or one list of token ids"
    if not all(isinstance(t, int) and not isinstance(t, bool) for t in prompt):
        return "a token-ID prompt must be a list of integers"
    bad = next((t for t in prompt if t < 0 or (vocab is not None and t >= vocab)), None)
    if bad is not None:
        return f"token id {bad} is outside the vocabulary (0 to {vocab - 1 if vocab else '?'})"
    return None


# -- patches/9001: logprobs -----------------------------------------------------------------------

LOGPROBS_MAX = 20


def logprobs_request(body: dict[str, Any], chat: bool) -> tuple[bool, int]:
    """(logprobs wanted, top alternatives 0..20) from a request; ValueError when malformed. Chat: ``logprobs`` bool
    plus ``top_logprobs``; completions: ``logprobs`` an integer (the alternatives), as OpenAI / vLLM take them."""

    lp = body.get("logprobs")
    if chat:
        if lp is None or lp is False:
            if body.get("top_logprobs"):
                raise ValueError("top_logprobs needs \"logprobs\": true")
            return False, 0
        if lp is not True:
            raise ValueError("logprobs must be true or false on /v1/chat/completions")
        top = body.get("top_logprobs")
        top = 0 if top is None else top
    else:
        if lp is None or lp is False:
            return False, 0
        top = 0 if lp is True else lp
    if isinstance(top, bool) or not isinstance(top, int) or not 0 <= top <= LOGPROBS_MAX:
        raise ValueError(f"top_logprobs must be an integer from 0 to {LOGPROBS_MAX} (got {top!r})")
    return True, int(top)


def _byte_decoder() -> dict[str, int]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\xa1"), ord("\xac") + 1)) \
        + list(range(ord("\xae"), ord("\xff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


_BYTES = _byte_decoder()


def token_bytes(tok, tid: int) -> list[int]:
    """The raw bytes of one token id (byte-level BPE pieces mapped back; added/special tokens as UTF-8)."""

    piece = tok.id_to_token(int(tid))
    if piece is None:
        return []
    if all(ch in _BYTES for ch in piece):
        return [_BYTES[ch] for ch in piece]
    return list(piece.encode("utf-8"))


def lp_entry(tok, tid: int, lp: float, tops: list[tuple[int, float]], top: int) -> dict[str, Any]:
    def one(t: int, v: float) -> dict[str, Any]:
        b = token_bytes(tok, t)
        return {"token": bytes(b).decode("utf-8", errors="replace"), "logprob": float(v), "bytes": b}

    e = one(tid, lp)
    e["top_logprobs"] = [one(t, v) for t, v in tops[:top]]
    return e


def legacy_logprobs(entries: list[dict[str, Any]]) -> dict[str, Any]:
    """``/v1/completions``' logprobs object from chat-style entries."""

    offs, at = [], 0
    for e in entries:
        offs.append(at)
        at += len(e["token"])
    return {"tokens": [e["token"] for e in entries], "token_logprobs": [e["logprob"] for e in entries],
            "top_logprobs": [{t["token"]: t["logprob"] for t in e["top_logprobs"]} for e in entries],
            "text_offset": offs}


# -- chat template -------------------------------------------------------------------------

class ChatTemplate:
    """The model's own Jinja chat template, rendered the way Hugging Face's apply_chat_template does."""

    def __init__(self, model_dir: Path):
        import jinja2
        import jinja2.ext
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        cfg = json.loads((model_dir / "tokenizer_config.json").read_text())
        source_path = model_dir / "chat_template.jinja"
        source = source_path.read_text() if source_path.exists() else cfg["chat_template"]

        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        def raise_exception(message):
            raise jinja2.exceptions.TemplateError(message)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=[jinja2.ext.loopcontrols])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        self.template = env.from_string(source)
        self.specials = {k: (v.get("content") if isinstance(v, dict) else v)
                         for k, v in cfg.items() if k in ("bos_token", "eos_token", "pad_token", "unk_token")}

    def render(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None,
               enable_thinking: bool, extra: dict[str, Any] | None = None) -> str:
        for m in messages:                       # tool_call arguments arrive as JSON strings
            for call in m.get("tool_calls") or []:
                fn = call.get("function") or {}
                if isinstance(fn.get("arguments"), str):
                    try:
                        fn["arguments"] = json.loads(fn["arguments"])
                    except json.JSONDecodeError:
                        pass
        kwargs = dict(self.specials, messages=messages, tools=tools or None, add_generation_prompt=True,
                      enable_thinking=enable_thinking)
        kwargs.update(extra or {})
        return self.template.render(**kwargs)


# -- HTTP ------------------------------------------------------------------------------------

class App:
    """One engine behind the OpenAI routes. ``sampling``: temperature, top_k and top_p for requests that do not
    set them (the model's generation config and the CLI's flags); ``max_tokens``: the reply length likewise."""

    def __init__(self, engine, model_dir: Path, served: str, *, default_thinking: bool = False,
                 sampling: dict[str, Any] | None = None, max_tokens: int = 4096):
        from tokenizers import Tokenizer

        from .health import Health       # patches/0150: /health that can say no, /metrics

        self.health = Health()
        self.engine = self.health.track(engine)     # patches/0150: generate() through the request bookkeeping
        self.served = served
        self.tok = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.template = ChatTemplate(model_dir)
        self.default_thinking = default_thinking
        self.sampling = {"temperature": 1.0, "top_k": 20, "top_p": 0.95, **(sampling or {})}
        self.max_tokens = int(max_tokens)
        self.lock = threading.Lock()
        self.created = int(time.time())             # patches/0490: /v1/models' ``created``

    def check(self, body: dict[str, Any]) -> str | None:
        """Why the request cannot run, or None. Checked before a stream's headers are sent."""

        import inspect

        if body.get("draft", True) is False and "draft" not in inspect.signature(self.engine.generate).parameters:
            return "this model's CUDA engine has no serial switch (\"draft\": false)"
        if not isinstance(body.get("messages", []), list):
            return "messages must be a list"
        try:                                        # patches/0160
            parse_stop(body.get("stop"))
        except ValueError as exc:
            return str(exc)
        if isinstance(body.get("prompt"), list):    # patches/0490: a token-ID prompt (/v1/completions)
            problem = token_ids_problem(body["prompt"], self.vocab_size())
            if problem:
                return Problem(problem, param="prompt")
        try:                                        # patches/9001
            logprobs_request(body, "messages" in body)
        except ValueError as exc:
            return Problem(str(exc), param="logprobs")
        mode = body.get("tensorfold_stats")         # patches/9004
        if mode is not None and mode not in STATS_MODES:
            return Problem(f"tensorfold_stats must be one of {', '.join(STATS_MODES)} (got {mode!r})",
                           param="tensorfold_stats")
        n = body.get("n")
        if n is not None and n != 1:
            return f"n must be 1 on this server (got {n!r}): it returns one choice a request"
        if body.get("tf_knobs") is not None:        # patches/0090: per-request engine knobs, validated up front
            parse = getattr(self.engine, "parse_knobs", None)
            if parse is None:
                return "this model's CUDA engine has no per-request knobs (tf_knobs)"
            try:
                parse(body["tf_knobs"])
            except ValueError as exc:
                return str(exc)
        return None

    def prompt_ids(self, text: str) -> list[int]:
        """The rendered prompt's token ids (patches/0210: a family's app may reuse ones it already has)."""

        return self.tok.encode(text, add_special_tokens=False).ids

    def given_ids(self, ids: list[int]) -> list[int]:
        """patches/0490: a request's own token ids (``/v1/completions`` with a token-ID ``prompt``)."""

        return [int(t) for t in ids]

    def vocab_size(self) -> int | None:
        """patches/0490: the tokenizer's vocabulary with added tokens (None when the tokenizer cannot say)."""

        size = getattr(self.tok, "get_vocab_size", None)
        return int(size(with_added_tokens=True)) if size is not None else None

    def context_limit(self) -> int | None:
        """patches/0490: the most prompt + reply tokens a request may use (the engine's ``limit``: ``--context``),
        or None when the engine does not say."""

        limit = getattr(self.engine, "limit", None)
        return int(limit) if isinstance(limit, int) and limit > 0 else None

    def tokenize(self, body: dict[str, Any]) -> list[int]:
        """patches/0490: ``POST /tokenize`` (vLLM's shape): a string ``prompt`` encoded as the tokenizer does it
        (``add_special_tokens``, default true as in vLLM), or ``messages`` rendered with the chat template as a chat
        request renders them. Raises ValueError for anything else."""

        if isinstance(body.get("messages"), list):
            kwargs = dict(body.get("chat_template_kwargs") or {})
            thinking = bool(kwargs.pop("enable_thinking", self.default_thinking))
            text = self.template.render(body["messages"], tools=body.get("tools") or [], enable_thinking=thinking,
                                        extra=kwargs)
            return list(self.tok.encode(text, add_special_tokens=False).ids)
        prompt = body.get("prompt")
        if not isinstance(prompt, str):
            raise ValueError("tokenize needs a string prompt or a list of messages")
        return list(self.tok.encode(prompt, add_special_tokens=bool(body.get("add_special_tokens", True))).ids)

    def sampling_for(self, body: dict[str, Any], prompt: list[int]):
        """Keyed sampling (the seed, else one drawn from the prompt), or None for greedy decoding."""

        from tensorfold.engine.exact_sampling import Sampling, seed_for

        temp = float(body["temperature"] if body.get("temperature") is not None else self.sampling["temperature"])
        if temp <= 0:
            return None
        seed = body.get("seed")
        top_k = body["top_k"] if body.get("top_k") is not None else self.sampling["top_k"]
        top_p = body["top_p"] if body.get("top_p") is not None else self.sampling["top_p"]
        return Sampling(int(seed) if seed is not None else seed_for(prompt), temp, int(top_k), float(top_p))

    def run(self, body: dict[str, Any], chat: bool, emit: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
        tools = body.get("tools") or []
        kwargs = dict(body.get("chat_template_kwargs") or {})
        thinking = bool(kwargs.pop("enable_thinking", self.default_thinking))
        if chat:
            text = self.template.render(body["messages"], tools=tools, enable_thinking=thinking, extra=kwargs)
        else:
            text = body["prompt"]
        if not chat and isinstance(text, list):    # patches/0490: token ids, used as they are
            prompt = self.given_ids(text)
        else:
            prompt = self.prompt_ids(text)         # patches/0210
        max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or self.max_tokens)
        sampling = self.sampling_for(body, prompt)
        out: list[int] = []
        sent = {"reasoning": 0, "content": 0}
        stopped = {"client": False}
        # patches/0160: OpenAI ``stop`` strings end the visible answer (not the reasoning, not a tool call's text):
        # the reply is cut before the first one and nothing after it is sent. ``at``: the tokens received when it
        # was seen (the reply's reported length)
        stops = parse_stop(body.get("stop"))
        hit = {"stop": False, "at": 0}
        stream = StreamDecoder(self.tok, tuple(self.engine.eos))
        # patches/9001: logprobs of the ANSWER tokens (after </think> when thinking; never the reasoning, never EOS)
        want_lp, top_lp = logprobs_request(body, chat)
        lp_sink: list = []
        lp_state = {"answer_from": None if (chat and thinking) else 0, "sent": 0}
        think_end = self.tok.token_to_id(_THINK_END) if chat and thinking else None
        eos_set = set(self.engine.eos)
        req = getattr(self.engine, "request", None)
        if req is not None:
            req.logprobs = (top_lp if top_lp > 0 else 1) if want_lp else 0
            req.lp_sink = lp_sink if want_lp else None

        def answer_lps(upto: int) -> list[dict[str, Any]]:
            """Entries for answer tokens out[lp_state.sent:upto] (advances ``sent``)."""
            start = lp_state["answer_from"]
            if start is None:
                return []
            lo = max(lp_state["sent"], start)
            hi = min(upto, len(lp_sink))
            got = []
            for i in range(lo, hi):
                if out[i] in eos_set:
                    continue
                if lp_sink[i] is None:
                    raise RuntimeError(f"logprobs: token {i} of the reply has no logprob row")
                lp, tops = lp_sink[i]
                got.append(lp_entry(self.tok, out[i], lp, tops, top_lp))
            lp_state["sent"] = max(lp_state["sent"], hi)
            return got

        def visible(finished: bool) -> tuple[str, str]:
            raw = stream.final() if finished else stream.text
            if chat and thinking:
                reasoning, answer = split_thinking(raw, finished=finished)
            else:
                reasoning, answer = "", raw
            if tools:
                answer = hide_tool_calls(answer, finished=finished)
            return reasoning, answer

        def on_tokens(new: list[int]) -> bool:
            base = len(out)
            out.extend(new)
            if want_lp and lp_state["answer_from"] is None and think_end is not None and think_end in new:
                lp_state["answer_from"] = base + new.index(think_end) + 1      # patches/9001
            if hit["stop"]:                         # patches/0160: past a stop string: nothing more is sent
                return True
            stream.add(new)
            reasoning, answer = visible(False)
            if stops:                               # patches/0160
                cut = find_stop(answer, stops, max(0, sent["content"] - max(map(len, stops)) + 1))
                if cut >= 0:
                    answer = answer[:cut]
                    hit.update(stop=True, at=len(out))
                else:
                    answer = answer[: len(answer) - stop_holdback(answer, stops)]
            delta: dict[str, Any] = {}
            if len(reasoning) > sent["reasoning"]:
                delta.update(reasoning_fields(reasoning[sent["reasoning"]:]))
                sent["reasoning"] = len(reasoning)
            if len(answer) > sent["content"]:
                delta["content"] = answer[sent["content"]:]
                sent["content"] = len(answer)
            if want_lp and "content" in delta:      # patches/9001: the answer tokens received so far
                delta["_logprobs"] = answer_lps(hit["at"] if hit["stop"] else len(out))
            if delta and not emit(delta):
                stopped["client"] = True
            # patches/0160: True after a stop string too; the batch engine (GLM53_TF_BATCH) then ends the request
            # in its next round (a cancel both ranks share), a single-request engine decodes on unheard
            return stopped["client"] or hit["stop"]

        draft = body.get("draft", True) is not False
        with self.lock:
            stats = self.engine.generate(prompt, max_tokens, sampling, on_tokens,
                                         **({} if draft else {"draft": False}))
        reasoning, answer = visible(True)
        final: dict[str, Any] = {}
        if len(reasoning) > sent["reasoning"]:
            final.update(reasoning_fields(reasoning[sent["reasoning"]:]))
        raw_answer = split_thinking(self.tok.decode([t for t in out if t not in self.engine.eos],
                                                    skip_special_tokens=False), finished=True)[1] \
            if chat and thinking else self.tok.decode([t for t in out if t not in self.engine.eos],
                                                      skip_special_tokens=False)
        stopped_at = -1
        if stops:                                   # patches/0160: cut the answer (and the raw text) at a stop
            shown, spans = _visible_spans(raw_answer, finished=True) if tools else (raw_answer, [(0, 0)])
            stopped_at = find_stop(shown, stops)
            if stopped_at >= 0:
                raw_answer = raw_answer[:_raw_index(spans, stopped_at)]
                cut = find_stop(answer, stops)
                answer = answer[:cut] if cut >= 0 else answer
        content, calls = parse_tool_calls(raw_answer, tools) if tools else (answer, None)
        tail = content[sent["content"]:] if content.startswith(answer[:sent["content"]]) else ""
        if tail:
            final["content"] = tail
        if calls:
            finish = "tool_calls"
        elif stopped_at >= 0 or (out and out[-1] in self.engine.eos):
            finish = "stop"
        else:
            finish = "length"
        used = hit["at"] if stopped_at >= 0 and hit["stop"] else len(out)
        result = {"final": final, "calls": calls, "finish": finish, "content": content, "reasoning": reasoning,
                  "prompt_tokens": len(prompt), "completion_tokens": used, "stats": stats}
        if want_lp:                                 # patches/9001
            if req is not None:
                req.logprobs, req.lp_sink = 0, None
            if len(lp_sink) < len(out):
                raise RuntimeError(f"logprobs: {len(out)} tokens, {len(lp_sink)} logprob rows (engine without "
                                   "logprobs support?)")
            tail = answer_lps(used)
            if tail and final:
                final["_logprobs"] = tail
            elif tail:
                result["lp_tail"] = tail
            everything = dict(lp_state, sent=0)
            lp_state.update(everything)
            result["logprobs"] = answer_lps(used)
        return result


def make_handler(app: App):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # quiet
            pass

        def _json(self, code: int, payload: dict[str, Any]) -> None:
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                entry: dict[str, Any] = {"id": app.served, "object": "model", "owned_by": "tensorfold"}
                limit = app.context_limit()                 # patches/0490: as vLLM (max_model_len) and others say it
                if limit is not None:
                    entry.update(created=getattr(app, "created", 0), root=app.served, max_model_len=limit, context_length=limit)
                self._json(200, {"object": "list", "data": [entry]})
            elif self.path.rstrip("/") in ("/health", "/v1/health"):
                self._json(*app.health.status())            # patches/0150
            elif self.path.rstrip("/") == "/metrics":       # patches/0150: Prometheus text
                data = app.health.metrics(app.served).encode()
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; version=0.0.4")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            path = self.path.rstrip("/")
            chat = path.endswith("/chat/completions")
            tokenize = path in ("/tokenize", "/v1/tokenize")        # patches/0490
            if not chat and not tokenize and not path.endswith("/completions"):
                return self._json(404, {"error": "not found"})
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            except json.JSONDecodeError:
                return self._json(400, error_body("the request body is not JSON"))
            if not isinstance(body, dict):
                return self._json(400, error_body("the request body must be a JSON object"))
            if tokenize:                        # patches/0490: host only (no engine, no lock, nothing to rank 1)
                try:
                    ids = app.tokenize(body)
                except ValueError as exc:
                    return self._json(400, error_body(str(exc)))
                return self._json(200, {"count": len(ids), "max_model_len": app.context_limit(), "tokens": ids,
                                        "token_strs": None})
            problem = app.check(body)
            if problem:
                return self._json(400, error_body(problem))     # patches/0490: with a Problem's code / param
            refused = app.health.reject()           # patches/0150: GLM53_TF_HEALTH=strict after a fatal error
            if refused:
                return self._json(503, {"error": {"message": refused, "type": "server_error"}})
            rid = f"chatcmpl-{uuid.uuid4().hex[:24]}" if chat else f"cmpl-{uuid.uuid4().hex[:24]}"
            created = int(time.time())
            stream = bool(body.get("stream"))
            kind = "chat.completion.chunk" if chat else "text_completion"

            def chunk(delta: dict[str, Any], finish: str | None = None) -> dict[str, Any]:
                delta = dict(delta)
                lp = delta.pop("_logprobs", None)          # patches/9001: this chunk's answer-token logprobs
                if chat:
                    choice = {"index": 0, "delta": delta, "finish_reason": finish}
                    if lp is not None:
                        choice["logprobs"] = {"content": lp}
                    return {"id": rid, "object": kind, "created": created, "model": app.served, "choices": [choice]}
                choice = {"index": 0, "text": delta.get("content", ""), "finish_reason": finish}
                if lp is not None:
                    choice["logprobs"] = legacy_logprobs(lp)
                return {"id": rid, "object": kind, "created": created, "model": app.served, "choices": [choice]}

            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()

                def emit(delta: dict[str, Any]) -> bool:
                    try:
                        self.wfile.write(f"data: {json.dumps(chunk(delta))}\n\n".encode())
                        self.wfile.flush()
                        return True
                    except (BrokenPipeError, ConnectionResetError):
                        return False

                if chat:
                    emit({"role": "assistant"})
                try:
                    result = app.run(body, chat, emit)
                except Exception as exc:        # patches/0150: an error event, not a dropped connection
                    try:
                        # patches/0490: a request the engine refused (ValueError) is the client's error, as in the
                        # non-streamed response
                        err = error_body(str(exc)[:500]) if isinstance(exc, ValueError) else \
                            {"error": {"message": f"{type(exc).__name__}: {exc}"[:500], "type": "server_error"}}
                        self.wfile.write(f"data: {json.dumps(err)}\n\ndata: [DONE]\n\n".encode())
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    self.close_connection = True
                    return
                if result["final"]:
                    emit(result["final"])
                elif result.get("lp_tail"):                # patches/9001: answer logprobs not sent with a delta yet
                    emit({"_logprobs": result["lp_tail"]})
                if result["calls"]:
                    for i, call in enumerate(result["calls"]):
                        emit({"tool_calls": [{"index": i, "id": call["id"], "type": "function",
                                              "function": {"name": call["function"]["name"],
                                                           "arguments": call["function"]["arguments"]}}]})
                end = chunk({}, result["finish"])
                usage = {"prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
                         "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
                         "prompt_tokens_details": _cached(result)}          # patches/0110
                if (body.get("stream_options") or {}).get("include_usage"):
                    end["usage"] = usage
                stats = response_stats(result["stats"], stats_mode(body))      # patches/0090, 9004
                if stats is not None:
                    end["tensorfold"] = stats
                try:
                    self.wfile.write(f"data: {json.dumps(end)}\n\ndata: [DONE]\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    pass
                self.close_connection = True
                return
            try:
                result = app.run(body, chat, lambda delta: True)
            except ValueError as exc:           # patches/0150: a request the engine refused
                return self._json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
            except Exception as exc:            # patches/0150: a JSON 500, not a dropped connection
                return self._json(500, {"error": {"message": f"{type(exc).__name__}: {exc}"[:500],
                                                  "type": "server_error"}})
            usage = {"prompt_tokens": result["prompt_tokens"], "completion_tokens": result["completion_tokens"],
                     "total_tokens": result["prompt_tokens"] + result["completion_tokens"],
                     "prompt_tokens_details": _cached(result)}              # patches/0110
            if chat:
                message: dict[str, Any] = {"role": "assistant", "content": result["content"] or None}
                if result["reasoning"]:
                    message.update(reasoning_fields(result["reasoning"]))      # patches/0160
                if result["calls"]:
                    message["tool_calls"] = result["calls"]
                choice = {"index": 0, "message": message, "finish_reason": result["finish"]}
                if result.get("logprobs") is not None:      # patches/9001
                    choice["logprobs"] = {"content": result["logprobs"]}
                payload = {"id": rid, "object": "chat.completion", "created": created, "model": app.served,
                           "choices": [choice], "usage": usage}
            else:
                choice = {"index": 0, "text": result["content"], "finish_reason": result["finish"]}
                if result.get("logprobs") is not None:      # patches/9001
                    choice["logprobs"] = legacy_logprobs(result["logprobs"])
                payload = {"id": rid, "object": "text_completion", "created": created, "model": app.served,
                           "choices": [choice], "usage": usage}
            stats = response_stats(result["stats"], stats_mode(body))      # patches/0090, 9004
            if stats is not None:
                payload["tensorfold"] = stats
            self._json(200, payload)

    return Handler


def _cached(result: dict[str, Any]) -> dict[str, int]:
    """patches/0110: OpenAI's ``usage.prompt_tokens_details``: the prompt tokens the engine resumed from a kept
    state instead of prefilling (its stats' ``cached``)."""

    return {"cached_tokens": int((result.get("stats") or {}).get("cached", 0) or 0)}


def serve(app: App, host: str, port: int) -> None:
    """Serve until interrupted (SIGTERM included)."""

    import signal
    from http.server import ThreadingHTTPServer

    def _terminate(signum: int, frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, _terminate)
    server = ThreadingHTTPServer((host, port), make_handler(app))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
