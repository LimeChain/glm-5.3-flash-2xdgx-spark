#!/usr/bin/env python3
"""Tool-call correctness harness for an OpenAI-compatible GLM-5.3-Flash endpoint.

Sends opencode-shaped tool calls and validates that every returned tool call is
(a) a known tool name, (b) JSON-parseable arguments with the declared keys, and
(c) free of leaked GLM markup (`<arg_key>`, `<arg_value>`, `<tool_call>`,
`</think>`) inside any string value. Emits a JSON report.

Usage:
  GLM_URL=http://127.0.0.1:8080/v1/chat/completions GLM_MODEL=<served name> \
      python3 bench/toolcall_harness.py --reps 3 --out results/toolcalls.json

GLM_API_KEY (optional) is sent as a Bearer token; GLM_EXTRA (JSON) is merged into every request.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field

import requests

DEFAULT_URL = os.environ.get("GLM_URL", "http://127.0.0.1:8080/v1/chat/completions")
AUTH = {"Authorization": f"Bearer {os.environ['GLM_API_KEY']}"} if os.environ.get("GLM_API_KEY") else {}
MODEL = os.environ.get("GLM_MODEL", "glm-5.3-flash")

LEAK_RE = re.compile(r"</?arg_(key|value)>|</?tool_call>|</think>|<think>")

SYSTEM = (
    "You are a helpful coding agent operating in a terminal. Use the provided tools to complete "
    "the user's request. Prefer making the tool call directly rather than explaining what you "
    "would do. Keep reasoning brief."
)


def tool(name, description, props, required):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


BASH = tool("bash", "Run a shell command.", {
    "command": {"type": "string"},
    "description": {"type": "string"},
    "workdir": {"type": "string"},
    "timeout": {"type": "integer"},
}, ["command"])

READ = tool("read", "Read a file.", {
    "filePath": {"type": "string"},
    "offset": {"type": "integer"},
    "limit": {"type": "integer"},
}, ["filePath"])

EDIT = tool("edit", "Replace exact text in a file.", {
    "filePath": {"type": "string"},
    "oldString": {"type": "string"},
    "newString": {"type": "string"},
    "replaceAll": {"type": "boolean"},
}, ["filePath", "oldString", "newString"])

GLOB = tool("glob", "Find files by glob.", {"pattern": {"type": "string"}, "path": {"type": "string"}}, ["pattern"])

GREP = tool("grep", "Search file contents.", {
    "pattern": {"type": "string"},
    "path": {"type": "string"},
    "include": {"type": "string"},
}, ["pattern"])

WRITE = tool("write", "Write a file.", {"filePath": {"type": "string"}, "content": {"type": "string"}}, ["filePath", "content"])

TASK = tool("task", "Launch a subagent.", {
    "description": {"type": "string"},
    "prompt": {"type": "string"},
    "subagent_type": {"type": "string", "enum": ["explore", "general"]},
}, ["description", "prompt", "subagent_type"])

TODOWRITE = tool("todowrite", "Write the todo list.", {
    "todos": {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "in_progress", "completed", "cancelled"]},
                "priority": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": ["content", "status", "priority"],
        },
    }
}, ["todos"])

WEBFETCH = tool("webfetch", "Fetch a URL.", {"url": {"type": "string"}, "format": {"type": "string"}}, ["url"])
PATCH = tool("patch", "Apply a patch.", {"patchText": {"type": "string"}}, ["patchText"])
SKILL = tool("skill", "Load a skill.", {"name": {"type": "string"}}, ["name"])
QUESTION = tool("question", "Ask the user a question.", {
    "questions": {"type": "array", "items": {"type": "object", "properties": {
        "question": {"type": "string"}, "header": {"type": "string"}}}}}, ["questions"])
INVALID = tool("invalid", "Report an invalid tool call.", {"tool": {"type": "string"}}, [])
T3_DEVICE_LIST = tool("t3-code_device_list", "List devices.", {"hostId": {"type": "string"}}, [])
T3_DEVICE_OPEN = tool("t3-code_device_open", "Open a device.", {
    "deviceId": {"type": "string"}, "platform": {"type": "string", "enum": ["ios", "android"]}}, [])
T3_LINK_PR = tool("t3-code_link_pull_request", "Link a PR.", {"url": {"type": "string"}}, [])

ALL_TOOLS = [BASH, READ, EDIT, GLOB, GREP, WRITE, TASK, TODOWRITE]
# opencode's real tool surface is larger; some corruption may be schema-set dependent.
OPENCODE_TOOLS = ALL_TOOLS + [WEBFETCH, PATCH, SKILL, QUESTION, INVALID, T3_DEVICE_LIST,
                              T3_DEVICE_OPEN, T3_LINK_PR]
KNOWN = {t["function"]["name"] for t in OPENCODE_TOOLS}


def _assistant_call(call_id: str, name: str, args: dict):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": call_id, "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)}}],
    }


def _tool_result(call_id: str, content: str):
    return {"role": "tool", "tool_call_id": call_id, "content": content}


@dataclass
class Case:
    name: str
    prompt: str
    tools: list = field(default_factory=lambda: ALL_TOOLS)
    expect: list[str] = field(default_factory=list)  # expected tool names (any order)
    setup: list = field(default_factory=list)  # prior messages (e.g. tool results)

    def messages(self):
        msgs = [{"role": "system", "content": SYSTEM}]
        msgs.extend(self.setup)
        msgs.append({"role": "user", "content": self.prompt})
        return msgs


CASES = [
    Case("bash_simple", "List the files in the current directory using bash.", expect=["bash"]),
    Case("bash_4arg",
         "Use bash to list /home/user/projects, set workdir to /tmp, timeout 10000, "
         "and description 'list projects'.", expect=["bash"]),
    Case("read_offset", "Read lines 100-150 of src/main.ts using the read tool.", expect=["read"]),
    Case("edit_file",
         "In src/app.ts replace the line `const port = 3000;` with `const port = 8080;` using the edit tool.",
         expect=["edit"]),
    Case("glob_grep",
         "Find all .ts files under src/ with glob, then search them for 'TODO' with grep. Make both calls.",
         expect=["glob", "grep"]),
    Case("task_3arg",
         "Launch a task subagent with description 'Research discipline', subagent_type 'explore', "
         "prompt 'Read the repo structure and return the last 5 lines of README.md'.", expect=["task"]),
    Case("todowrite_list",
         "Create a todo list with three items: 'Inspect repo' (completed, high), "
         "'Write tests' (in_progress, high), 'Ship' (pending, medium).", expect=["todowrite"]),
    Case("parallel_two",
         "Do two things at once: read package.json, and glob for `**/*.ts`. Make both tool calls in one turn.",
         expect=["read", "glob"]),
    Case("nested_request",
         "Use the task tool to spawn an explore subagent whose prompt includes the shell command "
         "`cd /home/user/projects && ls packages` and asks it to summarize. "
         "description 'Explore packages', subagent_type 'explore'.", expect=["task"]),
    Case("write_file",
         "Write a file notes.md containing the heading '# Notes' and one line 'hello'.", expect=["write"]),
    Case("multi_tool_four",
         "Perform all of: read README.md, glob `**/*.json`, grep for 'TODO', and run `git status`. "
         "Return all four tool calls.", expect=["read", "glob", "grep", "bash"]),
    Case("reasoning_then_call",
         "Think step by step about why a port might be in use, then run `ss -ltnp` with bash.",
         expect=["bash"]),

    # --- harder, conversation-shaped cases (where the live corruption appeared) ---

    Case("edit_with_content",
         "The current contents of src/app.ts are shown in the previous tool result. "
         "Change the port from 3000 to 8080 with the edit tool.",
         expect=["edit"],
         setup=[_assistant_call("call_ec", "read", {"filePath": "src/app.ts"}),
                _tool_result("call_ec", 'const port = 3000;\napp.listen(port);\n')],
         ),

    Case("after_tool_result",
         "The test run finished. Based on the tool result, make the next tool call to fix the failure.",
         expect=["bash", "edit", "read"],
         setup=[_assistant_call("call_at", "bash", {"command": "pytest -q"}),
                _tool_result("call_at", "FAIL tests/test_app.py::test_health\nAssertionError: 500 != 200\n")],
         ),

    Case("result_contains_markup",
         "A file on disk literally contains GLM tool-call markup. Read it back and then continue: "
         "the next step is to grep the repo for that marker.",
         expect=["grep", "read", "bash"],
         setup=[_assistant_call("call_rm", "bash", {"command": "cat marker.txt"}),
                _tool_result("call_rm", "<tool_call>bash<arg_key>command</arg_key>"
                                        "<arg_value>ls -la</arg_value></tool_call>\n")],
         ),

    Case("multi_turn_chain",
         "Continue the task: create the missing directory and write an empty index.ts in it.",
         expect=["bash", "write"],
         setup=[_assistant_call("call_mt", "bash", {"command": "ls -R"}),
                _tool_result("call_mt", "src/\nsrc/app.ts\n")],
         ),

    Case("three_calls_complex",
         "Do all of the following in one turn: 1) read tsconfig.json, 2) grep for 'strict' under src, "
         "3) run `pnpm build`. Return three tool calls.",
         expect=["read", "grep", "bash"],
         ),

    # --- self-reinforcing corruption loop (matches the live failure) ---

    Case("poisoned_tool_error",
         "Continue the task. Make the tool call you were trying to make.",
         expect=["bash", "read", "glob", "grep"],
         setup=[_assistant_call("call_p1", "bash", {"command": "ls packages"}),
                _tool_result("call_p1", "Model tried to call unavailable tool 'invalid</arg_value>'. "
                                        "Available tools: bash, edit, glob, grep, read, task, todowrite, write."),
                ],
         ),

    Case("poisoned_assistant_text",
         "Continue.",
         expect=["bash", "read", "glob", "grep"],
         setup=[{"role": "assistant", "content":
                 "I'll look at the repo.</think><tool_call>bash<arg_key>command</arg_key>"
                 "<arg_value>ls -la</arg_value></tool_call>"},
                _tool_result("call_p2", "total 12\ndrwxr-xr-x 4 user user 4096 src\n"),
                ],
         ),

    Case("long_reasoning_tool",
         "Think step by step, at length, about the tradeoffs of microservices versus a monolith "
         "(at least 15 distinct points). Then use the read tool to read README.md.",
         expect=["read", "bash", "glob"],
         ),

    Case("markup_in_result_repeat",
         "The previous tool result contained GLM tool-call markup. Continue by grepping the repo "
         "for 'tool_call'.",
         expect=["grep", "bash", "read"],
         setup=[_assistant_call("call_p3", "bash", {"command": "cat .opencode/note.txt"}),
                _tool_result("call_p3", "note:\n<tool_call>bash<arg_key>command</arg_key>"
                                        "<arg_value>rm -rf /tmp/x</arg_value></tool_call>\n"),
                ],
         ),
]


CORRUPTION_KINDS = ("unknown_tool", "bad_json", "leak_in_arg", "leaked_markup")


def validate(resp: dict, case: Case):
    """Return (ok, problems, detail). detail['corruption'] flags format corruption."""
    problems = []
    corruption = False
    choice = resp["choices"][0]
    msg = choice.get("message", {})
    calls = msg.get("tool_calls") or []
    detail = {"finish_reason": choice.get("finish_reason"), "n_calls": len(calls), "calls": []}

    if not calls:
        content = msg.get("content") or ""
        if LEAK_RE.search(content):
            problems.append("leaked_markup")
            corruption = True
        else:
            problems.append("no_tool_calls")
        detail["content_preview"] = content[:300]
        detail["corruption"] = corruption
        return False, problems, detail

    for c in calls:
        fn = c.get("function", {})
        name = fn.get("name", "")
        args_raw = fn.get("arguments", "")
        rec = {"name": name, "raw": args_raw[:400]}
        if name not in KNOWN:
            problems.append(f"unknown_tool:{name!r}")
            corruption = True
        try:
            parsed = json.loads(args_raw)
        except Exception as e:
            problems.append(f"bad_json:{name}:{e}")
            corruption = True
            parsed = None
        if parsed is not None:
            rec["keys"] = sorted(parsed)
            for k, v in parsed.items():
                if isinstance(v, str) and LEAK_RE.search(v):
                    problems.append(f"leak_in_arg:{name}.{k}")
                    corruption = True
            # required keys
            spec = next((t for t in case.tools if t["function"]["name"] == name), None)
            if spec:
                req = spec["function"]["parameters"].get("required", [])
                missing = [r for r in req if r not in parsed]
                if missing:
                    problems.append(f"missing_keys:{name}:{missing}")
        detail["calls"].append(rec)

    if case.expect:
        got = {c["function"]["name"] for c in calls}
        if not (got & set(case.expect)):
            problems.append(f"missing_expected:any_of({case.expect})")

    detail["corruption"] = corruption
    return (not problems), problems, detail


def _consume_stream(r):
    """Reassemble an OpenAI streaming response into a non-streaming-shaped dict."""
    calls: dict[int, dict] = {}
    content_parts = []
    finish = None
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except Exception:
            continue
        choice = (chunk.get("choices") or [{}])[0]
        if choice.get("finish_reason"):
            finish = choice["finish_reason"]
        delta = choice.get("delta") or {}
        if delta.get("content"):
            content_parts.append(delta["content"])
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            rec = calls.setdefault(idx, {"id": "", "function": {"name": "", "arguments": ""}})
            if tc.get("id"):
                rec["id"] = tc["id"]
            fn = tc.get("function") or {}
            if fn.get("name"):
                rec["function"]["name"] += fn["name"]
            if fn.get("arguments"):
                rec["function"]["arguments"] += fn["arguments"]
    ordered = [calls[k] for k in sorted(calls)]
    return {"choices": [{"finish_reason": finish,
                         "message": {"content": "".join(content_parts), "tool_calls": ordered}}]}


def run_case(case: Case, url: str, temperature: float, max_tokens: int, timeout: float,
             stream: bool = False, tools=None):
    payload = {
        "model": MODEL,
        "max_tokens": max_tokens,
        "tools": tools or case.tools,
        "messages": case.messages(),
    }
    if temperature is not None and temperature >= 0:
        payload["temperature"] = temperature
    if stream:
        payload["stream"] = True
    payload.update(json.loads(os.environ.get("GLM_EXTRA", "{}")))   # glm53-tensorfold-spark: e.g. {"tf_knobs": {...}}
    t0 = time.time()
    try:
        if stream:
            r = requests.post(url, json=payload, headers=AUTH,
                              timeout=timeout, stream=True)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
            resp = _consume_stream(r)
        else:
            r = requests.post(url, json=payload, headers=AUTH, timeout=timeout)
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
            resp = r.json()
        dt = time.time() - t0
    except Exception as e:
        return {"case": case.name, "ok": False, "problems": [f"request_error:{e}"], "elapsed_s": round(time.time() - t0, 1)}
    if tools:
        case = Case(case.name, case.prompt, tools=tools, expect=case.expect, setup=case.setup)
    ok, problems, detail = validate(resp, case)
    return {"case": case.name, "ok": ok, "problems": problems, "elapsed_s": round(dt, 1), **detail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=DEFAULT_URL)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="sampling temperature; <0 omits it so the server default applies")
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", default=None, help="comma-separated case names")
    ap.add_argument("--stream", action="store_true", help="use SSE streaming (like opencode)")
    ap.add_argument("--toolset", choices=["basic", "opencode"], default="basic",
                    help="tool surface to advertise")
    args = ap.parse_args()

    cases = [c for c in CASES if not args.only or c.name in args.only.split(",")]
    toolset = OPENCODE_TOOLS if args.toolset == "opencode" else None
    results = []
    for rep in range(args.reps):
        for case in cases:
            res = run_case(case, args.url, args.temperature, args.max_tokens, args.timeout,
                           args.stream, tools=toolset)
            res["rep"] = rep
            results.append(res)
            mark = "OK  " if res["ok"] else "FAIL"
            print(f"[{rep}] {mark} {res['case']:20s} {res.get('elapsed_s')}s "
                  f"{('| ' + '; '.join(res['problems'])) if res['problems'] else ''}", flush=True)

    n = len(results)
    passed = sum(1 for r in results if r["ok"])
    corrupt = sum(1 for r in results if r.get("corruption"))
    print(f"\ntemperature={args.temperature} {args.reps} reps: {passed}/{n} clean "
          f"({100*passed/max(n,1):.1f}%), corruption={corrupt}/{n} ({100*corrupt/max(n,1):.1f}%)")
    report = {"config": vars(args), "passed": passed, "total": n, "corruption": corrupt, "results": results}
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.out}")
    return 0 if passed == n else 1


if __name__ == "__main__":
    sys.exit(main())
