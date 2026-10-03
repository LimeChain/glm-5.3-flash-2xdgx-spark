#!/usr/bin/env python3
"""Dependency-free release checks for the public source tree."""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED = {
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "config/cluster.env.example",
    "config/versions.env",
    "container/Dockerfile",
    "container/flashinfer-glm53.patch",
    "overlay/sitecustomize.py",
    "scripts/build-image.sh",
    "scripts/preflight-tp2.sh",
    "scripts/rank-tp2.sh",
    "scripts/start-tp2.sh",
    "scripts/stop-tp2.sh",
    "scripts/verify-runtime-contract.py",
    "bench/benchmark.py",
    "results/production-tp2-mtp3-262k.json",
    "docs/vllm-path.md",
    "tensorfold/config/prod-c8.env.example",
    "tensorfold/scripts/tf-prod.sh",
    "tensorfold/grafana/tf_vllm_adapter.py",
    "tensorfold/patches/9001-glm-logprobs.patch",
    "tensorfold/patches/Dockerfile.logprobs",
    "mia-tensorfold/Dockerfile",
    "mia-tensorfold/lc-9001-9002.patch",
    "mia-tensorfold/scripts/glm-prod.sh",
}
missing = sorted(path for path in REQUIRED if not (ROOT / path).is_file())
assert not missing, f"missing required files: {missing}"
assert not (ROOT / "config/cluster.env").exists(), "live cluster config must not be published"

for path in ROOT.rglob("*"):
    if ".git" in path.parts:
        continue
    assert not path.is_symlink(), f"symlink is not allowed: {path.relative_to(ROOT)}"

versions = (ROOT / "config/versions.env").read_text()
assert "@sha256:905c02933be6021301db2dc284e24e3727467aa3a0f63b41d609885778a07bce" in versions
assert "BASE_VLLM_VERSION=0.1.dev20051+g487ecf187" in versions
assert "BASE_FLASHINFER_VERSION=0.6.17" in versions

patch = (ROOT / "container/flashinfer-glm53.patch").read_text()
for expected in (
    "DSV3_2_DISPATCH(32, 2176)",
    "DSV3_2_DISPATCH(64, 2176)",
    "ComputeMode::FP8, 32, 2176, 64",
    "ComputeMode::FP8, 64, 2176, 64",
):
    assert expected in patch, f"missing patch contract: {expected}"

overlay = (ROOT / "overlay/sitecustomize.py").read_text()
for expected in (
    "required_glm_dispatch = {(32, 2176), (64, 2176)}",
    "qk_rope_head_dim=64",
    "valid_counts.clamp(min=1)",
    "out.masked_fill_",
):
    assert expected in overlay, f"missing runtime-overlay contract: {expected}"

rank_launcher = (ROOT / "scripts/rank-tp2.sh").read_text()
assert (
    "--default-chat-template-kwargs "
    "'{\"enable_thinking\":true,\"reasoning_effort\":\"high\"}'"
    in rank_launcher
), "rank launcher must default to thinking enabled at High"

receipt = json.loads((ROOT / "results/production-tp2-mtp3-262k.json").read_text())
assert receipt["schema"] == "glm53-sm121-public-benchmark-summary.v1"
assert receipt["source_revision"] == "4c290638cb174721217c903e2dbf92b9858a080b"
expected_throughput = {
    "c1": 29.737457574057196,
    "c4": 68.07078743138442,
    "c6": 87.39052568246866,
    "c8": 101.73763877706034,
}
for scenario, expected in expected_throughput.items():
    actual = receipt["summary"][scenario][
        "matched_cold_run_mean_aggregate_decode_tokens_per_second"
    ]
    assert actual == expected, f"{scenario} throughput drift: {actual}"
assert receipt["validation"]["qualified_long_context_prompt_tokens"] == 140012
assert receipt["validation"]["c12_throughput_claimed"] is False

# The original vLLM recipe (and its historical receipt claims) moved to docs/vllm-path.md on 2026-09-30.
readme = (ROOT / "docs/vllm-path.md").read_text()
for expected in (
    "29.74 tok/s",
    "68.07 tok/s",
    "87.39 tok/s",
    "101.74 aggregate tok/s",
    "C4/C6/C8 are **aggregate throughput**",
    "no C12 throughput is claimed",
    "140,012-token prompt",
    "Thinking enabled by default",
    "enabled by default at High",
):
    assert expected in readme, f"README claim missing: {expected}"

top = (ROOT / "README.md").read_text()
for expected in (
    "Path A — Mia's TensorFold recipe",
    "Path B — jayleaton's kit",
    "Path C — vLLM + NVFP4",
    "GLM53_TF_BATCH_ADMIT_GB",
    "docs/vllm-path.md",
    "each ran **once**",
):
    assert expected in top, f"top-level README claim missing: {expected}"

profile = (ROOT / "tensorfold/config/prod-c8.env.example").read_text()
for expected in (
    "GLM53_TF_BATCH=8",
    "GLM53_TF_BATCH_ADMIT_GB=0.25",
    "GLM53_TF_MULTI_PREFILL=1",
    'WORKER_SSH="<worker-ssh>"',
    'HEAD_IP="<head-ip>"',
):
    assert expected in profile, f"TensorFold profile contract missing: {expected}"

# Reject environment-specific production identifiers without spelling them into
# this repository's own source as one contiguous scanner false positive.
forbidden = [
    "spark" + "-4651",
    "spark" + "-4b8e",
    "10.100" + ".40.",
    "/home/" + "chris/",
    "127.0.0.1:" + "28089",
]
text_files = [
    path
    for path in ROOT.rglob("*")
    if path.is_file() and ".git" not in path.parts and path != Path(__file__)
]
for path in text_files:
    try:
        text = path.read_text()
    except UnicodeDecodeError:
        continue
    for value in forbidden:
        assert value not in text, f"private identifier in {path.relative_to(ROOT)}"

# Verify local Markdown links point to files that exist in the candidate tree.
link_pattern = re.compile(r"\[[^\]]*\]\(([^)]+)\)")
for markdown in ROOT.rglob("*.md"):
    for target in link_pattern.findall(markdown.read_text()):
        if target.startswith(("http://", "https://", "#")):
            continue
        target_path = target.split("#", 1)[0]
        if not target_path:
            continue
        resolved = (markdown.parent / target_path).resolve()
        assert ROOT == resolved or ROOT in resolved.parents, f"link escapes root: {markdown}: {target}"
        assert resolved.exists(), f"broken link: {markdown.relative_to(ROOT)} -> {target}"

print(f"public repository contract verified: {len(text_files)} files inspected")
