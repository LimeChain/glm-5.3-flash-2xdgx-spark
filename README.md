# GLM-5.3 Flash on 2× NVIDIA DGX Spark

Three tested ways to serve **GLM-5.3 Flash** across two DGX Spark (GB10) systems linked over their ConnectX-7 ports, each behind an OpenAI-compatible API:

| Path | Engine | Weights | Use it when |
|---|---|---|---|
| **A — Mia's TensorFold recipe, 8 streams (recommended, LimeChain production since 2026-10-03)** | [`MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) @ `4bbf2f3` (TensorFold 0.6.0) + our image in [`mia-tensorfold/`](mia-tensorfold/): 8 streams instead of 4, OpenAI logprobs, stop at `<\|assistant\|>` | EXL3 4-bit TR3 + DFlash2 drafter | you want the fastest C8, logprobs for verifiers, and a 1M context |
| **B — jayleaton's TensorFold kit + C8 profile** | [`jayleaton/glm53-tensorfold-spark`](https://github.com/jayleaton/glm53-tensorfold-spark) + [`tensorfold/`](tensorfold/) | EXL3 4-bit TR3 + DFlash2 drafter | you want the most memory headroom (12-14 GiB floor at C8) or C12 |
| **C — vLLM + NVFP4** | this repository's original vLLM TP2 recipe ([`docs/vllm-path.md`](docs/vllm-path.md)) | NVFP4 | you need the full vLLM API surface (`n>1`, vLLM metrics and tooling) |

Measured with the same harness, prompts and hardware: the [2026-10-03 bake-off](#bake-off-2026-10-03) (A, B) and the [2026-09-30 bake-off](#bake-off-2026-09-30) (B, C, and the vLLM EXL3 kit).

> The repository was renamed from `glm-5.3-flash-2xdgx-spark-vllm` on 2026-09-30. Old URLs redirect.

---

## Bake-off 2026-10-03

Same pair of Sparks, same weights (`brandonmusic/GLM-5.3-Flash-tr3-4bpw` @ `5ab363a8`, DFlash2 @ `7d74cdd8`), same suite ([`bench/bakeoff_suite.py`](bench/bakeoff_suite.py), `CONC=` sets the concurrency of the C8 sections). One run per arm, so treat differences under ~5% as noise. Summaries: [`results/2026-10-03-bakeoff/`](results/2026-10-03-bakeoff/).

| Test (tok/s unless noted) | **A: Mia recipe, 8 streams + LimeChain** | Mia recipe as shipped (4 streams) | B: jayleaton kit b11 + LimeChain, C8 | B at C12 | B on `5624dfc` (2026-09-30) |
|---|---:|---:|---:|---:|---:|
| C1, short prompt | **114** | 116 | 97 | 101 | 99 |
| C1 prose / code, thinking off | 50 / **70** | 50 / 70 | 47 / 56 | | 45 / 61 |
| **Concurrent short prompts, aggregate (per stream)** | **528 (66)** at C8 | 300 (76) at C4 | 381 (48) | 374 (32) | 368 (46) |
| Concurrent reasoning, thinking High, aggregate | **133** | 102 (C4) | 94 | 110 | 93 |
| 8 × ~23K follow-up turn, aggregate | **92** | 59 (C4) | 65 | 69 (C12) | 46 |
| follow-up: prefix-cache hit / slowest first token | 98.7% / **2.0 s** | 98.7% / 1.4 s (C4) | 98.7% / 2.7 s | 98.7% / 3.5 s | 98.7% / 5 s |
| 8 × ~23K fresh prompts: aggregate / slowest first token | **14.5 / 134 s** | 13.7 / 67 s (C4) | 12.4 / 154 s | 13.3 / 219 s | 10.8 / 183 s |
| 128K+ single prefill | **1,729** | 1,711 | 1,484 | | 1,517 |
| Short request's first token during a 137-150K prefill | 1.8 s | 2.2 s | 1.4 s | **0.4 s** | 1.5 s |
| Lowest MemAvailable, head / worker (GiB) | 7.3 / 7.6 | 10.7 / 11.1 | 12.8 / 14.2 | 13.0 / 14.0 | 12.9 / 13.2 |
| Shared KV pool | **2.58M tokens** | 2.58M | 1.05M | 1.05M | 1.05M |

Correctness and agent gates on A (the B column passed the same set): known answers 12/12, Korean rare-token probe clean, count-to-200 3/3, tool-call harness 40/42 with 0 corrupted (at default effort High), 42K-token tool-call nondeterminism repro 0/16 diverging, a 111K-token verifier request with `top_logprobs: 20`, effort High and `max_tokens: 65536` streamed and non-streamed, and a fake-turn probe (`Output exactly: <score_A> B </score_A>`, 6 seeds) clean. A worker-rank kill was recovered by the watchdog in about 6.5 minutes, with logprobs working afterwards.

**What A changes on Mia's recipe:**
- **8 streams.** Mia's engine allows 4 (`PARALLEL_MOST` and the segmented kernels' `MAX_SEGS`, a 32-row batched verify window). Raising those to 8 streams and 64 rows ran every section above with no errors. C8 short-prompt throughput rose from 300 (C4) to 528 tok/s; per-stream speed fell from 76 to 66.
- **OpenAI logprobs on the multi-stream engine** ([`lc-9001-9002.patch`](mia-tensorfold/lc-9001-9002.patch)). TensorFold 0.6.x refuses them for GLM; with the patch, `logprobs` / `top_logprobs` (≤ 20) cover the answer tokens, streamed or not, with thinking and tools: raw natural-log values over the whole vocabulary, combined exactly across both ranks. Same generated text with or without them.
- **Stop at `<|assistant|>`** (`TF_GLM_EXTRA_STOP`). On short exact-format replies GLM ends its turn with `<|assistant|>` and the engine wrote a fake next turn into the answer: 5 of 6 seeds before, 0 of 6 after.
- **Effort High by default** (`--reasoning-effort high`). Mia's default is Max; at Max the tool harness passed 36/42 (two multi-step cases answered with a read instead of acting), at High 40/42.

**Not adopted:** C12 on B adds no throughput over C8 (374 vs 381) and slows each session; upstream TensorFold 0.6.2 alone serves GLM's logprobs only without streaming, thinking or tools, so it was not run as its own arm (Mia's recipe is its 0.6.0 GLM path plus her patches).

### Known limits of path A
- **Memory:** the lowest MemAvailable during a 150K prompt next to other streams was 7.3 GiB on the head (B keeps 12-14 GiB). Nothing failed in any section; leave the Sparks to this server.
- 8 long **fresh** prompts arriving together still prefill mostly in turn (8 × 23K: last first token 134 s; 8 × 32K: 218 s).
- Reasoning comes back as `reasoning_content` only (no `reasoning` field); `/v1/models` has no `max_model_len`; no `n > 1`; send a `seed` per best-of-N candidate (no seed: identical requests, identical replies). Logprobs are refused together with `response_format` or a thinking budget.
- The DFlash2 drafter is **CC BY-NC-ND 4.0 (non-commercial)**.

---

## Path A — Mia's TensorFold recipe with 8 streams

1. **Mia's recipe as she documents it** (both Sparks: link, Docker, weights): clone [`MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold`](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) at `4bbf2f3` on the head, copy [`mia-tensorfold/local.sh.example`](mia-tensorfold/local.sh.example) to its `scripts/local.sh`, fill in `WORKER` / `FABRIC_PEER`, and run `scripts/prepare.sh` once (it pulls her image and copies the weights to the worker).
2. **Our image, on both Sparks** (it is Mia's pinned image plus the changes above; ~1 minute):

   ```bash
   cd ~/glm-5.3-flash-2xdgx-spark/mia-tensorfold && docker build -t glm53-lc:prod .
   ```

3. **Let `start.sh` accept 8** (its own check stops at 4): in the Mia checkout, `sed -i 's/=~ ^\[1-4\]\$/=~ ^[1-8]$/' start.sh`.
4. **Start and operate** with [`mia-tensorfold/scripts/glm-prod.sh`](mia-tensorfold/scripts/glm-prod.sh): `start` frees page cache on both nodes without sudo, starts with `--reasoning-effort high`, and requires `/health` ok with 8 streams; `stop` pauses the watchdog; the cron line below restarts after 3 failed checks, one restart at a time (a lock).

   ```bash
   KIT=~/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold ~/glm-5.3-flash-2xdgx-spark/mia-tensorfold/scripts/glm-prod.sh start
   ( crontab -l; echo "* * * * * KIT=$HOME/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold $HOME/glm-5.3-flash-2xdgx-spark/mia-tensorfold/scripts/glm-prod.sh watch" ) | crontab -
   ```

5. **Check:** `python3 bench/lime_probe.py http://127.0.0.1:8000` (logprobs streamed and not, the fake-turn probe, seeds, tool calls) and `python3 bench/bigprompt_lp_test.py out.json http://127.0.0.1:8000`.

Grafana: [`tensorfold/grafana/tf_vllm_adapter.py`](tensorfold/grafana/tf_vllm_adapter.py) also maps the recipe's `tensorfold:*` metrics to the `vllm:*` series (`TF_METRICS_URL`, `TF_MODEL`).

---

## Path B — jayleaton's kit (the 2026-09-30 production)

### Bake-off 2026-09-30

One pair of DGX Sparks, one arm at a time with nothing else on the GPUs. The matched suite is [`bench/bakeoff_suite.py`](bench/bakeoff_suite.py); raw summaries are in [`results/2026-09-30-bakeoff/`](results/2026-09-30-bakeoff/). All arms ran at temperature 0 with the thinking modes stated, and each ran **once**, so treat differences under ~5% as noise.

#### C8 and single-stream speed (tok/s)

| Test | **B: TensorFold kit (this profile)** | vLLM Mia EXL3 kit | vLLM NVFP4 (LibertAIDAI, prior production) |
|---|---:|---:|---:|
| C1 decode, short prompt | **99** | 72 | 37 |
| C1 prose / code, thinking off ¹ | **45 / 61** | 21 / 33 | 23 / 29 |
| **C8 short prompts, aggregate (per stream)** | **368 (46)** | 251 (34) | 185 (24) |
| C8 reasoning, thinking High, aggregate | **93** | 74 | 76 |
| C8 × ~23K follow-up turn, aggregate | **46** | 31 | 38 |
| C8 × ~23K follow-up: prefix-cache hit | **98.7%** | 83.2% | 76.5% |
| C8 × ~23K follow-up: slowest first token | **5 s** | 52 s | 30 s |
| C8 × ~23K **cold** (fresh) prompts, aggregate | 10.8 | 10.2 | **13.8** |
| C8 × ~23K cold: slowest first token | 183 s | 187 s | **127 s** |
| Short request's first token during a 145K prefill | **1.5 s** | 6.5 s | 8.8 s |
| 128K+ single prefill, tok/s | 1,517 | 1,447 | 1,518 |
| Lowest MemAvailable, head / worker (GiB) | **12.9 / 13.2** | 5.5 / 7.4 | 4.5 / 6.5 |
| Context per request | **1,048,576** | 262,144 | 262,144 |

¹ The TensorFold C1 prose/code cells were measured on the previous kit build (`80dba15`) with the same weights. Single-stream decode does not depend on the C8 admission settings, and the release build adds about +2.5% decode. Every other TensorFold cell is from the published config on `5624dfc`.

#### Correctness gates

| Gate | B: TensorFold kit | Mia EXL3 | NVFP4 LibertAIDAI | NVFP4 nvidia |
|---|---|---|---|---|
| Korean rare-token corruption probe (U+FFFD in output) | **clean** | clean | **corrupted (6–9 U+FFFD)** | clean |
| Known-answer set (6 questions × thinking on/off) | 12/12 | 12/12 | 11/12 | 12/12 |
| Count to 200, greedy | 3/3 | 3/3 | 3/3 | 3/3 |
| Tool-call harness (42 calls, opencode-shaped) | 40/42, 0 corrupted | 39/42, 0 corrupted | 42/42 | not reached |
| 42K-token tool-call nondeterminism repro | not run (needs logprobs; patch 9001 now provides them) | 0/16 diverge | 0/16 diverge | not reached |

The Korean probe reproduces [mmastrac's report](https://github.com/mmastrac/glm-5.3-flash-4x-gx10) of intermittent mid-word corruption in the `LibertAIDAI/GLM-5.3-Flash-NVFP4` build. If you stay on the vLLM path, prefer another NVFP4 checkpoint and re-qualify.

#### What else was tried

- **The TensorFold kit with its own defaults serves C8 strictly one request at a time** (C8 short-prompt e2e 89 tok/s). Its admission gate needs `GLM53_TF_BATCH_ADMIT_GB` (2 GB) of free device memory before admitting a second request, and a 2× Spark never has that once one request runs. Setting it to 0.25 GB admits all 8 slots within ~1.5 s. This is the main change in our profile.
- **Multi-prompt prefill** (`GLM53_TF_MULTI_PREFILL`, kit default on) was A/B'd. It gives +61% short-prompt C8 (368 vs 228) but is about 15–25% slower when 8 long fresh prompts arrive at once (8 × 23K cold: 10.8 vs 12.7 tok/s; 8 × 56–58K: 4.5 vs 5.9, last first token 450 s vs 334 s). **We keep it on**, because agent traffic is mostly cached follow-ups plus short appends.
- **Official `nvidia/GLM-5.3-Flash-NVFP4` on the vLLM path:** clean on correctness. Its BF16 MTP layer needs an `exclude_modules` patch and a non-Marlin MoE backend (`flashinfer_cutlass`), which leaves only a 300K-token KV pool. The head dropped to 1.5 GiB MemAvailable at C8 and the run was aborted by our memory guard. It is not viable on 2 Sparks at C8 as configured.
- **vLLM NVFP4 overlays** (per-process memory cap 0.92, 2 ms shm spin-wait, prompt-token details): about +5–9% on C1/C8. Adding `--long-prefill-token-threshold 3584` dropped follow-up prefix-cache reuse from 76% to 29%, so we don't recommend it.

#### Known limits of path B

- 8 long **fresh** prompts arriving together prefill mostly in turn: the last one waits minutes for its first token (8 × 23K: up to 183 s; 8 × 58K: up to 450 s). Raise client HTTP timeouts accordingly.
- `logprobs` / `top_logprobs` (≤ 20, answer tokens only, streaming included) need our patch [9001](tensorfold/patches/README.md): the kit alone returns `null`. There is no `n > 1`, and each request should send its own `seed` for varied samples: without one the kit derives the seed from the prompt, so identical requests return identical text. There is no vLLM `/metrics` schema; [`tensorfold/grafana/`](tensorfold/grafana/) has an adapter.
- The DFlash2 drafter is **CC BY-NC-ND 4.0 (non-commercial)**. Check every weight's license against your use, or set `DRAFTER=` empty to use MTP drafts only (slower).

---

### Path B quick start

This layers our C8 profile on the upstream kit. Follow the kit's own [`AGENTS.md`](https://github.com/jayleaton/glm53-tensorfold-spark/blob/main/AGENTS.md) for node checks, link settings and troubleshooting.

**Requirements:** two DGX Sparks cabled CX7 to CX7 with an IP on the link, Docker with the NVIDIA runtime on both, and passwordless SSH from the head to the worker. Each node needs about 165 GB for the checkpoint plus about 84 GB for prepared weights. Sudo is **not** required.

#### 1. Kit and weights (both nodes)

```bash
git clone --recurse-submodules https://github.com/jayleaton/glm53-tensorfold-spark ~/glm53-tensorfold-spark
cd ~/glm53-tensorfold-spark && git checkout 5624dfc6fce32d747727eaf4dbda98999f868066 && git submodule update --init

hf download brandonmusic/GLM-5.3-Flash-tr3-4bpw --revision 5ab363a8dcf6405955fd5f99671e01a1c9fb124b
hf download incoai/GLM-5.3-Flash-DFlash2 --revision 7d74cdd881ed7e32c31175984a67823127b66cfe   # CC BY-NC-ND 4.0
```

#### 2. Config (head)

```bash
git clone https://github.com/LimeChain/glm-5.3-flash-2xdgx-spark ~/glm-5.3-flash-2xdgx-spark
cp ~/glm-5.3-flash-2xdgx-spark/tensorfold/config/prod-c8.env.example ~/glm53-tensorfold-spark/config/prod.env
$EDITOR ~/glm53-tensorfold-spark/config/prod.env
```

Fill in `WORKER_SSH`, `HEAD_IP`, `HEAD_HF`, `WORKER_HF`, and check `NCCL_SOCKET_IFNAME` / `NCCL_IB_HCA` against `ibdev2netdev`. Leave everything else as shipped: the header of [`prod-c8.env.example`](tensorfold/config/prod-c8.env.example) lists each value that differs from the kit's release config and why.

#### 3. Build, start, verify

```bash
cd ~/glm53-tensorfold-spark
scripts/serve.sh build            # builds the image and ships it to the worker
scripts/serve.sh preflight
TF_KIT=~/glm53-tensorfold-spark ~/glm-5.3-flash-2xdgx-spark/tensorfold/scripts/tf-prod.sh start
curl -s http://127.0.0.1:8000/v1/models
```

For OpenAI `logprobs` (verifiers, best-of-N scoring), build the patched image once on both nodes and set `IMAGE=glm53-tensorfold:lp9001` before starting ([`tensorfold/patches/`](tensorfold/patches/README.md)). The profile expects that image.

The first start compiles kernels and writes prepared weights (about 10 minutes); later starts take 20–60 s. `tf-prod.sh start` evicts page cache on both nodes first (no sudo) and refuses to report READY until all 8 slots have loaded. Without that step, a node that just copied weights boots with **1 slot** and serves one request at a time.

#### 4. Operate

```bash
tf-prod.sh status | stop | restart
# watchdog: restart after 3 failed health checks (stop sets a DISABLED marker so it stays down)
( crontab -l; echo "* * * * * TF_KIT=$HOME/glm53-tensorfold-spark $HOME/glm-5.3-flash-2xdgx-spark/tensorfold/scripts/tf-prod.sh watch" ) | crontab -
```

Clients: base URL `http://127.0.0.1:8000/v1`, model `glm-5.3-flash`, context 1,048,576. Thinking is on at High by default and returned in both `reasoning` and `reasoning_content`. The API has no auth and binds to loopback; put an authenticating proxy in front before exposing it.

#### Grafana / Prometheus

[`tensorfold/grafana/tf_vllm_adapter.py`](tensorfold/grafana/tf_vllm_adapter.py) (stdlib only, runs on the head) turns TensorFold's metrics and request log into `vllm:*` series: tokens, prefix-cache hits, running/waiting, KV %, finish reasons, and TTFT/TPOT/E2E/length histograms.
- Add it as a second target of your vLLM scrape job ([example](tensorfold/grafana/prometheus-scrape.example.yml)).
- It goes silent when vLLM serves the API port again, so the same dashboard works for both paths.
- Approximations: running = in-flight capped at the slot count; KV % comes from the last finished request; histograms cover completed requests only.

---

## Path C — vLLM + NVFP4

The original recipe (FlashInfer sparse-MLA TP2 specialization, FP8 KV, MTP3, 262K context) is unchanged and documented in [`docs/vllm-path.md`](docs/vllm-path.md), together with its historical benchmark receipts, the MTP-depth comparison and the tuning guides. Its bake-off numbers are the "NVFP4 LibertAIDAI" column above.

## Reproduce the bake-off

```bash
# on the head, against either path's endpoint; reports go to results/<arm>/
MODEL=glm-5.3-flash WORKER_SSH=user@worker python3 bench/bakeoff_suite.py --arm my-run --out results/my-run \
    --base http://127.0.0.1:8000 --model glm-5.3-flash [--no-logprobs]    # --no-logprobs for TensorFold
python3 bench/bakeoff_summarize.py results/my-run
```

Sections: api, correct, korean, count, toolcall, c1, repo (C1/C8 short), c8think, c8x19k (cold + follow-up), depth, prefill128k and ping120k. A memory guard cancels the run if either node's MemAvailable stays under 2 GiB.

## Credits and licensing

- TensorFold engine: [ashhart/TensorFold](https://github.com/ashhart/TensorFold) (Apache-2.0 from 0.6.0; earlier code MIT). Path A builds on [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) (Apache-2.0) and her published image; [`mia-tensorfold/NOTICE`](mia-tensorfold/NOTICE) lists what we change. Path B builds on [jayleaton/glm53-tensorfold-spark](https://github.com/jayleaton/glm53-tensorfold-spark) (Apache-2.0). This repository ships patches, config profiles and ops scripts, not the engines.
- vLLM path: see [`docs/vllm-path.md`](docs/vllm-path.md) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md).
- Compared, not redistributed: Mia's earlier vLLM kit [MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks). Correctness probes are adapted from [mmastrac/glm-5.3-flash-4x-gx10](https://github.com/mmastrac/glm-5.3-flash-4x-gx10); the tool-call harness is from the TensorFold kit (Apache-2.0).
- Weights are external and not included: `brandonmusic/GLM-5.3-Flash-tr3-4bpw`, `incoai/GLM-5.3-Flash-DFlash2` (CC BY-NC-ND 4.0), and the NVFP4 checkpoints on the vLLM path. Their licenses apply.
- Measurements, the C8/8-stream profiles and patches 9001-9004: Christian Veselinov / LimeChain.
