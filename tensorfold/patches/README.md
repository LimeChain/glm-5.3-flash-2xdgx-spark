# TensorFold patches (LimeChain)

## 9001 — OpenAI logprobs (`9001-glm-logprobs.patch`)

The kit returns `logprobs: null`. This patch adds real log-probabilities so verifiers and best-of-N scorers can rank candidates.

**Behavior**
- `/v1/chat/completions`: `"logprobs": true, "top_logprobs": 0..20` returns `choices[0].logprobs.content`, one entry per answer token with `token`, `logprob`, `bytes` and `top_logprobs`.
- `/v1/completions`: `"logprobs": N` returns the legacy object (`tokens`, `token_logprobs`, `top_logprobs`, `text_offset`).
- **Streaming:** every chunk that carries answer `content` also carries `choices[0].logprobs.content` for exactly those tokens.
- **Reasoning on (high / xhigh / max):** logprobs cover the final answer only (tokens after `</think>`), never the reasoning tokens or EOS. The concatenated `token`s equal `content`.
- **Values:** natural-log probabilities from the model's raw logits over the **full vocabulary**. Both ranks' halves are combined with an exact log-sum-exp. The top alternatives are the true top 20 across both ranks. Temperature, top-k and top-p are not applied, the same as vLLM's default `raw_logprobs`. All values are ≤ 0.
- **Cost:** zero for requests that don't ask. Requests that ask pay one extra small all-gather per verify round.
- **Unchanged:** sampling, drafting and the generated tokens. With the same seed or greedy, a request returns byte-identical text with or without `logprobs`.
- Bad values (e.g. `top_logprobs: 50`) get HTTP 400 with `param: "logprobs"`.

**Build** — the patch applies on top of the kit's full series. The simplest way is to overlay the patched files on an image built from the kit at `5624dfc`:

```bash
# 1. build the kit's image as usual (scripts/serve.sh build -> glm53-tensorfold:dev on both nodes), then on BOTH nodes:
cd ~/glm-5.3-flash-2xdgx-spark/tensorfold/patches
docker build --build-arg BASE=glm53-tensorfold:dev -t glm53-tensorfold:lp9001 -f Dockerfile.logprobs overlay
# 2. IMAGE=glm53-tensorfold:lp9001 in config/prod.env, then tf-prod.sh restart
```

`overlay/tensorfold/` is the five patched files for kit `5624dfc` exactly. For another kit revision, apply
`9001-glm-logprobs.patch` (paths `src/tensorfold/...`) to `vendor/TensorFold` after the kit's own patch series.

The patch adds one integer (`logprobs`) to the request header that rank 0 sends rank 1, so **both ranks must run the patched code**. The kit refuses mismatched headers at the first request.

**Verified on 2× DGX Spark, 8 slots (2026-09-30)**
- Greedy / seeded text byte-identical to the unpatched image: 3/3 hashes, reasoning off/high/xhigh.
- Non-streamed and streamed, high and xhigh: token texts rebuild `content` exactly; every `logprob` ≤ 0 and ≤ the top alternative; 20 alternatives in descending order.
- Streaming: 16/16 and 18/18 content chunks carried their logprobs.
- Greedy: the chosen token equals the top-1 alternative on every position.
- C8 mixed (4 with logprobs, 4 without, sampled): 8/8 `stop`, all valid.
- `/v1/completions` legacy object.
- Speed A/B, same session, 2 runs each (`bench/bakeoff_suite.py --only repo`, no logprobs asked): C8 short-prompt aggregate 363 / 329 tok/s unpatched vs 362 / 368 patched; C1 99 / 72 vs 99 / 99. No measurable cost.

## 9002 — stop at `<|assistant|>` (in the same patch file and overlay)

GLM-5.3 lists `<|endoftext|>`, `<|user|>` and `<|observation|>` as end tokens, but on short exact-format replies it ends the turn with `<|assistant|>` and then writes a fake next turn into the answer (`...</score_A><|assistant|>We need answer exactly...`), logprobs included. `extra_stop()` (engine.py) adds `<|assistant|>` and generation_config's `eos_token_id` to the stop set by token id at load, on both ranks, before the engine and the batcher copy it. The decode loops already compare ids and cut a drafted window at the first stop token. `GLM53_TF_EXTRA_STOP` overrides the list (empty turns it off).

Verified: `Output exactly: <score_A> B </score_A>` with top_logprobs 20 and reasoning high, 10 seeds (5 streamed, 5 not). Before the fix, 10/10 leaked a fake turn. After it, 10/10 return exactly the answer with `finish_reason: stop`, and the last logprob token is `>`. The OpenAI `stop` parameter is honored too.
