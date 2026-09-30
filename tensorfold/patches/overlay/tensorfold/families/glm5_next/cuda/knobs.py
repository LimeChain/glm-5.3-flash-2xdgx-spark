"""Per-request engine knobs for GLM-5.3-Flash (patches/0090): ``"tf_knobs": {...}`` in a request body.

The GLM53_TF_* environment of the earlier patches sets each knob's DEFAULT at load; a request may override any
subset of the knobs below for itself only, so one loaded server (a load takes minutes) can A/B many variants.
After the request, the defaults apply again. Every knob here but ``fast_prefill`` only picks among speed paths
already proven exact (drafted == serial, chunk-size independence, bit-identical kernels), so a request's tokens do not
depend on them. ``fast_prefill`` (patches/0091) runs the prompt through patches/0080's fast kernels: the request's
tokens then depend on it and on ``prefill_rows`` (its chunk grid, rounded down to a multiple of 64), while drafted
== serial and resumed == fresh still hold within the same setting (fast snapshots resume fast requests of the same
grid only). ``fp8_prefill`` (patches/0092) likewise changes a fast request's tokens (patches/0083's FP8 kernels);
its snapshots are tagged C + 1 and resume FP8 fast requests of the same grid only.

    key             values                   env default                    what it switches
    lookup          0 | 1                    GLM53_TF_LOOKUP (1)            prompt-lookup drafts inside ``auto``
    lookup_min      1 .. 64                  GLM53_TF_LOOKUP_MIN (4)        shortest match ``auto``'s lookup drafts
    auto_fdrafts    1 .. 7                   GLM53_TF_AUTO_FDRAFTS (7)      most DFlash2 drafts a round of ``auto``
    expert_loop     0 | 1                    GLM53_TF_EXPERT_LOOP (1)       EXL3 ``grouped_loop`` for windows > 16 rows
    prefill_rows    1 .. max | "auto"        GLM53_TF_PREFILL_ROWS (64)     prefill chunk rows (max: the buffers',
                                                                            GLM53_TF_PREFILL_ROWS_MAX; "auto",
                                                                            patches/0085: chosen per prefill)
    calib_online    0 | 1                    GLM53_TF_CALIB_ONLINE (0)      rank 0's online verify-cost table
    longctx_graphs  0 | 1                    GLM53_TF_LONGCTX_GRAPHS (1)    long-context graphs / bounded indexer
                                                                            (1 needs the load-time default 1)
    profile         0 | 1                    GLM53_TF_PROFILE (0)           prefill timing probe (rank 0 prints)
    depth           "cost" | "threshold"     GLM53_TF_DEPTH (threshold)     plain ``auto``'s draft depths
    fast_prefill    0 | 1                    GLM53_TF_FAST_PREFILL (0)      patches/0080's fast prefill (needs a
                                                                            PREFILL_ROWS_MAX of at least 64)
    fp8_prefill     0 | 1                    GLM53_TF_FP8_PREFILL (0)       patches/0083's FP8 kernels in fast
                                                                            chunks (no effect with fast_prefill 0;
                                                                            needs FP8 tensor cores)
    prefill_overlap 0 | 1                    GLM53_TF_PREFILL_OVERLAP (0)   patches/0084's pipelined lean chunks
                                                                            (same bits; 1 = the env's variant, or
                                                                            gather,slab)
    fat_experts     0 | 1 | 2                GLM53_TF_FAST_EXPERTS=fat (1)  patches/0170's fat expert kernels in
                                             / auto (2), else 0             fast chunks (same bits as fast2); 2
                                                                            (patches/0270): fast2 or fat by the
                                                                            chunk's rows (GLM53_TF_FAST2_ROWS)
    moe_glue        0 .. 7                   GLM53_TF_MOE_GLUE (0)          patches/0190, bits: 1 grouping by a
                                                                            parallel sort, 2 one-kernel router, 4
                                                                            shared expert read in place by the
                                                                            combine (64+-row windows; same bits)
    mtp_window      0 .. 2^30 tokens         GLM53_TF_MTP_PREFILL_WINDOW    patches/0190: prefill runs the MTP
                                             (0 = off)                      head on the prompt's last ~N positions
                                                                            only (drafts may change, replies not)
    hc_fused        0 .. 3                   GLM53_TF_HC_FUSED (0)          patches/0190: 1 hc_post + next hc_pre's
                                                                            dots fused, 2 unrolled hc finish, 3 both
                                                                            (fast pipelined chunks; same bits)
    attn_bm32       0 | 1                    GLM53_TF_ATTN_BM32 (0)         patches/0190: 32-query latent attention
                                                                            tiles in bf16 fast chunks (same bits)
    b12x            0 .. 7                   GLM53_TF_B12X (0)              patches/0240, bits: 1 hc mixing dots
                                                                            fused with hc_post, 2 KDA in 16-row
                                                                            tiles, 4 one-pass sparse attention (fast
                                                                            chunks; NEW arithmetic, own snapshot tag)

Knobs fixed at load (``LOAD_ONLY``) are refused with the reason. Rank 0 validates a request's knobs
(``parse``), resolves every knob to the value the request runs with, and sends the integers rank 1 needs in the
request header (``encode`` / ``decode``); both ranks set them before the prefill and restore their own values
after the request. Nothing here imports torch (host tests).
"""

from __future__ import annotations

from typing import Any, Mapping

from . import deep as _deep                 # patches/0380

DEPTHS = ("threshold", "cost")

# key -> (lowest, highest or None for "the engine's limit"), all integers (JSON true/false count as 1/0)
RANGES: dict[str, tuple[int, int | None]] = {
    "lookup": (0, 1),
    "lookup_min": (1, 64),
    "auto_fdrafts": (1, _deep.DRAFTS),   # patches/0380: up to GLM53_TF_MAX_DRAFT_ROWS - 1
    "expert_loop": (0, 1),
    "prefill_rows": (1, None),
    "calib_online": (0, 1),
    "longctx_graphs": (0, 1),
    "profile": (0, 1),
    "fast_prefill": (0, 1),              # patches/0091
    "fp8_prefill": (0, 1),               # patches/0092
    "prefill_overlap": (0, 1),           # patches/0093
    "fat_experts": (0, 2),               # patches/0170 (2: patches/0270's auto)
    "moe_glue": (0, 7),                  # patches/0190
    "mtp_window": (0, 1 << 30),          # patches/0190
    "hc_fused": (0, 3),                  # patches/0190
    "attn_bm32": (0, 1),                 # patches/0190
    "b12x": (0, 7),                      # patches/0240
}
PER_REQUEST = tuple(RANGES) + ("depth",)

# the knobs rank 1 applies itself, in header order (lookup / lookup_min travel in the policy code, depth as
# patches/0071's cost flag, calib_online as rank 0's cost table)
HEADER = ("expert_loop", "prefill_rows", "longctx_graphs", "profile", "auto_fdrafts", "calib_online", "fast_prefill",
          "fp8_prefill", "prefill_overlap", "fat_experts", "moe_glue", "mtp_window", "hc_fused", "attn_bm32",
          "b12x",                                                        # patches/0240
          "logprobs")                                                    # patches/9001: top logprobs (0: off)

LOAD_ONLY: dict[str, str] = {
    "nonexpert": "GLM53_TF_NONEXPERT is the stored weight format, quantized once at load",
    "latent_kv": "GLM53_TF_LATENT_KV is the attention cache layout, allocated at load",
    "kv_dtype": "GLM53_TF_KV_DTYPE (patches/0220) is the latent cache's row format (bf16 / fp8), allocated at load",
    "comm": "GLM53_TF_COMM (prefetch / NCCL protocol) is captured into the CUDA graphs and set before NCCL starts",
    "batch": "GLM53_TF_BATCH builds the batch scheduler at load",
    "prefill_rows_max": "GLM53_TF_PREFILL_ROWS_MAX sizes the window buffers at load (a request picks any "
                        "prefill_rows up to it)",
    "calib": "GLM53_TF_CALIB is the load-time cost calibration (use calib_online per request)",
    "kda_proj_bf16": "GLM53_TF_KDA_PROJ_BF16 keeps (or not) the KDA projections' bf16 copies, made at load, and "
                     "changes a fast prefill's bits (snapshots are not tagged by it)",
    "latent_tc": "GLM53_TF_LATENT_TC (patches/0190) changes a fast prefill's bits and its snapshot tag; the batch "
                 "scheduler and the session store key snapshots by the load-time tag",
}


def parse(raw: Any, *, rows_max: int, longctx_ok: bool = True, batch: bool = False,
          fp8_ok: bool = True) -> dict[str, Any]:
    """A request's ``tf_knobs`` as {key: value} (only the keys it sets), or ValueError naming the problem.
    ``rows_max``: the most prefill rows the buffers hold; ``longctx_ok``: long-context graphs can be turned on
    (the engine was loaded with GLM53_TF_LONGCTX_GRAPHS=1); ``batch``: the engine batches requests (patches/0120:
    every knob but ``calib_online``, whose table would time shared rounds); ``fp8_ok``: the FP8 prefill kernels can
    run here (patches/0092)."""

    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError("tf_knobs must be an object, e.g. {\"prefill_rows\": 256, \"lookup\": 0}")
    out: dict[str, Any] = {}
    for key, value in raw.items():
        if key in LOAD_ONLY:
            raise ValueError(f"tf_knobs.{key} cannot change per request: {LOAD_ONLY[key]}; restart both ranks")
        if key == "depth":
            v = str(value).strip().lower() if isinstance(value, str) else None
            if v not in DEPTHS:
                raise ValueError(f"tf_knobs.depth={value!r}: expected \"cost\" or \"threshold\"")
            out[key] = v
            continue
        if key == "prefill_rows" and isinstance(value, str) and value.strip().lower() == "auto":
            out[key] = 0                 # patches/0085: ``pfgrid.AUTO``, each prefill picks its chunk rows
            continue
        if key not in RANGES:
            raise ValueError(f"tf_knobs.{key}: unknown knob (per request: {', '.join(PER_REQUEST)}; "
                             f"fixed at load: {', '.join(LOAD_ONLY)})")
        if isinstance(value, bool):
            value = int(value)
        if not isinstance(value, int):
            raise ValueError(f"tf_knobs.{key}={value!r}: expected an integer"
                             + (" or \"auto\"" if key == "prefill_rows" else ""))
        lo, hi = RANGES[key]
        hi = rows_max if hi is None else hi
        if not lo <= value <= hi:
            extra = " (GLM53_TF_PREFILL_ROWS_MAX at load)" if key == "prefill_rows" else ""
            raise ValueError(f"tf_knobs.{key}={value}: expected {lo} to {hi}{extra}")
        if key == "fast_prefill" and value and rows_max < 64:          # patches/0091: the smallest chunk grid
            raise ValueError(f"tf_knobs.fast_prefill=1: fast prefill chunks are multiples of 64 rows, the buffers hold "
                             f"{rows_max} (GLM53_TF_PREFILL_ROWS_MAX at load)")
        if key == "fp8_prefill" and value and not fp8_ok:              # patches/0092
            raise ValueError("tf_knobs.fp8_prefill=1: this GPU / Triton has no FP8 (e4m3) tensor-core dots")
        if key == "calib_online" and value and batch:                  # patches/0120
            raise ValueError("tf_knobs.calib_online=1: the online cost table times one request's rounds; with "
                             "GLM53_TF_BATCH > 1 requests share their rounds")
        if key == "longctx_graphs" and value and not longctx_ok:
            raise ValueError("tf_knobs.longctx_graphs=1: this engine was loaded with GLM53_TF_LONGCTX_GRAPHS=0 (no "
                             "long-context graph buffers); load with the default 1 to switch it per request")
        out[key] = value
    return out


def encode(values: Mapping[str, int]) -> list[int]:
    """The header block: [count, values in ``HEADER`` order]."""

    return [len(HEADER)] + [int(values[k]) for k in HEADER]


def decode(block: list[int]) -> tuple[dict[str, int], list[int]]:
    """(knobs, the rest of the list) from a list starting with an ``encode`` block."""

    n = int(block[0])
    if n != len(HEADER):
        raise RuntimeError(f"the request header carries {n} knobs, this rank expects {len(HEADER)}: both ranks must "
                           "run the same patched TensorFold")
    return dict(zip(HEADER, (int(v) for v in block[1:1 + n]))), list(block[1 + n:])
