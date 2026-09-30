"""GLM-5.3-Flash's CUDA engine behind ``tensorfold.cuda.server``: two ranks over NCCL, one per machine.

Rank 0 serves HTTP and sends each request's header and prompt to rank 1 over the engine's all-gather; both ranks
then run the same prefill and drafted decode. Both sample every row with the same keyed rule from the same
gathered candidates, so they agree without a broadcast, and the reply is byte-identical to serial decoding on
the same two ranks.

Prefix reuse: the committed state after the last request's prompt and after its reply are kept (``decode.
Snapshot``), and a prompt that extends either resumes from it. Rows never depend on their chunk-mates, so a
resumed prompt ends in the state a fresh prefill gives. Both ranks keep the same snapshots; rank 0 names the one
it resumes from in the header.

Fast prefill (patches/0080, GLM53_TF_FAST_PREFILL=1, ``fastpf.py``): prefill chunks run kernels that are not
row-invariant, at absolute multiples of a chunk grid C; the only state kept is the prompt's at its last grid point
(no snapshot after the reply), and fast requests resume only from such snapshots of their own grid, which keeps
resumed == fresh and drafted == serial (the proof is in ``fastpf``).

Sessions (patches/0110, GLM53_TF_SESSION_GIB > 0, ``sessions.py``): besides those two snapshots, a store of many
sessions' snapshots and attention rows (paged, shared prefixes stored once, least recently used evicted) that a
request resumes from by copying them into the live caches, when that resumes more of its prompt; rank 0 decides and
sends the plan after the prompt. ``cached`` in the stats (``usage.prompt_tokens_details.cached_tokens``) is the
prompt prefix a request resumed from either. patches/0250 (GLM53_TF_SESSION_DISK, ``sessdisk.py``): the store's entries
also on local NVMe, restored from there (read straight into the live caches) when the RAM store no longer holds
them, also after a restart; ``stats["disk"]`` reports the read.

A request's draft policy is a spec (the engine's default, or the request's through ``app.GlmApp``):

    0             serial: one token a round
    N             N MTP drafts a round (the checkpoint's MTP head)
    a[:LOW:HIGH]  1 to 3 MTP drafts from the running acceptance (the default, a:0.6:0.85)
    cN:P          up to N MTP drafts while the product of the drafts' own probabilities stays at or above P
    f...          the same with DFlash2 drafts (fN, fcN:P, fa:...), when both ranks loaded the draft model
    auto          the default. Greedy requests: each round drafts with the MTP head (c3:0.35) or DFlash2
                  (fc7:0.3 with patches/0010, GLM53_TF_AUTO_FDRAFTS; upstream fc5:0.3), whichever has committed
                  more tokens per millisecond in this request
                  (``decode.DrafterChoice``: 2 rounds of each first, then a 3% margin to switch and one round of
                  the other every 8). Sampled requests: MTP drafts, 1 to 3 from the running acceptance
                  (a:0.6:0.85), where DFlash2's sampled chains measured slower. MTP only without the draft model.
                  On an EXL3 checkpoint with the draft model, every request drafts with DFlash2 (fc5:0.3), which
                  measured best or tied in all four cells there. A checkpoint without the MTP head drafts with
                  DFlash2 only (MTP specs run as their DFlash2 versions) and needs the draft model.
    auto:E:EVERY:MARGIN
                  the same choice with E rounds of each first, a probe every EVERY rounds and a MARGIN to switch,
                  for sampled requests too (MTP a:0.6:0.85 against DFlash2 there)
    lN[:M]        (patches/0020, ``lookup.py``) prompt-lookup drafts: each round whose history (prompt and reply)
                  ends with a span of at least M tokens (default 3) that occurred before verifies up to N (1 to
                  7) of the tokens that followed it, with no drafter step; other rounds draft as auto's MTP arm.
                  ``auto`` (kinds 4, 5) also drafts from the history when GLM53_TF_LOOKUP=1 (the default): a
                  round with a match of at least GLM53_TF_LOOKUP_MIN tokens (default 4) verifies the lookup's
                  tokens instead when their expected tokens a millisecond beat the request's MTP / DFlash2 rounds
    o             (patches/0071, ``depth.py``) auto with cost-derived depths: each round verifies the number of drafts
                  that maximizes expected committed tokens less the request's running rate times their ms (verify
                  window, draft steps), from the drafters' probabilities corrected per position by their running
                  acceptance; the lookup gate the same way. ``auto`` does this when GLM53_TF_DEPTH=cost (rank 0's
                  setting travels in the header; default: the thresholds above)
    om[N], of[N]  the same depths with MTP drafts only / DFlash2 drafts only, up to N (default 7)
"""

from __future__ import annotations

import os

import hashlib
import json
import struct
import threading
import time
from pathlib import Path
from typing import Any, Callable

from .lookup import LOOKUP_KIND, lookup_for, parse_policy, with_lookup     # patches/0020
from . import depth as depth_mod                                            # patches/0071
from . import knobs as knobs_mod                                            # patches/0090
from . import deep as deep_mod                                              # patches/0380

DEFAULT_POLICY = "auto"
DFLASH_POLICY = "fc5:0.3"             # DFlash2 drafts every round: up to 5 while their probability product holds 0.3
EXL3_AUTO = DFLASH_POLICY             # what auto runs on an EXL3 checkpoint with the draft model
GRAPH_ROWS = (1, 2, 3, 4, 5, 6)       # verify windows captured as CUDA graphs
# the widest verify window (a pending token and up to 7 drafts); patches/0380: GLM53_TF_MAX_DRAFT_ROWS (8 to 16)
MAX_ROWS = deep_mod.ROWS
DENSE_CAPACITY = 2560                 # cache slots while DSA attention stays dense (contexts up to 2,051 tokens)


def auto_f_most() -> int:
    """GLM53_TF_AUTO_FDRAFTS (patches/0010): the most DFlash2 drafts a round of ``auto`` verifies, still cut where
    the drafts' probability product falls under 0.3 (upstream 5). A DFlash2 block proposes 7 positions for the
    cost of one pass, and every verify row from the third on costs the same ~4.7 ms, so a 6th and 7th draft are
    admitted by the same rule and at the same price as the 5th; on confident stretches (structured text, code
    copied from the context) a round then commits up to 8 tokens instead of 6. Both ranks must agree on it."""

    most = int(os.environ.get("GLM53_TF_AUTO_FDRAFTS", "7"))
    if not 1 <= most < MAX_ROWS:
        raise ValueError(f"GLM53_TF_AUTO_FDRAFTS={most}: expected 1 to {MAX_ROWS - 1}")
    return most


def encode_policy(spec: str) -> list[int]:
    """A policy spec as 4 ints: kind (0 serial, 1 fixed, 2 running acceptance, 3 confidence; plus 10 for DFlash2
    drafts), most drafts, two parameters in millionths."""

    spec = str(spec).strip()
    bad = ValueError(f"draft policy {spec!r}: expected auto[:E:EVERY:MARGIN], 0, N, a[:LOW:HIGH], cN:P, or one of "
                     f"these after f (N from 1 to {MAX_ROWS - 1})")
    if spec.startswith("fl"):
        raise bad
    if spec == "o":                                 # patches/0071: auto with cost-derived depths (header flag)
        return encode_policy("auto")
    if spec.startswith("o"):                        # patches/0071: om[N], of[N]
        code = depth_mod.parse_policy(spec)
        if code is None:
            raise bad
        return code
    if spec.startswith("l"):                        # patches/0020: lN[:M], prompt-lookup drafts
        return parse_policy(spec)
    try:
        if spec == "auto" or spec.startswith("auto:"):
            parts = spec.split(":")
            if len(parts) not in (1, 4):
                raise bad
            explore, every, margin = (int(parts[1]), int(parts[2]), float(parts[3])) if len(parts) == 4 else (2, 8, 0.03)
            if explore < 1 or every < 0 or not 0 <= margin < 1:
                raise bad
            return [4 if len(parts) == 1 else 5, explore, every, int(round(margin * 1e6))]
        if spec.startswith("f"):
            code = encode_policy(spec[1:])
            return [code[0] + 10] + code[1:] if code[0] else code
        if spec.startswith("a"):
            parts = spec.split(":")
            if parts[0] != "a" or len(parts) not in (1, 3):
                raise bad
            low, high = (float(parts[1]), float(parts[2])) if len(parts) == 3 else (0.8, 0.9)
            return [2, 3, int(round(low * 1e6)), int(round(high * 1e6))]
        if spec.startswith("c"):
            most_text, conf = spec[1:].split(":")
            most = int(most_text)
            if not 0 < most < MAX_ROWS:
                raise bad
            return [3, most, int(round(float(conf) * 1e6)), 0]
        most = int(spec)
    except ValueError:
        raise bad from None
    if not 0 <= most < MAX_ROWS:
        raise bad
    return [1 if most > 0 else 0, most, 0, 0]


def decode_policy(code: list[int]):
    """The ``decode.DepthPolicy`` for a code, None for serial decoding, or ("auto", explore, every, margin,
    choose for sampled requests too)."""

    from .decode import DepthPolicy

    kind, most, a, b = code[:4]                     # patches/0020: auto codes carry lookup settings after these
    if kind == depth_mod.OPT_KIND:                  # patches/0071: om / of; the depths come from depth.py, and this
        # threshold policy (the drafter's usual one) only where no optimizer runs (GLM53_TF_BATCH rounds)
        return DepthPolicy(min(most, MAX_ROWS - 1), fixed=True, confidence=0.3 if a else 0.35)
    if kind == LOOKUP_KIND:
        return ("lookup", most, a)
    if kind in (4, 5):
        return ("auto", most, a, b / 1e6, kind == 5)
    kind %= 10
    if kind == 2:
        return DepthPolicy(min(most, MAX_ROWS - 1), low=a / 1e6, high=b / 1e6)
    if kind == 3:
        return DepthPolicy(min(most, MAX_ROWS - 1), fixed=True, confidence=a / 1e6)
    return DepthPolicy(min(most, MAX_ROWS - 1), fixed=True) if kind == 1 else None


def _f64_ints(x: float) -> list[int]:
    return list(struct.unpack("<2i", struct.pack("<d", float(x))))


def _ints_f64(lo: int, hi: int) -> float:
    return struct.unpack("<d", struct.pack("<2i", lo, hi))[0]


class GlmEngine:
    """GLM-5.3-Flash on two ranks (this one ``rank``): weights, MTP and DFlash2 drafting, per-request policies."""

    vision = None                           # patches/0500: rank 0's ``vision.Encoder`` with GLM53_TF_VISION=1

    def __init__(self, model_dir: Path, *, rank: int, master: str, port: int, policy: str = DEFAULT_POLICY,
                 drafter: Path | None = None, context: int = 0, serial_only: bool = False, comm=None) -> None:
        """``comm``: a communicator with ``all_gather`` and ``barrier`` instead of NCCL between two machines (tests)."""

        import torch

        from .comm import NCCL
        from .decode import Engine
        from .weights import Config, load

        encode_policy(policy)                           # a bad default fails here, not in the first request
        torch.cuda.set_device(0)
        from . import fastboot                          # patches/0140: the boot timeline ([boot] lines)

        fastboot.set_rank(rank)
        fastboot.mark("start: container, imports, CLI")
        self.torch = torch
        self.rank = rank
        self.model_dir = Path(model_dir)     # patches/0070: the tokenizer the calibration text is tokenized with
        self.policy = "0" if serial_only else policy
        self.serial_only = serial_only
        cfg = Config.read(model_dir)
        long_context = context > cfg.dense_limit
        capacity = max(DENSE_CAPACITY, context + MAX_ROWS) if long_context else DENSE_CAPACITY
        self.limit = capacity - MAX_ROWS if long_context else cfg.dense_limit
        from . import cpupin, decode_overlap

        cpupin.early(rank)                  # patches/0370 (GLM53_TF_CPU_PIN): before NCCL makes its threads
        decode_overlap.apply_gil()          # patches/0370 (GLM53_TF_DECODE_OVERLAP ``gil``)
        self.comm = comm if comm is not None else NCCL(rank, 2, master, port)
        self.comm.barrier()
        fastboot.mark("NCCL init (both ranks up)")
        if comm is None:
            # patches/0230: GLM53_TF_COMM_BACKEND=roce sends the model's small exchanges over a one-shot RoCE
            # all-gather (checked equal on both ranks; a failed setup falls back to NCCL on both)
            # (both ranks always call it: it compares the ranks' settings, whichever backend they asked for)
            from .roce import select as comm_select

            self.comm = comm_select(self.comm)
            if hasattr(self.comm, "rt"):
                fastboot.mark("RoCE all-gather connected")
        # both ranks must run the same calls: refuse to start when they were given different settings
        self.f_most = auto_f_most()
        longctx = int(os.environ.get("GLM53_TF_LONGCTX_GRAPHS", "1").strip() != "0")      # patches/0050
        # GLM53_TF_PREFILL_ROWS: rows a prefill chunk commits at once (upstream 64). Each chunk reads every weight
        # once, so larger chunks prefill faster; rows never depend on their chunk-mates, so the committed state
        # keeps the same bits. Costs about 5 MB of window buffers a row on an EXL3 checkpoint. patches/0090:
        # GLM53_TF_PREFILL_ROWS_MAX (default: GLM53_TF_PREFILL_ROWS) sizes the buffers, and a request may pick any
        # chunk up to it (``tf_knobs.prefill_rows``); GLM53_TF_PREFILL_ROWS is the default chunk. patches/0085: it may
        # be "auto" (``pfgrid.AUTO``: each prefill picks its chunk, up to the buffers; PREFILL_ROWS_MAX default 64).
        from . import pfgrid

        prefill_rows = pfgrid.parse_rows(os.environ.get("GLM53_TF_PREFILL_ROWS", "64"))
        rows_max = int(os.environ.get("GLM53_TF_PREFILL_ROWS_MAX", "") or prefill_rows or 64)
        if rows_max < max(prefill_rows, 1):
            raise ValueError(f"GLM53_TF_PREFILL_ROWS={prefill_rows}, GLM53_TF_PREFILL_ROWS_MAX={rows_max}: expected "
                             "1 <= PREFILL_ROWS <= PREFILL_ROWS_MAX")
        # patches/0380: GLM53_TF_MAX_DRAFT_ROWS sizes the windows (buffers, graphs, calibration); GLM53_TF_DFLASH_BLOCK
        # the drafter's block pass: both change what a round drafts, so both ranks must agree
        dblock = deep_mod.dflash_block(MAX_ROWS)
        mine = [int(drafter is not None), capacity, int(long_context), int(serial_only), self.f_most, longctx,
                prefill_rows, rows_max, MAX_ROWS, dblock]
        both = self._gather_ints(mine)
        if both[0] != both[1]:
            raise RuntimeError("the two ranks were started with different settings (draft model, context, drafts): "
                               f"rank 0 {both[0]}, rank 1 {both[1]}; pull the draft model on both machines (or pass "
                               "--drafter none to both) and give both the same flags (and GLM53_TF_LONGCTX_GRAPHS, "
                               "GLM53_TF_PREFILL_ROWS, GLM53_TF_PREFILL_ROWS_MAX, GLM53_TF_MAX_DRAFT_ROWS, "
                               "GLM53_TF_DFLASH_BLOCK)")
        from . import mtpw                  # patches/0430: GLM53_TF_MTP_WEIGHTS (unset: ``load`` as before)

        w = mtpw.load(model_dir, rank=rank)
        fastboot.mark("weights", f"{w.nbytes() / 1e9:.1f} GB on the GPU")
        w.comm = self.comm
        self.comm.barrier()
        fastboot.mark("barrier: the other rank's weights")
        if w.mtp is None and drafter is None and not serial_only:
            raise ValueError("this checkpoint has no MTP head and no DFlash2 draft model was given, so every round "
                             "would decode one token: pull the draft model on both machines (--drafter), or pass "
                             "--no-drafts to both for the serial reference")
        self.w = w
        # patches/0500: GLM53_TF_VISION=1: rank 0 loads the vision tower (BF16, ~1.13 GB) and encodes a request's
        # images; rank 1 needs none (it receives the rows with the prompt, whatever its own setting)
        from . import vision_prep

        self.vision = None
        if rank == 0 and vision_prep.enabled():
            from .vision import Encoder

            self.vision = Encoder(model_dir)
            fastboot.mark("vision tower", f"{self.vision.tower.nbytes / 1e9:.2f} GB on the GPU")
            print(f"[tensorfold] vision (patches/0500): image input on, tower {self.vision.tower.nbytes / 1e9:.2f} "
                  "GB on rank 0", flush=True)
        # patches/0420: GLM53_TF_DRAFT_VOCAB: the drafters' heads over a token list (before the drafter, the engine and
        # their CUDA graphs); it changes which graphs exist and what a round drafts, so both ranks must agree
        from . import draftvocab

        dvs = draftvocab.settings()
        dv_both = self._gather_ints(dvs.ints() if dvs is not None else [0] * 7)
        if dv_both[0] != dv_both[1]:
            raise RuntimeError("the two ranks were started with different GLM53_TF_DRAFT_VOCAB / _ARMS / _FALLBACK "
                               f"settings (or token lists): {dv_both[0]}, {dv_both[1]}")
        dv = draftvocab.attach(w, dvs) if dvs is not None else None
        w.draft_vocab = dv
        if dv is not None:
            fastboot.mark("draft vocabulary", f"{dv.n} rows a rank, {dv.head.nbytes() / 1e6:.1f} MB")
            if rank == 0:
                print(f"[tensorfold] {draftvocab.describe(dv, rank)}", flush=True)
        # GLM53_TF_LATENT_KV=1 (patches/0060): DSA layers cache the MLA latent (1 KB a token a layer) and attend in
        # latent space. The replies' bits depend on it, so both ranks must agree.
        from .latent import enabled as latent_kv, kv_dtype

        # patches/0220: GLM53_TF_KV_DTYPE (bf16 / fp8 latent rows) decides the replies' bits too
        lat_both = self._gather_ints([int(latent_kv()), int(kv_dtype() == "fp8")])
        if lat_both[0] != lat_both[1]:
            raise RuntimeError("the two ranks were started with different GLM53_TF_LATENT_KV / GLM53_TF_KV_DTYPE: "
                               f"{lat_both[0]}, {lat_both[1]}")
        # patches/0520: GLM53_TF_HC_CUDA, the fused hc boundary kernel (the Triton kernels' bits): both ranks must
        # agree, and it runs only if its bitwise self-check against the Triton kernels passed on both ranks
        from . import hc_cuda

        hc_both = self._gather_ints(hc_cuda.settings())
        if hc_both[0] != hc_both[1]:
            raise RuntimeError(f"the two ranks were started with different GLM53_TF_HC_CUDA: {hc_both[0][0]}, "
                               f"{hc_both[1][0]}")
        if hc_cuda.REQUESTED:
            why = hc_cuda.self_check(w.device)
            ok = self._gather_ints([int(why is None)])
            hc_cuda.configure(bool(ok[0][0] and ok[1][0]))
            if why is not None:
                print(f"[tensorfold] rank {rank}: GLM53_TF_HC_CUDA self-check failed ({why}); Triton hc kernels",
                      flush=True)
            elif rank == 0:
                print("[tensorfold] hc boundaries (patches/0520): one CUDA kernel a boundary, windows of up to "
                      f"{hc_cuda.ROWS} rows" + ("" if hc_cuda.ON else " -- off: rank 1's self-check failed"),
                      flush=True)
        # patches/0080: GLM53_TF_FAST_PREFILL and GLM53_TF_FAST_GATHER decide a fast prefill's bits, and so do the
        # optional fast kernels (patches/0081): both ranks must agree on all of them
        from . import fastpf

        fp_both = self._gather_ints(fastpf.settings())
        if fp_both[0] != fp_both[1]:
            raise RuntimeError("the two ranks were started with different fast-prefill settings "
                               "(GLM53_TF_FAST_PREFILL, GLM53_TF_FAST_GATHER, fast_kda / fast_qmm present): "
                               f"{fp_both[0]}, {fp_both[1]}")
        from . import pfgrid as _pfgrid

        be_both = self._gather_ints([int(_pfgrid.before_end())])      # patches/0540: it moves a cut (collectives)
        if be_both[0] != be_both[1]:
            raise RuntimeError(f"the two ranks were started with different {_pfgrid.BEFORE_ENV}: "
                               f"{be_both[0]}, {be_both[1]}")
        if fastpf.enabled() and rank == 0:
            gone = "; ".join(f"{k}: {v}" for k, v in fastpf.MISSING.items())
            from . import pfgrid

            rows = pfgrid.parse_rows(os.environ.get("GLM53_TF_PREFILL_ROWS", "64"))
            print(f"[tensorfold] fast prefill on: chunks of {pfgrid.show_rows(rows)} rows (patches/0085: the state "
                  f"does not depend on them), snapshots every {pfgrid.snapshot_grid()} tokens (tail "
                  f"{pfgrid.snapshot_tail()}), {'bf16' if fastpf.gather16() else 'fp32'} gathers"
                  + (f"; row-invariant kernels for {gone}" if gone else ""), flush=True)
        # patches/0400: GLM53_TF_KDA_V2 (fast chunks' KDA recurrence with fast_kda's bits: speed only, so the ranks
        # need not agree); its settings are parsed here so a bad value fails at load, not in the first prefill
        if fastpf.enabled() and fastpf.kda_v2 is not None and fastpf.kda_v2.mode():
            kmode, kcfg = fastpf.kda_v2.mode(), fastpf.kda_v2.config()
            if rank == 0:
                print(f"[tensorfold] KDA recurrence v2 ({'split' if kmode == 1 else 'fused'}, fast_kda's bits): "
                      + ", ".join(f"{k} {v}" for k, v in kcfg.items()) + " (patches/0400)", flush=True)
        # patches/0410: GLM53_TF_SPARSE_V2 (b12x bit 4's one-pass sparse attention through the Gluon v2 kernel, same
        # bits: speed only, so the ranks need not agree); parsed and compiled-module-imported here, not at the first
        # prefill
        if os.environ.get("GLM53_TF_SPARSE_V2", "").strip() not in ("", "0", "off"):
            from . import b12x_attn

            if b12x_attn.enable_v2() and rank == 0:
                st, qkl, qreg = b12x_attn.V2.config(True)
                print(f"[tensorfold] sparse latent attention v2 for b12x bit 4 (the one-pass kernel's bits): FP8 "
                      f"stages {st}, QK layout {'[2, 4]' if qkl else '[1, 8]'}, "
                      f"q {'half in registers' if qreg else 'in shared memory'} (patches/0410)", flush=True)
        # patches/0170: GLM53_TF_KDA_PROJ_BF16=1: a bf16 copy of each KDA input projection (the 4-bit weights'
        # values, rounded once) for fast chunks; checked equal on both ranks by fastpf.settings() above
        if fastpf.kda_bf16():
            extra = fastpf.attach_kda_bf16(w)
            fastboot.mark("KDA projections in bf16", f"+{extra / 2**30:.2f} GiB")
            if rank == 0:
                print(f"[tensorfold] KDA input projections: bf16 copies for fast prefill chunks, +{extra / 2**30:.2f} "
                      "GiB a rank (patches/0170)", flush=True)
        # patches/0083: GLM53_TF_FP8_PREFILL (the default of the FP8 kernels in fast chunks) and whether they can run
        from . import fp8pf

        f8_both = self._gather_ints(fp8pf.settings())
        if f8_both[0] != f8_both[1]:
            raise RuntimeError("the two ranks were started with different FP8 prefill settings (GLM53_TF_FP8_PREFILL, "
                               f"FP8 tensor cores available): {f8_both[0]}, {f8_both[1]}")
        if fp8pf.enabled() and rank == 0:
            print("[tensorfold] FP8 prefill: " + ("on in fast prefill chunks" if fp8pf.available() else
                                                  "requested, but no FP8 tensor cores / Triton float8e4nv: off"),
                  flush=True)
        # patches/0082: GLM53_TF_LEAN_PREFILL / GLM53_TF_LEAN_BLOCK decide how many all-gathers a lean chunk makes
        from . import lean as lean_mod

        ln_both = self._gather_ints(lean_mod.settings())
        if ln_both[0] != ln_both[1]:
            raise RuntimeError("the two ranks were started with different lean-prefill settings "
                               f"(GLM53_TF_LEAN_PREFILL, GLM53_TF_LEAN_BLOCK): {ln_both[0]}, {ln_both[1]}")
        # patches/0084: GLM53_TF_PREFILL_OVERLAP reorders a lean chunk's work (same kernels, same collectives in the
        # same order, same bits), so the ranks need not agree; parsed here so a bad value fails at load
        from . import pfoverlap

        if pfoverlap.DEFAULT.pipe and rank == 0:
            note = "" if lean_mod.enabled() else " (inactive without GLM53_TF_LEAN_PREFILL=1)"
            print(f"[tensorfold] prefill overlap: {pfoverlap.describe(pfoverlap.DEFAULT)}{note}", flush=True)
        # patches/0320: GLM53_TF_PREFILL_PP changes a pipelined chunk's collectives (row swaps instead of all-gathers):
        # both ranks must agree on it and, with it on, on whether the pipeline is the default
        from . import pfpp

        pp_both = self._gather_ints(pfpp.settings())
        if pp_both[0] != pp_both[1]:
            raise RuntimeError("the two ranks were started with different GLM53_TF_PREFILL_PP / GLM53_TF_PREFILL_OVERLAP "
                               f"settings: {pp_both[0]}, {pp_both[1]}")
        if pfpp.on() and rank == 0:
            note = "" if (lean_mod.enabled() and pfoverlap.DEFAULT.pipe) else \
                " (inactive without GLM53_TF_LEAN_PREFILL=1 and GLM53_TF_PREFILL_OVERLAP)"
            print(f"[tensorfold] prefill row split: hyper-connections on each rank's half of a sub-block (patches/0320)"
                  f"{note}", flush=True)
        # patches/0290: GLM53_TF_KV_POOL_TOKENS / _PAGE / _SLACK decide where rows live and the admission policy (rank
        # 0 plans, rank 1 follows it with a pool of its own): both ranks must agree
        from . import kvpool

        kp = kvpool.settings()
        kp_both = self._gather_ints(list(kp))
        if kp_both[0] != kp_both[1]:
            raise RuntimeError("the two ranks were started with different KV pool settings (GLM53_TF_KV_POOL_TOKENS, "
                               f"GLM53_TF_KV_POOL_PAGE, GLM53_TF_KV_POOL_SLACK): {kp_both[0]}, {kp_both[1]}")
        if kp[0] and rank == 0:
            kp = (kvpool.pool_tokens(kp[0], kp[1], capacity, self.limit),) + tuple(kp[1:])   # patches/0490: its size
            per_slot = kvpool.pages_for(capacity, kp[1])
            print(f"[tensorfold] KV pool (patches/0290): {kp[0]} tokens in pages of {kp[1]} shared by every slot; a "
                  f"slot grows to {capacity} tokens ({per_slot} pages)"
                  + ("" if kp[0] >= capacity else f"; one request alone cannot reach it (the pool holds {kp[0]})"),
                  flush=True)
        self.drafter = None
        if drafter is not None:
            from .dflash2 import Drafter

            self.drafter = Drafter(drafter, w, capacity=capacity, block=dblock or None)     # patches/0380
            fastboot.mark("DFlash2 drafter weights")
        # patches/0430 (GLM53_TF_DRAFT_DUMP): both ranks agree on the dump; its tap layers (the drafter's by default)
        from . import dump as dump_mod

        self.drafter, taps = dump_mod.setup(self, rank, self.drafter)
        self.e = Engine(w, capacity=capacity, max_rows=MAX_ROWS, prefill_rows=rows_max, graphs=True,
                        graph_rows=GRAPH_ROWS, long_context=long_context, taps=taps)
        self.e.prefill_rows = prefill_rows          # patches/0090: the default chunk; the buffers hold rows_max
        # patches/0090: long-context graphs can be switched on per request only when their buffers exist
        self.longctx_buffers = self.e.longctx
        self.long_context = long_context
        fastboot.mark("engine: caches, buffers, main + MTP CUDA graphs")
        if self.drafter is not None:
            self.drafter.dv_st = self.e.st          # patches/0420: the lone engine's fallback rule
            self.drafter.capture()
            fastboot.mark("DFlash2 CUDA graphs")
        # patches/0140: what the calibration table depends on besides the knobs (``fastboot.calib_ident``)
        self._calib_engine = {"capacity": capacity, "long_context": int(long_context), "limit": self.limit,
                              "drafter": str(drafter) if drafter is not None else "", "serial_only": int(serial_only),
                              "policy": self.policy, "f_most": self.f_most, "prefill_rows": prefill_rows,
                              "rows_max": rows_max, "max_rows": MAX_ROWS, "graph_rows": list(GRAPH_ROWS),
                              "model": fastboot.model_rev(model_dir), "mtp": int(w.mtp is not None)}
        if dv is not None:                  # patches/0420: trimmed draft heads time differently
            self._calib_engine["draft_vocab"] = dv.s.ints()
        self.costs = self._calibrate()
        fastboot.mark("calibration", str(self.costs.get("calib")))
        if rank == 0:
            c = self.costs
            print(f"[tensorfold] drafter timings (ms, fastest of 7, {c['calib']} tokens): {c['timed']}", flush=True)
            print("[tensorfold] drafter costs (ms): verify " + " ".join(f"{v:.1f}" for v in c["verify"]) +
                  f"; MTP draft {c['mtp']:.2f} (+{c['mtp_step']:.2f} a chained draft, +{c['mtp_row']:.2f} a row); "
                  f"DFlash2 block {c['block']:.2f} (+{c['taps_row']:.3f} a tap row)", flush=True)
        self.eos = tuple(w.cfg.eos)
        self.request = threading.local()    # the calling request's policy and stop-at-EOS (``app.GlmApp``)
        self.cache: list = []               # decode.Snapshot entries, each a prefix of the next
        # GLM53_TF_BATCH=N (patches/0030, 0120): up to N requests decode together, one forward a round over all their
        # verify windows (``batch.py``, built below once the knob defaults exist); 1 (default) serves one request at
        # a time as upstream. Both ranks must agree.
        self.batch = None
        n = int(os.environ.get("GLM53_TF_BATCH", "1") or 1)
        both = self._gather_ints([n])
        if both[0] != both[1]:
            raise RuntimeError(f"the two ranks were started with different GLM53_TF_BATCH: {both[0][0]}, {both[1][0]}")
        # patches/0070: GLM53_TF_CALIB_ONLINE=1, rank 0 refines the verify windows' costs from the windows rounds
        # run and sends the table in each request's header (calib.py); both ranks use the header's table
        from . import calib

        self.base_costs = self.costs
        self.online = (calib.OnlineCosts(self.costs["verify"]) if calib.online() and rank == 0 and n == 1
                       else None)
        self.calib_on = calib.online()      # patches/0090: the default of ``tf_knobs.calib_online``
        # patches/0190: the prefill glue knobs' defaults (per request: tf_knobs.moe_glue / mtp_window / hc_fused /
        # attn_bm32)
        from . import glue, latent, pfglue

        pfglue.set_moe_glue(glue, pfglue.moe_glue_default())
        pfglue.MTP_WINDOW = pfglue.mtp_window_default()
        glue.HC_FUSED = pfglue.hc_default()
        latent.BF16_BM32 = pfglue.bm32_default()
        # GLM53_TF_LATENT_TC (load-time: other bits, snapshots tagged G + 2): both ranks must agree
        latent.BF16_TC = pfglue.latent_tc_default()
        tc_both = self._gather_ints(pfglue.load_settings())
        if tc_both[0] != tc_both[1]:
            raise RuntimeError(f"the two ranks were started with different GLM53_TF_LATENT_TC / "
                               f"GLM53_TF_MTP_PREFILL_CACHE: {tc_both[0]}, {tc_both[1]}")
        pfglue.MTP_CACHE = pfglue.mtp_cache_default()   # prefill's MTP rows: cache writes only (same bits)
        # patches/0240: the b12x-derived fast-prefill kernels (per request: tf_knobs.b12x; new arithmetic, own snapshot
        # tag bits); the KDA kernel's load-time precision / value block decide its bits: both ranks must agree
        from . import b12xpf

        b12xpf.set_bits(b12xpf.default())
        bx_both = self._gather_ints(b12xpf.load_settings())
        if bx_both[0] != bx_both[1]:
            raise RuntimeError(f"the two ranks were started with different GLM53_TF_B12X_KDA_PREC / _BV / _WARPS: {bx_both[0]}, "
                               f"{bx_both[1]}")
        if rank == 0 and b12xpf.BITS:
            print(f"[tensorfold] b12x fast-prefill kernels (patches/0240): {b12xpf.describe()} by default (new "
                  "arithmetic in fast chunks, own snapshot tag)", flush=True)
        if rank == 0 and pfglue.MTP_CACHE:
            print("[tensorfold] MTP prefill rows: head cache writes only (patches/0190, GLM53_TF_MTP_PREFILL_CACHE=1)"
                  + ("" if latent.on(w) else "; inactive: GLM53_TF_LATENT_KV=0"), flush=True)
        if rank == 0 and latent.BF16_TC:
            print("[tensorfold] latent absorb / expand on bf16 tensor cores in fast chunks (patches/0190, "
                  "GLM53_TF_LATENT_TC=1: new arithmetic, own snapshot tag)"
                  + ("" if latent.on(w) else "; inactive: GLM53_TF_LATENT_KV=0"), flush=True)
        if rank == 0 and (pfglue.moe_glue(glue) or pfglue.MTP_WINDOW or glue.HC_FUSED or latent.BF16_BM32):
            print(f"[tensorfold] prefill glue (patches/0190): MoE glue {pfglue.moe_glue(glue)}, "
                  f"MTP prefill window {pfglue.MTP_WINDOW or 'off'}, fused hc {glue.HC_FUSED}, 32-query attention "
                  f"tiles {int(latent.BF16_BM32)}", flush=True)
        # patches/0450 (GLM53_TF_GPU_ROUND): both ranks run the same parts; the device sampler is first checked
        # against this host's numpy (libm and draws); a difference on either rank turns the device draw off on both
        from . import gpuround

        gr = gpuround.env()
        gr_both = self._gather_ints([gr.code(), int(gpuround.peek())])
        if gr_both[0] != gr_both[1]:
            raise RuntimeError("the two ranks were started with different GLM53_TF_GPU_ROUND / "
                               f"GLM53_TF_GPU_ROUND_PEEK: rank 0 {gr_both[0]}, rank 1 {gr_both[1]}")
        if gr.sample:
            from . import gpusample

            why = gpusample.self_check(w.device)
            ok = self._gather_ints([int(why is None)])
            w.meta["gpu_sample"] = bool(ok[0][0] and ok[1][0])
            if why is not None:
                print(f"[tensorfold] rank {rank}: the GPU sampler differs from numpy here ({why}); GLM53_TF_GPU_ROUND "
                      "draws on the host", flush=True)
            elif rank == 0:
                off = "" if w.meta["gpu_sample"] else " (the device draw is off: rank 1's check failed)"
                print(f"[tensorfold] GPU round (patches/0450): {gr.describe()}{off}", flush=True)
        if n > 1:                           # patches/0120: per-request knobs, fast / lean prefill and every drafter
            from .batch import Batcher

            self.batch = Batcher(self, n)
        self.store = self._sessions()       # patches/0110
        fastboot.mark("engine ready (session store, batcher)")
        dump_mod.arm()                      # patches/0430: records from here on (not the calibration passes)

    def _sessions(self):
        """patches/0110: the session store (``sessions.py``), or None (GLM53_TF_SESSION_GIB=0, the default).
        patches/0180: with GLM53_TF_BATCH > 1 only when GLM53_TF_BATCH_SESSIONS=1 (one store behind every slot,
        ``Batcher.attach_store``); otherwise the slots keep their own states as patches/0120 does. Both ranks must
        agree on its settings."""

        from . import sessions

        mine = sessions.settings()
        both = self._gather_ints(mine)
        if both[0] != both[1]:
            raise RuntimeError("the two ranks were started with different session settings (GLM53_TF_SESSION_GIB, "
                               f"GLM53_TF_SESSION_EVERY, GLM53_TF_SESSION_FORK_MIN): {both[0]}, {both[1]}")
        in_batch = False
        if self.batch is not None:
            in_batch = sessions.batch_enabled()
            got = self._gather_ints([int(in_batch)])
            if got[0] != got[1]:
                raise RuntimeError(f"the two ranks were started with different GLM53_TF_BATCH_SESSIONS: {got[0][0]}, "
                                   f"{got[1][0]}")
        if not mine[0] or (self.batch is not None and not in_batch):
            if mine[0] and self.rank == 0:
                print("[tensorfold] GLM53_TF_SESSION_GIB is ignored with GLM53_TF_BATCH > 1 unless "
                      "GLM53_TF_BATCH_SESSIONS=1", flush=True)
            return None
        store = sessions.SessionStore(self.e, self.drafter, sessions.budget_bytes())
        if self.batch is not None:          # patches/0180: every slot's caches, one index / budget / slab pool
            self.batch.attach_store(store)
        from . import sessdisk              # patches/0250: the NVMe tier (GLM53_TF_SESSION_DISK), both ranks

        sessdisk.attach(self, store)
        store.attach_share(sessions.PrefixShare.from_env(self.model_dir))     # patches/0310 (None: off)
        if self.rank == 0:
            ix = store.index
            print(f"[tensorfold] session store: {ix.budget / 2 ** 30:.1f} GiB, pages of {sessions.PAGE} tokens "
                  f"({ix.page_bytes / 2 ** 20:.2f} MiB, {ix.page_bytes / sessions.PAGE / 1024:.2f} KB a token), "
                  f"marks every {store.every} tokens and at forks of {store.fork_min}+"
                  + (f", behind {self.batch.n} batch slots" if self.batch is not None else ""), flush=True)
            if store.share is not None:
                sh = store.share
                print(f"[tensorfold] shared prefixes (patches/0310): a mark at the end of the system prompt (role "
                      f"tokens {sh.roles}, {sh.lo}-{sh.hi} tokens, {sh.points} a prompt), at most {sh.keep} "
                      f"system-only entries kept, admissions wait for a partner's mark: {int(sh.wait)}", flush=True)
        return store

    def _store_save(self, snaps: list) -> None:
        """patches/0110: store snapshots of the live state; rank 0 decides (stored or not, evictions), rank 1
        applies its decisions."""

        snaps = [s for s in snaps if s is not None]
        if getattr(self, "store", None) is None or not snaps:
            return
        if self.rank == 0:
            self._share(self.store.save_all(snaps))
        else:
            self.store.save_all(snaps, forced=self._share(None))

    # -- patches/0090: per-request knobs ---------------------------------------------------------------------------
    def parse_knobs(self, raw) -> dict:
        """A request's ``tf_knobs`` validated against this engine (ValueError says why not); rank 0."""

        most = getattr(self.e, "prefill_max", self.e.rows)          # patches/0082: the lean set's rows, if any
        from . import fp8pf

        return knobs_mod.parse(raw, rows_max=most, batch=self.batch is not None,
                               longctx_ok=self.longctx_buffers or not self.long_context,
                               fp8_ok=fp8pf.available())          # patches/0092

    def _knob_state(self) -> dict[str, int]:
        """This rank's current value of every knob in the header (the load-time defaults between requests)."""

        from . import b12xpf, exl3_mm, glue, latent, pfglue, pfoverlap, profile

        return {"expert_loop": int(bool(exl3_mm.LOOP)), "prefill_rows": int(self.e.prefill_rows),
                "longctx_graphs": int(bool(self.e.longctx)), "profile": int(bool(profile.ON)),
                "auto_fdrafts": int(self.f_most), "calib_online": int(bool(self.calib_on)),
                "fast_prefill": int(bool(self.e.fast_prefill)),          # patches/0091
                "fp8_prefill": int(bool(getattr(self.e, "fp8_prefill", False))),     # patches/0092
                "prefill_overlap": int(pfoverlap.on()),                  # patches/0093
                "fat_experts": exl3_mm.family(),                         # patches/0170 / 0270
                "moe_glue": pfglue.moe_glue(glue),                       # patches/0190
                "mtp_window": int(pfglue.MTP_WINDOW),
                "hc_fused": int(glue.HC_FUSED),
                "attn_bm32": int(bool(latent.BF16_BM32)),
                "b12x": int(b12xpf.BITS),                                # patches/0240
                "logprobs": 0}                                           # patches/9001: per request only

    def _set_knobs(self, values: dict[str, int]) -> None:
        """Set the knobs whose value differs from the current one (the others are left exactly as they are)."""

        from . import exl3_mm, profile

        now = self._knob_state()
        changed = {k: int(v) for k, v in values.items() if k in now and now[k] != int(v)}
        if "expert_loop" in changed:
            exl3_mm.LOOP = bool(changed["expert_loop"])
        if "prefill_rows" in changed:
            rows = changed["prefill_rows"]
            most = getattr(self.e, "prefill_max", self.e.rows)      # patches/0082: the lean set's rows, if any
            if rows and not 1 <= rows <= most:                     # patches/0085: 0 = "auto"
                raise RuntimeError(f"prefill_rows={rows}: the buffers hold {most} rows; both ranks must be "
                                   "started with the same GLM53_TF_PREFILL_ROWS_MAX")
            self.e.prefill_rows = rows
        if "longctx_graphs" in changed:
            on = bool(changed["longctx_graphs"]) and self.longctx_buffers
            self.e.longctx = on
            self.w.meta["longctx_bound"] = on
        if "profile" in changed:
            profile.ON = bool(changed["profile"])
        if "auto_fdrafts" in changed:
            self.f_most = changed["auto_fdrafts"]
        if "calib_online" in changed:
            self.calib_on = bool(changed["calib_online"])
        if "fast_prefill" in changed:           # patches/0091
            self.e.fast_prefill = bool(changed["fast_prefill"])
        if "fp8_prefill" in changed:            # patches/0092
            self.e.fp8_prefill = bool(changed["fp8_prefill"])
        if "prefill_overlap" in changed:        # patches/0093: the same bits either way (patches/0084)
            from . import pfoverlap

            pfoverlap.set_request(bool(changed["prefill_overlap"]))
        if "fat_experts" in changed:            # patches/0170 / 0270: the same bits every way (fat == fast2)
            exl3_mm.set_family(changed["fat_experts"])
        if "moe_glue" in changed:               # patches/0190: the same bits either way
            from . import glue, pfglue

            pfglue.set_moe_glue(glue, changed["moe_glue"])
        if "mtp_window" in changed:             # patches/0190: drafts only, replies unchanged
            from . import pfglue

            pfglue.MTP_WINDOW = int(changed["mtp_window"])
        if "hc_fused" in changed:               # patches/0190: the same bits either way (bitwise-tested)
            from . import glue

            glue.HC_FUSED = int(changed["hc_fused"])
        if "attn_bm32" in changed:              # patches/0190: the same bits (bitwise-tested), fast chunks only
            from . import latent

            latent.BF16_BM32 = bool(changed["attn_bm32"])
        if "b12x" in changed:                   # patches/0240: new arithmetic in fast chunks (the tag carries it)
            from . import b12xpf

            b12xpf.set_bits(changed["b12x"])

    def _knobs(self, values: dict[str, int]):
        """Context: the request's knobs on this rank, then this rank's previous values again (also on errors)."""

        import contextlib

        @contextlib.contextmanager
        def scope():
            saved = self._knob_state()
            try:
                self._set_knobs(values)
                yield
            finally:
                self._set_knobs(saved)

        return scope()

    def _calibrate(self) -> dict:
        """Milliseconds for ``decode.DrafterChoice``, timed as rounds use them and made the same on both ranks (the
        slower rank's time of each): a verify window of 1 to MAX_ROWS rows; an MTP draft (absorbing one row,
        sampling its draft), each further chained draft and each further absorbed row; a DFlash2 block with its host
        chain and each tap row DFlash2 takes. The machine has slow moments of a few seconds (page migration, most of
        all right after loading), so every piece is timed in turns over several passes and keeps its fastest run,
        and the windows of 2 rows and more follow a line through their times fitted with the median of pairwise
        slopes.

        patches/0070 (``calib.py``): the windows' tokens. GLM53_TF_CALIB=real (the default) times them on the
        model's own greedy continuation of a fixed text, SPANS spans of MAX_ROWS consecutive tokens each at their
        own positions, because an extra row costs mostly the experts it adds and real consecutive tokens share
        many; GLM53_TF_CALIB=random is upstream (uniform random ids, which overprice every row past the first).
        Both ranks run the same calls in either mode (the mode is checked equal first; the prompt is rank 0's)."""

        import numpy as np

        from . import calib
        from .decode import draft, prefill, serial_decode

        torch = self.torch
        e, st = self.e, self.e.st
        rng = np.random.default_rng(0)
        vocab = self.w.cfg.vocab
        how = calib.mode()
        modes = self._gather_ints([calib.MODES.index(how)])
        if modes[0] != modes[1]:
            raise RuntimeError(f"the two ranks were started with different GLM53_TF_CALIB: "
                               f"{calib.MODES[modes[0][0]]}, {calib.MODES[modes[1][0]]}")
        # patches/0140: ``cached`` reuses the table a real calibration stored for this pair of ranks (rank 0 reads,
        # both use its bytes); ``real`` measures and refreshes it; a miss measures as ``real``
        cache_name = None
        if how in ("real", "cached"):
            cache_name, hit = self._calib_cache(how)
            if hit is not None:
                return hit
            how = "real"

        def tokens(n: int) -> list[int]:
            return [int(t) for t in rng.integers(0, vocab, n)]

        stride = deep_mod.DEFAULT                     # patches/0380 (== MAX_ROWS without it)
        if how == "random":
            prompt, cont, spans = tokens(64), [], 0
        else:
            prompt = self._share(calib.prompt_ids(self.model_dir, vocab) if self.rank == 0
                                 else None)
            first = prefill(e, prompt, None, mtp=False, drafter=None)
            # patches/0380: spans start every 8 tokens whatever MAX_ROWS is (a deeper window's first 8 rows are the
            # tokens and positions an 8-row calibration times), and run MAX_ROWS tokens from there
            cont = serial_decode(e, first, calib.SPANS * stride + MAX_ROWS - stride, None).tokens   # greedy: both ranks
            spans = calib.SPANS
        best: dict[str, float] = {}
        for s in range(max(spans, 1)):
            prefill(e, prompt + cont[:s * stride], None, mtp=True, drafter=self.drafter)
            hidden = e.main_hidden(slice(0, MAX_ROWS)).clone()
            span = cont[s * stride:s * stride + MAX_ROWS]
            one, six = (tokens(1), tokens(6)) if not spans else (span[:1], span[:6])
            start = st.mtp_len

            def rewind(start=start) -> None:
                st.set_mtp_len(start)
                st.mtp_drafted = 0

            if not spans:
                pieces: dict[str, tuple] = {f"v{r}": (lambda w=tokens(r): e.forward(w), None)
                                            for r in range(1, MAX_ROWS + 1)}
            else:
                pieces = {f"v{r}_{s}": (lambda w=span[:r]: e.forward(w), None) for r in range(1, MAX_ROWS + 1)}
            if self.w.mtp is not None and s == 0:
                pieces["m1"] = (lambda: draft(e, hidden[:1], one, st.pos + 1, 1, None), rewind)
                pieces["m3"] = (lambda: draft(e, hidden[:1], one, st.pos + 1, 3, None), rewind)
                pieces["m6"] = (lambda: draft(e, hidden[:6], six, st.pos + 1, 1, None), rewind)
            back = None
            if self.drafter is not None and s == 0:
                d = self.drafter
                taps = e.tap_rows(8).clone()
                ctx = d.context_end

                def back() -> None:
                    if d.context_end != ctx:
                        d.pos_dev.sub_(d.context_end - ctx)
                        d.context_end = ctx

                pieces["block"] = (lambda: d.propose(one[0], 5, None, 0.0), None)
                pieces["taps8"] = (lambda: d.add_taps(taps), back)
            best.update({name: float("inf") for name in pieces})
            for turn in range(9):
                for name, (fn, prep) in pieces.items():
                    if prep is not None:
                        prep()
                    torch.cuda.synchronize()
                    t = time.perf_counter()
                    fn()
                    torch.cuda.synchronize()
                    if turn >= 2:
                        best[name] = min(best[name], (time.perf_counter() - t) * 1e3)
                rewind()
                if back is not None:
                    back()
        names = list(best)
        mine = torch.tensor([best[n] for n in names], dtype=torch.float32, device="cuda")
        got = torch.empty((2 * mine.numel(),), dtype=torch.float32, device="cuda")
        self.comm.all_gather(mine, got)
        both = calib.slower(names, got.tolist())
        e.reset()
        if self.drafter is not None:
            self.drafter.reset()
        verify = calib.window_costs(both, MAX_ROWS, spans)
        mtp = both.get("m1", 0.0)
        out = {"verify": verify, "mtp": mtp, "mtp_step": max((both.get("m3", 0.0) - mtp) / 2, 0.0),
               "mtp_row": max((both.get("m6", 0.0) - mtp) / 5, 0.0), "block": both.get("block", 0.0),
               "taps_row": max(both.get("taps8", 0.0) / 8, 0.0), "timed": {k: round(v, 2) for k, v in both.items()},
               "calib": how, "windows": [list(prompt), list(cont)]}
        if cache_name is not None and self.rank == 0:          # patches/0140
            from . import fastboot

            where = fastboot.calib_write(cache_name, out)
            if where is not None:
                print(f"[tensorfold] calibration table stored: {where}", flush=True)
        return out

    def _calib_cache(self, how: str) -> tuple[str, dict | None]:
        """patches/0140: the calibration table's cache name for this pair of ranks (both ranks' identities, all-
        gathered) and, with GLM53_TF_CALIB=cached, the stored table: rank 0 reads it and shares its bytes, so both
        ranks hold the same floats; None when there is none (or ``real``). A hit warms the prefill path once."""

        import json as _json

        from . import fastboot

        ident = fastboot.calib_ident(**self._calib_engine)
        ids = self._gather_ints(fastboot.ident_ints(ident))
        name = fastboot.calib_name(ids[0], ids[1])
        if how != "cached":
            return name, None
        payload = None
        if self.rank == 0:
            got = fastboot.calib_read(name)
            payload = [1] + fastboot.to_ints(_json.dumps(got).encode()) if got is not None else [0]
        shared = self._share(payload)
        if not shared or shared[0] != 1:
            if self.rank == 0:
                print(f"[tensorfold] no stored calibration table ({fastboot.calib_dir() / name}): measuring",
                      flush=True)
            return name, None
        costs = _json.loads(fastboot.from_ints(shared[1:]))
        costs["calib"] = f"cached ({costs.get('calib', 'real')})"
        if self.rank == 0:
            print(f"[tensorfold] calibration table from {fastboot.calib_dir() / name}", flush=True)
        if os.environ.get("GLM53_TF_BOOT_WARMUP", "1").strip() != "0":
            from .decode import prefill

            windows = costs.get("windows") or [[]]
            prompt = [int(t) for t in windows[0]]
            if prompt:          # the kernels a calibration would have run first, once (both ranks alike)
                prefill(self.e, prompt, None, mtp=True, drafter=self.drafter)
                self.torch.cuda.synchronize()
                self.e.reset()
                if self.drafter is not None:
                    self.drafter.reset()
        return name, costs

    def _gather_ints(self, values: list[int]) -> list[list[int]]:
        torch = self.torch
        mine = torch.tensor(values, dtype=torch.int32, device="cuda")
        got = torch.empty((2 * len(values),), dtype=torch.int32, device="cuda")
        self.comm.all_gather(mine, got)
        return [got[:len(values)].tolist(), got[len(values):].tolist()]

    def _share(self, values: list[int] | None) -> list[int]:
        """Rank 0's int list on every rank (a length, then the values, through the all-gather)."""

        torch = self.torch
        n = torch.tensor([len(values) if self.rank == 0 else 0], dtype=torch.int32, device="cuda")
        got = torch.empty((2,), dtype=torch.int32, device="cuda")
        self.comm.all_gather(n, got)
        count = int(got[0].item())
        buf = (torch.tensor(values, dtype=torch.int32, device="cuda") if self.rank == 0
               else torch.zeros((count,), dtype=torch.int32, device="cuda"))
        allv = torch.empty((2 * count,), dtype=torch.int32, device="cuda")
        self.comm.all_gather(buf, allv)
        return [int(v) for v in allv[:count].tolist()]

    def _adopt(self, table: list[int]) -> None:
        """patches/0070: the costs a request uses, the load-time ones with rank 0's verify table from the header (the
        same integers on both ranks, so the same floats), fixed for the whole request."""

        from .calib import with_table

        self.costs = with_table(self.base_costs, table)

    def _effective(self, code: list[int]) -> list[int]:
        """The code a request runs: plain ``auto`` is ``EXL3_AUTO`` on an EXL3 checkpoint with the draft model; on a
        checkpoint without the MTP head, ``auto`` is ``DFLASH_POLICY`` and an MTP spec runs as its DFlash2 version."""

        # EXL3_AUTO was measured with BF16 non-expert weights; stored in 4 bits (GLM53_TF_NONEXPERT) the MTP head is
        # as cheap as on the MLX checkpoint, so plain ``auto`` chooses between the drafters as it does there
        from . import weights as weights_mod                  # patches/0470: every class BF16 (NONEXPERT + _MAP)

        if code[0] == 4 and self.drafter is not None and self.w.cfg.quant == "exl3" \
                and weights_mod.precision_key() == "bf16":
            return encode_policy(EXL3_AUTO)
        if self.w.mtp is None and code[0] == depth_mod.OPT_KIND:       # patches/0071: om runs as of
            return [code[0], code[1], 1, 0]
        if self.w.mtp is None and code[0] in (1, 2, 3, 4, 5):
            return encode_policy(DFLASH_POLICY) if code[0] in (4, 5) else [code[0] + 10] + code[1:]
        return code

    def _drafters(self, code: list[int]) -> tuple[bool, bool, bool]:
        """(auto, MTP drafts, DFlash2 drafts) for a policy code."""

        auto = code[0] in (4, 5)
        if code[0] == depth_mod.OPT_KIND:           # patches/0071: om / of
            dflash = bool(code[2]) and self.drafter is not None
            return False, not dflash, dflash
        dflash = (auto or code[0] // 10 == 1) and self.drafter is not None
        return auto, auto or not dflash, dflash

    def _grid(self, fast: bool | None = None, rows: int | None = None, fp8: bool | None = None,
              b12x: int | None = None) -> int:
        """patches/0080: the snapshot tag a request uses: 0 (exact prefill), or a fast one (patches/0083: + 1 for an
        FP8 fast prefill). patches/0085: by mode only (``pfgrid.tag``: the snapshot grid G, G + 1), never by the
        request's chunk size, so fast requests of any ``prefill_rows`` share their snapshots (``rows`` is ignored)."""

        from . import pfgrid

        fast = self.e.fast_prefill if fast is None else fast
        fp8 = bool(getattr(self.e, "fp8_prefill", False)) if fp8 is None else fp8
        from . import pfglue

        from . import b12xpf

        return pfgrid.tag(bool(fast), fp8, getattr(self.e, "snap_grid", None),
                          pfglue.latent_tc(getattr(self, "w", None)),          # patches/0190: GLM53_TF_LATENT_TC
                          b12xpf.tag_bits(getattr(self, "w", None), b12x))     # patches/0240: tf_knobs.b12x

    def _request_grid(self, values: dict) -> int:
        """patches/0360: the snapshot tag a request's prefill will write, from the request's knobs (``values``, before
        they are applied): the lookup (``_resume``, the session store's plan) must use the same tag ``decode._prefill``
        gives the snapshots under those knobs. 0240 left ``b12x`` out of the lone engine's lookup, so the lookup used the
        rank's default bits (0 between requests) while the snapshots carried the request's: a request with b12x bits
        never resumed (``cached`` 0). The batcher uses this too."""

        return self._grid(bool(values["fast_prefill"]), int(values["prefill_rows"]), bool(values["fp8_prefill"]),
                          int(values.get("b12x", 0)))

    def _resume(self, prompt: list[int], code: list[int], grid: int | None = None):
        """The longest snapshot of a strict prefix of ``prompt`` whose draft caches fit the request's drafters (and,
        patches/0080, made by the request's prefill mode: ``grid``, default the engine's current one)."""

        _, mtp, dflash = self._drafters(code)
        grid = self._grid() if grid is None else grid
        best = None
        for snap in self.cache:
            fits = (not dflash or snap.drafter_end == len(snap.ids)) and (not mtp or snap.mtp_len >= 0) \
                and snap.grid == grid
            if fits and len(snap.ids) < len(prompt) and prompt[:len(snap.ids)] == snap.ids and (
                    best is None or len(snap.ids) > len(best.ids)):
                best = snap
        return best

    def _remember(self, snap) -> None:
        self.cache = [c for c in self.cache if len(c.ids) < len(snap.ids) and snap.ids[:len(c.ids)] == c.ids]
        self.cache = self.cache[-1:] + [snap]

    def _run(self, prompt: list[int], max_tokens: int, sampling, stop_eos: bool, on_tokens: Callable[[list[int]], Any],
             code: list[int], hit, draft: bool, sess=None, vis=None) -> dict[str, Any]:
        from .decode import (DepthPolicy, DrafterChoice, auto_decode, dflash_decode, mtp_decode, prefill,
                             serial_decode, take_snapshot)

        auto, use_mtp, use_dflash = self._drafters(code)
        drafter = self.drafter if use_dflash else None
        t0 = time.perf_counter()
        store = getattr(self, "store", None)        # patches/0110
        if store is not None and sess is not None and sess.entry is not None:
            # a stored session into the live caches; the live snapshots no longer match them
            hit = store.restore(sess.entry, drafter)
            self.cache = []
        elif store is not None and sess is not None and sess.disk is not None:
            # patches/0250: a session read from disk into the live caches (None: a rank failed, both prefill cold)
            hit = store.restore_disk(sess, drafter)
            self.cache = []
        # a request writes the attention caches from its resume point on: every longer snapshot is overwritten
        cut = len(hit.ids) if hit is not None else 0
        self.cache = [c for c in self.cache if len(c.ids) <= cut]
        from . import pfgrid

        # patches/0540: the prompt's snapshot strictly before its end (``decode._prefill``: ``e.snap_before`` = the
        # length of the prompt it applies to, 0: none)
        before = draft and pfgrid.before_end()
        self.e.snap_before = len(prompt) if before else 0
        if store is not None:                       # patches/0110
            store.begin(hit, cut)
            self.e.checkpoints = sess.marks if sess is not None and draft else ()
        from . import vision as vision_mod

        try:
            with vision_mod.active(vis):            # patches/0500: the prompt's image rows (None: none)
                first = prefill(self.e, prompt, sampling, mtp=use_mtp, drafter=drafter, resume=hit)
        finally:
            self.e.checkpoints = ()
        prefill_s = time.perf_counter() - t0
        self.e.snap_before = 0
        grid = self._grid()                 # patches/0080: 0, or the fast chunk grid this prefill ran on
        saves = list(getattr(self.e, "mark_snaps", ())) if draft else []       # patches/0110
        if draft and (grid or before):
            if self.e.fast_snap is not None:        # the state at the prompt's last grid point (0540: before its end)
                self._remember(self.e.fast_snap)
                saves.append(self.e.fast_snap)
        elif draft:
            snap = take_snapshot(self.e, prompt, self.e.last_hidden if use_mtp else None, mtp=use_mtp,
                                 drafter=drafter)
            self._remember(snap)
            saves.append(snap)
        self.e.mark_snaps = []
        if store is not None:
            self._store_save(saves)                 # patches/0110
        stats: dict[str, Any] = {"prefill_s": prefill_s, "cached": cut}
        if store is not None and sess is not None and sess.entry is not None:
            stats["restored"] = sess.entry.id       # patches/0300: resumed from the session store (RAM)
        elif store is not None and sess is not None and sess.disk is not None and hit is not None:
            stats["restored_disk"] = sess.disk.id   # patches/0300: from its NVMe tier
        if store is not None and sess is not None and sess.disk is not None:
            stats["disk"] = dict(store.disk.last)   # patches/0250: the read (bytes, s, GB/s, pages read / skipped)
        if grid:
            stats["fast_prefill"] = grid - grid % 64            # patches/0083: the chunk grid C of the tag
            from . import pfgrid

            if pfgrid.is_fp8(grid):                              # patches/0240: not every offset is FP8
                stats["fp8_prefill"] = 1
            if grid % 64 // pfgrid.B12X:
                stats["b12x"] = grid % 64 // pfgrid.B12X             # patches/0240: the b12x kernels it ran
            stats["fast_prefill"] = getattr(self.e, "fast_rows", 0) or stats["fast_prefill"]   # patches/0085: its C
        on_tokens([first])
        if max_tokens <= 1 or (stop_eos and first in self.eos):
            return stats
        policy = decode_policy(code)
        # patches/0020: the request's lookup drafter (lookup.py), read by auto_decode; None: no lookup drafts
        self.e.lookup = lookup_for(code, prompt, self.costs, eos=self.eos, stop_eos=stop_eos)
        # patches/0070: rank 0 only; patches/0090: while the request's calib_online is on
        self.e.calib = self.online.observe if self.online is not None and self.calib_on else None
        # patches/0071: cost-derived depths for om / of, and for auto when the header's flag says so
        self.e.depth = None
        if policy is not None and (code[0] == depth_mod.OPT_KIND or (auto and getattr(self, "depth_cost", 0))):
            most_m = code[1] if code[0] == depth_mod.OPT_KIND else MAX_ROWS - 1
            most_f = code[1] if code[0] == depth_mod.OPT_KIND else self.f_most
            self.e.depth = depth_mod.DepthOptimizer(self.costs, most_m=most_m, most_f=most_f)
            if self.e.lookup is not None:
                self.e.lookup.opt = self.e.depth
        if policy is None:
            res = serial_decode(self.e, first, max_tokens, sampling, stop_eos=stop_eos, on_tokens=on_tokens)
        elif code[0] == LOOKUP_KIND:        # patches/0020: lookup drafts, MTP drafts in the rounds without a match
            greedy = sampling is None or sampling.temperature <= 0
            m_policy = DepthPolicy(3, fixed=True, confidence=0.35) if greedy else DepthPolicy(3, low=0.6, high=0.85)
            res = auto_decode(self.e, None, first, max_tokens, sampling, choice=None, m_policy=m_policy,
                              f_policy=m_policy, stop_eos=stop_eos, on_tokens=on_tokens)
        elif auto:
            greedy = sampling is None or sampling.temperature <= 0
            m_policy = DepthPolicy(3, fixed=True, confidence=0.35) if greedy else DepthPolicy(3, low=0.6, high=0.85)
            _, explore, every, margin, sampled_too = policy
            choice = None
            if drafter is not None and (greedy or sampled_too):
                choice = DrafterChoice(self.costs, first="f" if greedy else "m", explore=explore, every=every,
                                       margin=margin)
            res = auto_decode(self.e, drafter, first, max_tokens, sampling, choice=choice, m_policy=m_policy,
                              f_policy=DepthPolicy(self.f_most, fixed=True, confidence=0.3), stop_eos=stop_eos,
                              on_tokens=on_tokens)
        elif use_dflash:
            res = dflash_decode(self.e, self.drafter, first, max_tokens, sampling, policy=policy, stop_eos=stop_eos,
                                on_tokens=on_tokens)
        else:
            res = mtp_decode(self.e, first, max_tokens, sampling, policy=policy, stop_eos=stop_eos,
                             on_tokens=on_tokens)
        self.e.depth = None                 # patches/0071: the optimizer is the request's
        if draft and policy is not None and res.keeps and not grid:     # patches/0080: fast: no reply snapshot
            committed = list(prompt) + res.tokens[:self.e.st.pos - len(prompt)]
            if auto or code[0] == LOOKUP_KIND:     # patches/0020: lN runs auto_decode too
                pending = res.pending
            else:
                pending = None if use_dflash else self.e.main_hidden(slice(0, res.keeps[-1]))
            snap = take_snapshot(self.e, committed, pending, mtp=use_mtp, drafter=drafter)
            self._remember(snap)
            if store is not None:
                self._store_save([snap])            # patches/0110
        elif policy is None:
            self.cache = [c for c in self.cache if len(c.ids) <= len(prompt)]   # the reply's rows are not kept
        stats.update(decode_s=res.seconds, rounds=res.rounds, min_rows=1 + min(res.depths, default=0),
                     tokens_per_round=round((len(res.tokens) - 1) / max(res.rounds, 1), 3),
                     sha256=hashlib.sha256(json.dumps(res.tokens).encode()).hexdigest()[:16])
        if res.arms:
            stats.update(drafters=res.arms, keeps=res.keeps)
        if res.depths:
            stats["depths"] = res.depths        # patches/0380: drafts verified a round (keeps are the rows kept)
        if res.stages:
            stats["stages_ms"] = {k: round(v * 1e3, 1) for k, v in res.stages.items()}
        return stats

    def _vision_table(self, prompt: list[int]):
        """patches/0500: rank 0, before a request is queued or shared: its images' rows as a ``vision.Table`` (the
        calling thread's ``request.vision``, a ``vision_prep.Request``), or None for a prompt without image rows."""

        from .vision_prep import has_vids

        req = getattr(self.request, "vision", None)
        if not has_vids(prompt):
            return None
        if req is None or self.vision is None:
            raise ValueError("a prompt with image rows needs GLM53_TF_VISION=1 on rank 0 and the request's images")
        return self.vision.table(req)

    def generate(self, prompt: list[int], max_tokens: int, sampling, on_tokens, draft: bool = True) -> dict[str, Any]:
        """Rank 0: one request, mirrored by rank 1 (``follow``). ``draft=False``: serial decoding and a fresh
        prefill, the reference drafted replies must equal."""

        vis = self._vision_table(prompt)     # patches/0500: the request's images encoded (rank 0), or None
        if self.batch is not None:          # patches/0120: the request's knobs, policy and priority travel with it
            return self.batch.generate(prompt, max_tokens, sampling, on_tokens, draft, vis=vis)
        # patches/0090: the request's knobs (validated), everything else at this rank's current (load-time) values
        asked = self.parse_knobs(getattr(self.request, "knobs", None))
        if len(prompt) >= self.limit:
            raise ValueError(f"prompt of {len(prompt)} tokens: this engine serves contexts up to {self.limit}")
        max_tokens = max(1, min(int(max_tokens), self.limit - len(prompt)))
        if not draft or self.serial_only:
            spec = "0"
        else:
            spec = getattr(self.request, "policy", None) or self.policy
        code = encode_policy(spec)
        # patches/0071: ``o``, ``om`` / ``of``, or ``auto`` with GLM53_TF_DEPTH=cost choose depths from the costs
        depth_cost = asked["depth"] == "cost" if "depth" in asked else depth_mod.env_cost()     # patches/0090
        cost = int(str(spec).strip() == "o" or code[0] == depth_mod.OPT_KIND
                   or (code[0] in (4, 5) and depth_cost))
        code = self._effective(code)
        if cost:
            code = depth_mod.as_cost(code)
        # patches/0020: lookup settings travel in the header, so both ranks draft alike whatever their environment
        code = with_lookup(code, self.w.mtp is not None, encode_policy(DFLASH_POLICY))
        from .lookup import env_settings

        look = list(env_settings())                 # patches/0090: the request's lookup knobs replace the env's
        look = [int(asked.get("lookup", look[0])), int(asked.get("lookup_min", look[1]))]
        if code[0] in (4, 5) and len(code) >= 6:
            code = code[:4] + look + code[6:]
        values = dict(self._knob_state(), **{k: v for k, v in asked.items() if k in knobs_mod.HEADER})
        if not self.longctx_buffers:                # no long-context graph buffers: that path stays off
            values["longctx_graphs"] = 0
        if values["calib_online"] and self.online is None and self.rank == 0:
            from .calib import OnlineCosts

            self.online = OnlineCosts(self.base_costs["verify"])
        stop_eos = bool(getattr(self.request, "stop_eos", True))
        # patches/0091: a snapshot made by the request's own prefill mode (exact, or fast on the request's grid)
        grid = self._request_grid(values)           # patches/0360: with the request's b12x bits
        hit = self._resume(list(prompt), code, grid) if draft else None
        sess = None
        if self.store is not None:          # patches/0110: a stored session that resumes more, and the marks
            _, need_mtp, need_f = self._drafters(code)
            sess = self.store.plan(list(prompt), grid, need_mtp, need_f and self.drafter is not None,
                                   len(hit.ids) if hit is not None else 0, draft)
            if sess.entry is not None:
                hit = sess.entry.payload[0]
            elif sess.disk is not None:     # patches/0250: from disk; the read starts now, before the header travels
                hit = None
                self.cache = []
                self.store.prefetch(sess)
        # patches/0070: frozen for the request; patches/0090: only while its calib_online is on
        table = self.online.table() if self.online is not None and values["calib_online"] else []
        seed = (sampling.seed if sampling else 0) & 0xFFFFFFFFFFFFFFFF
        resumed = len(hit.ids) if hit is not None else (len(sess.disk.ids) if sess is not None and sess.disk is not None
                                                        else 0)                         # patches/0250
        header = [max_tokens, int(stop_eos), int(draft), resumed,
                  seed & 0x7FFFFFFF, (seed >> 31) & 0x7FFFFFFF, seed >> 62,
                  *_f64_ints(sampling.temperature if sampling else 0.0), int(sampling.top_k) if sampling else 0,
                  *_f64_ints(sampling.top_p if sampling else 1.0), len(table), *table, cost,
                  *knobs_mod.encode(values)] + code          # patches/0090: the knobs rank 1 applies
        self._share(header)
        self._share(list(prompt))
        from . import vision as vision_mod

        vis = vision_mod.exchange(self, prompt, vis)        # patches/0500: the image rows to rank 1 (none: nothing)
        if self.store is not None:          # patches/0110: the plan, and rank 0's store digest for rank 1 to check
            self._share(sess.encode(self.store.index.digest()))
        self._adopt(table)
        self.depth_cost = cost
        from . import cpupin, decode_overlap

        # patches/0370: this request's thread decodes on the serving core (GLM53_TF_CPU_PIN); with
        # GLM53_TF_DECODE_OVERLAP ``emit`` the HTTP callback runs on an emitter thread, not inside the decode loop
        emitter = decode_overlap.Emitter(on_tokens) if decode_overlap.env().emit else None
        try:
            with cpupin.serving_request(), self._knobs(values):
                stats = self._run(list(prompt), max_tokens, sampling, stop_eos, emitter or on_tokens, code, hit,
                                  draft, sess, vis)
        except BaseException:
            if emitter is not None:
                emitter.close(raise_error=False)
            raise
        if emitter is not None:
            emitter.close()                 # every token delivered (or the callback's exception) before the stats
        stats.update(policy=spec, drafts=draft)
        if vis is not None and self.vision is not None:
            stats["vision"] = dict(self.vision.last)       # patches/0500
        if self.store is not None:          # patches/0110
            stats["sessions"] = self.store.describe()
        stats["tf_knobs"] = dict(values, lookup=look[0], lookup_min=look[1],
                                 depth="cost" if depth_cost else "threshold")
        if table:
            stats["verify_ms"] = [round(v, 2) for v in self.costs["verify"]]
        if not values["prefill_rows"]:                  # patches/0085: the header carries "auto" as 0
            stats["tf_knobs"]["prefill_rows"] = "auto"
        return stats

    def follow(self) -> None:
        """Rank 1: mirror every request rank 0 serves, forever."""

        from tensorfold.engine.exact_sampling import Sampling

        if self.batch is not None:
            return self.batch.follow()
        from . import cpupin

        cpupin.serving()                    # patches/0370 (GLM53_TF_CPU_PIN)
        while True:
            cpupin.tick()
            max_tokens, stop_eos, draft, cached, s_lo, s_hi, s_top, t_lo, t_hi, top_k, p_lo, p_hi, n, *rest = \
                self._share(None)
            table, code = rest[:n], rest[n:]            # patches/0070: rank 0's cost table, then the policy code
            self.depth_cost, code = code[0], code[1:]  # patches/0071: rank 0's cost-depth flag
            values, code = knobs_mod.decode(code)       # patches/0090: rank 0's knobs for this request
            prompt = self._share(None)
            from . import vision as vision_mod

            vis = vision_mod.exchange(self, prompt, None)        # patches/0500: rank 0's image rows, if any
            sess = self.store.follow(self._share(None)) if self.store is not None else None     # patches/0110
            self._adopt(table)
            temperature = _ints_f64(t_lo, t_hi)
            seed = (s_top << 62) | (s_hi << 31) | s_lo
            sampling = Sampling(seed, temperature, top_k, _ints_f64(p_lo, p_hi)) if temperature > 0 else None
            hit = None
            if cached and sess is not None and sess.entry is not None:          # patches/0110
                hit = sess.entry.payload[0]
                if len(hit.ids) != cached or prompt[:cached] != hit.ids:
                    raise RuntimeError(f"rank 1's session entry does not hold the {cached} tokens rank 0 resumes from")
            elif cached and sess is not None and sess.disk is not None:        # patches/0250: read it from disk
                if len(sess.disk.ids) != cached or prompt[:cached] != sess.disk.ids:
                    raise RuntimeError(f"rank 1's disk entry does not hold the {cached} tokens rank 0 resumes from")
                self.cache = []
                self.store.prefetch(sess)
            elif cached:
                hit = next((c for c in self.cache if len(c.ids) == cached and prompt[:cached] == c.ids), None)
                if hit is None:
                    raise RuntimeError(f"rank 1 has no snapshot of the {cached} tokens rank 0 resumes from")
            with self._knobs(values):
                self._run(prompt, max_tokens, sampling, bool(stop_eos), lambda new: None, code, hit, bool(draft),
                          sess, vis)
