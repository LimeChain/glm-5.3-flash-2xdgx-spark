"""GLM-5.3-Flash decode on CUDA: prefill, serial decoding, and MTP-drafted decoding byte-identical to it.

Every emitted token is the keyed sample (``tensorfold.engine.exact_sampling``: seeded Gumbel over top-k/top-p, ties
by token id) of this engine's logits at its position, so a drafted round keeps a draft exactly when it equals
what serial decoding samples there. A round verifies the pending token and up to ``depth`` MTP drafts as one
chain window, keeps rows up to the first mismatch (``forward.commit``), then the MTP head absorbs the kept
positions and chains the next drafts. With two ranks each holds half of the vocabulary: both gather every
row's top candidates and draw with the same rule, so both ranks agree without a broadcast.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch

from tensorfold.engine.exact_sampling import MARGIN, Sampling, choose_rows

from . import comm as comm_mod, draftvocab, fastpf, fp8pf, lean as lean_mod, pfglue, pfgrid, profile
from . import decode_overlap as dover                                       # patches/0370
from . import vsplit                                                        # patches/0510
from . import gpuround, gpusample                                            # patches/0450
from . import dump as dump_mod                                               # patches/0430
from .forward import Buffers, State, chunks_for, commit, compute, stage
from .mtp import mtp_compute, mtp_stage
from .weights import Weights


def sample_rows(w: Weights, logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None,
                offset: int | None = None, probs: list[float] | None = None, draft: bool = False) -> list[int]:
    """Rows of (this rank's vocabulary slice of) logits at their absolute positions -> tokens, same on all ranks.
    patches/0420: ``draft`` logits over the listed rows (``draftvocab``) map their columns through the list; the
    target's rows (``draft`` False) never do."""

    R = logits.shape[0]
    greedy = sampling is None or sampling.temperature <= 0
    k = 1 if greedy else min(logits.shape[1], int(sampling.top_k) + MARGIN)
    if probs is not None and greedy:
        k = min(logits.shape[1], 20 + MARGIN)       # the draft's confidence needs its competitors too
    vals, ids = torch.topk(logits.float(), k, dim=-1)
    if draft and offset is None:                    # patches/0420: token ids, before the exchange and the draw
        ids = draftvocab.ids_of(w, ids, logits.shape[1])
    else:
        ids = (ids + (w.vocab_offset if offset is None else offset)).to(torch.int32)
    world = 1 if w.comm is None else w.world
    if gpuround.sample_on(w) and gpusample.fits(k, world, sampling, probs is not None):
        return _sample_device(w, vals, ids, positions, sampling, k, world, probs)      # patches/0450
    if w.comm is None:
        values = vals.cpu().numpy().astype(np.float32)
        tokens = ids.cpu().numpy().astype(np.int64)
    else:
        packed = torch.cat([vals, ids.view(torch.float32)], dim=1).contiguous()
        got = torch.empty((w.world * packed.numel(),), dtype=torch.float32, device=logits.device)
        comm_mod.fast_gather(w.comm, packed.view(-1), got)      # patches/0230
        g = got.view(w.world, R, 2 * k).cpu()
        comm_mod.check(w.comm)                   # a failed RoCE exchange of this step raises before any token
        values = torch.cat([g[r, :, :k] for r in range(w.world)], dim=1).numpy().astype(np.float32)
        tokens = torch.cat([g[r, :, k:].contiguous().view(torch.int32) for r in range(w.world)], dim=1).numpy()
        tokens = tokens.astype(np.int64)
    if greedy:
        order = np.lexsort((tokens, -values), axis=-1)
        chosen = [int(tokens[i, order[i, 0]]) for i in range(R)]
    else:
        chosen = choose_rows(values, tokens, positions, sampling)
    if probs is not None:
        probs.extend(_probability(values, tokens, chosen, sampling))
    return chosen


LP_TOP = 20     # patches/9001: alternatives computed per row (OpenAI / vLLM cap top_logprobs at 20)


def logprob_rows(w: Weights, logits: torch.Tensor, chosen: Sequence[int]) -> list[tuple[float, list[tuple[int, float]]]]:
    """patches/9001: each row's natural-log probability of its ``chosen`` token and the LP_TOP most likely tokens,
    from the raw logits over the WHOLE vocabulary (both ranks' halves: a log-sum-exp and a top-k per rank, one
    all-gather). Temperature, top-k and top-p are not applied (vLLM's default ``raw_logprobs``). Reads the logits
    only: the tokens are the sampler's. Every rank must call it for the same rows (it is a collective)."""

    R = int(logits.shape[0])
    x = logits[:R].float()
    V = int(x.shape[1])
    k = min(LP_TOP, V)
    lse = torch.logsumexp(x, dim=-1, keepdim=True)
    tv, ti = torch.topk(x, k, dim=-1)
    ti = (ti + w.vocab_offset).to(torch.int32)
    c = torch.tensor([int(t) for t in chosen], dtype=torch.int64, device=x.device) - w.vocab_offset
    inside = (c >= 0) & (c < V)
    cv = x.gather(1, c.clamp(0, V - 1)[:, None])
    cv = torch.where(inside[:, None], cv, torch.full_like(cv, float("-inf")))
    packed = torch.cat([lse, cv, tv, ti.view(torch.float32)], dim=1).contiguous()
    width = 2 + 2 * k
    if w.comm is None:
        world = 1
        g = packed.view(1, R, width).cpu()
    else:
        world = w.world
        buf = torch.empty((world * packed.numel(),), dtype=torch.float32, device=x.device)
        comm_mod.fast_gather(w.comm, packed.view(-1), buf)
        g = buf.view(world, R, width).cpu()
        comm_mod.check(w.comm)
    lses = torch.logsumexp(g[:, :, 0].double(), dim=0)                    # [R]
    cvs = g[:, :, 1].double().max(dim=0).values                           # [R]
    vals = torch.cat([g[r, :, 2:2 + k] for r in range(world)], dim=1).double().numpy()
    ids = torch.cat([g[r, :, 2 + k:].contiguous().view(torch.int32) for r in range(world)], dim=1).numpy()
    out = []
    for i in range(R):
        order = np.lexsort((ids[i], -vals[i]))[:k]
        z = float(lses[i])
        tops = [(int(ids[i][j]), min(0.0, float(vals[i][j]) - z)) for j in order]
        out.append((min(0.0, float(cvs[i]) - z), tops))
    return out


def _sample_device(w: Weights, vals: torch.Tensor, ids: torch.Tensor, positions: Sequence[int], sampling,
                   k: int, world: int, probs: list[float] | None) -> list[int]:
    """patches/0450 (GLM53_TF_GPU_ROUND ``sample``): ``sample_rows``' draw on the GPU (``gpusample``: the host's bits);
    the host reads the tokens (and the probabilities) only."""

    R = vals.shape[0]
    packed = torch.cat([vals, ids.view(torch.float32)], dim=1).contiguous().view(-1)
    if w.comm is None:
        got = packed
    else:
        got = torch.empty((world * packed.numel(),), dtype=torch.float32, device=vals.device)
        comm_mod.fast_gather(w.comm, packed, got)           # patches/0230
    want = probs is not None
    tok, prob = gpusample.choose(got, packed.numel(), world,
                                 [(r * 2 * k, k, int(positions[r]), sampling, want) for r in range(R)], want=want)
    host = (torch.cat([tok.to(torch.float64), prob]) if want else tok).cpu()
    comm_mod.check(w.comm)                   # a failed RoCE exchange of this step raises before any token
    if want:
        probs.extend(float(p) for p in host[R:].tolist())
        return [int(t) for t in host[:R].tolist()]
    return [int(t) for t in host.tolist()]


def _probability(values: np.ndarray, tokens: np.ndarray, chosen: list[int], sampling: Sampling | None) -> list[float]:
    """Each row's probability of its chosen token under the top-k / top-p distribution the sampler draws from
    (the draft's own confidence; greedy uses temperature 1 over the candidates)."""

    temp = sampling.temperature if sampling is not None and sampling.temperature > 0 else 1.0
    top_p = sampling.top_p if sampling is not None else 1.0
    top_k = sampling.top_k if sampling is not None and sampling.top_k else values.shape[1]
    out = []
    for i, tok in enumerate(chosen):
        order = np.lexsort((tokens[i], -values[i]))[:top_k]
        v = values[i][order].astype(np.float64) / temp
        p = np.exp(v - v.max())
        p /= p.sum()
        if 0.0 < top_p < 1.0:
            keep = int(np.searchsorted(np.cumsum(p), top_p) + 1)
            p = p[:keep] / p[:keep].sum()
            order = order[:keep]
        ids = tokens[i][order]
        hit = np.nonzero(ids == tok)[0]
        out.append(float(p[hit[0]]) if len(hit) else 0.0)
    return out


class Engine:
    """Weights, one sequence's state, and buffers for windows (main model and MTP head)."""

    def __init__(self, w: Weights, *, capacity: int = 2560, max_rows: int = 8, prefill_rows: int = 64,
                 graphs: bool = False, graph_rows: tuple[int, ...] = (1, 2, 3, 4), long_context: bool = False,
                 taps: tuple[int, ...] = ()) -> None:
        self.w = w
        w.meta["long_context"] = long_context
        # patches/0050, GLM53_TF_LONGCTX_GRAPHS (default 1): past 2,051 tokens, score only the pools that exist and
        # replay CUDA graphs per (rows, parity, pool bucket); 0 is upstream's path (eager, every pool scored)
        self.longctx = long_context and os.environ.get("GLM53_TF_LONGCTX_GRAPHS", "1").strip() != "0"
        w.meta["longctx_bound"] = self.longctx
        from .latent import enabled as latent_kv

        w.meta["latent_kv"] = latent_kv()        # GLM53_TF_LATENT_KV (patches/0060): before any State or Buffers
        from .latent import kv_dtype

        w.meta["kv_fp8"] = kv_dtype() == "fp8"   # GLM53_TF_KV_DTYPE (patches/0220): the latent rows' format
        # patches/0082 (GLM53_TF_LEAN_PREFILL=1): past GLM53_TF_LEAN_BLOCK rows, the window buffers stay at the block
        # and fast-prefill chunks of up to ``prefill_rows`` rows run through the lean set (``lean.py``: sub-blocks
        # on these buffers, the routed experts over the whole chunk; the same bits)
        chunk_rows = prefill_rows
        lean_rows = 0
        if lean_mod.enabled() and prefill_rows > lean_mod.block():
            lean_rows, prefill_rows = prefill_rows, lean_mod.block()
        rows = max(max_rows, prefill_rows)
        self.rows = rows
        self.prefill_rows = chunk_rows
        self.buf = Buffers(w, rows, capacity)
        if taps:
            self.buf.set_taps(tuple(taps), w.cfg.hidden)         # before any graph capture
        self.mbuf = Buffers(w, rows, capacity) if w.mtp is not None else None
        if self.longctx:
            from .sparse import LongScratch

            c = w.cfg
            for bb in (self.buf, self.mbuf):
                if bb is not None:
                    bb.lc = LongScratch(max_rows, capacity, c.heads // w.world, c.qk_dim, w.device)
        self.long_rows = tuple(range(1, max_rows + 1)) if self.longctx else ()
        # patches/0290 (GLM53_TF_KV_POOL_TOKENS, default off): every State's capacity-sized caches on one page pool
        from . import kvpool

        w.meta["kv_pool"] = kvpool.build(w, capacity, limit=capacity - max_rows)     # patches/0490: limit
        self.st = State(w, capacity, rows)
        self.lean = lean_mod.LeanBuffers(w, lean_rows, rows, len(taps)) if lean_rows > rows else None
        self.rows_from = None            # the lean set when the last main-model forward was a lean chunk
        self.prefill_max = max(rows, lean_rows)      # the largest prefill chunk (patches/0090: tf_knobs.prefill_rows)
        if self.lean is not None and w.rank == 0:
            print(f"[tensorfold] lean prefill: window buffers of {rows} rows, fast chunks of up to {lean_rows} rows "
                  f"in sub-blocks of {rows} ({self.lean.nbytes() / 2**30:.2f} GiB of chunk rows)", flush=True)
        # patches/0080: prefill chunks through the fast kernels (``fastpf``: chunks and snapshots on the chunk grid);
        # ``fast_snap``: the grid snapshot the last fast prefill left (None: none)
        self.fast_prefill = fastpf.enabled()
        self.fp8_prefill = fp8pf.default()          # patches/0083: fast chunks on the FP8 kernels (fp8pf)
        self.fast_snap: "Snapshot | None" = None
        # patches/0085: where fast snapshots sit (``pfgrid``; both ranks agree, ``fastpf.settings``) and the chunk
        # size the last fast prefill ran (``prefill_rows`` may be ``pfgrid.AUTO``: chosen per prefill)
        self.snap_grid, self.snap_tail = pfgrid.snapshot_grid(), pfgrid.snapshot_tail()
        # patches/0540 (``pfgrid.before_end``): the callers' whole prompt length; its prefill keeps its snapshot
        # strictly before its end (0: none; a piece that ends earlier keeps its snapshot at its end)
        self.snap_before = 0
        self.fast_rows = 0
        self.mark_snaps: list = []
        w.meta["fast_gather16"] = fastpf.gather16()
        if w.rank == 0 and w.meta["latent_kv"]:          # patches/0060
            from .latent import kv_bytes_per_token

            per = kv_bytes_per_token(self.st)
            kvd = "fp8" if w.meta.get("kv_fp8") else "bf16"          # patches/0220
            kp = w.meta.get("kv_pool")
            if kp is not None:                                        # patches/0290: one pool for every slot
                print(f"[tensorfold] latent KV cache ({kvd} rows): {per / 1024:.1f} KB a token a rank, in the KV pool: "
                      f"{kp.npages * kp.page} tokens in pages of {kp.page} ({kp.nbytes() / 2**30:.2f} GB, every slot "
                      f"up to {capacity})", flush=True)
            else:
                print(f"[tensorfold] latent KV cache ({kvd} rows): {per / 1024:.1f} KB a token a rank "
                      f"({capacity} slots, {per * capacity / 2**30:.2f} GB)", flush=True)
        self.last_hidden: torch.Tensor | None = None
        self.draft_n = w.head.n
        self.graphs = None
        from .overlap import install

        install(w)                       # GLM53_TF_COMM (default: nothing); before any graph is captured
        from . import l2pf

        l2pf.install(w)                  # patches/0460: GLM53_TF_L2PF (default: nothing); replaces 0040's prefetcher
        if graphs:
            from .graphs import Graphs

            self.graphs = Graphs(self, graph_rows, graph_rows, self.long_rows)
            self.reset()

    def reset(self) -> None:
        self.st.reset()

    def forward(self, tokens: Sequence[int]) -> torch.Tensor:
        """A step's forward (a CUDA graph when one was captured for its shape): logits [R, V/world]."""

        R = stage(self.w, self.st, self.buf, tokens)
        self.rows_from = None
        dense = self.st.pos + R <= self.w.cfg.dense_limit
        g = self.graphs.main.get((R, self.st.parity)) if self.graphs is not None and dense else None
        if g is not None:
            g.replay()
            return self.buf.logits[:R]
        npb = self._long_bucket(self.st.pos, R)
        if npb is not None:                  # patches/0050: every row past 2050, a graph per pool bucket
            w, st, b = self.w, self.st, self.buf
            self.graphs.long_step(("main", R, st.parity, npb), lambda: compute(w, st, b, R, npb=npb),
                                  lambda n: vsplit.compute_pieces(w, st, b, R, n, npb=npb))   # patches/0510
            return self.buf.logits[:R]
        return compute(self.w, self.st, self.buf, R, nch=chunks_for(self.st, R), host_pos=self.st.pos)

    def _long_bucket(self, pos: int, R: int) -> int | None:
        """patches/0050: the pool bucket of a window at ``pos`` whose rows are all sparse (pos >= 2051), when a
        long-context graph serves it; else None (eager)."""

        g = self.graphs
        if not self.longctx or g is None or R not in g.long_rows or pos < self.w.cfg.dense_limit \
                or self.st.index is None:        # patches/0090: ``longctx`` is the request's knob
            return None
        from .sparse import pool_bucket

        return pool_bucket(pos + R, self.st.index[0][2].shape[0] - 2)

    def mtp(self, next_tokens: Sequence[int], hidden: torch.Tensor) -> torch.Tensor:
        """The MTP head on rows (hidden, next token): logits of the last row [1, V/world] (patches/0420: [1, listed
        rows] while the request drafts over the list)."""

        n = mtp_stage(self.w, self.st, self.mbuf, next_tokens, hidden)
        full = draftvocab.full(self.w, self.st, "mtp")                   # patches/0420: the fallback rule
        width = draftvocab.head_for(self.w, "mtp", full).n
        dense = self.st.mtp_len + n <= self.w.cfg.dense_limit
        graphs = None if self.graphs is None else self.graphs.mtp_full if full else self.graphs.mtp
        # patches/0500: a prefill's image rows (virtual ids) get their rows from ``glue.embed``'s substitution, which a
        # CUDA graph cannot do (a replay would read the ids as vocabulary rows, out of bounds): such rows run eager
        from . import vision

        eager = vision.ACTIVE is not None and vision.has_vids(next_tokens)
        g = graphs.get(n) if graphs is not None and not self.mbuf.zero_first and dense and not eager else None
        if g is not None:
            g.replay()
            return self.mbuf.logits[:1, :width]
        npb = None if self.mbuf.zero_first or eager else self._long_bucket(self.st.mtp_len, n)
        if npb is not None:                  # patches/0050
            w, st, b = self.w, self.st, self.mbuf
            key = ("mtp", n, npb, "full") if full else ("mtp", n, npb)
            self.graphs.long_step(key, lambda: mtp_compute(w, st, b, n, npb=npb, full=full))
            return self.mbuf.logits[:1, :width]
        from .attention import CHUNK

        return mtp_compute(self.w, self.st, self.mbuf, n, nch=-(-(self.st.mtp_len + n) // CHUNK),
                           host_pos=self.st.mtp_len, full=full)

    def sample(self, logits: torch.Tensor, positions: Sequence[int], sampling: Sampling | None, *,
               draft: bool = False, probs: list[float] | None = None) -> list[int]:
        out = sample_rows(self.w, logits, positions, sampling, None, probs, draft=draft)
        if not draft and getattr(self, "lp_want", 0):      # patches/9001: a prefill's first token (both ranks)
            self.lp_last = logprob_rows(self.w, logits, out)
        return out

    def tap_rows(self, n: int) -> torch.Tensor:
        """The last forward's first n rows of DFlash2 taps, concatenated in layer order: [n, taps * D]."""

        if getattr(self, "rows_from", None) is not None:        # patches/0082: after a lean chunk
            return self.rows_from.tapcat[:n]
        return torch.cat([t[:n] for t in self.buf.taps], dim=1)

    def main_hidden(self, rows: slice) -> torch.Tensor:
        """The main model's rows the MTP head reads (after a forward that computed logits): the final-normed rows,
        as vLLM's GLM-5.3 MTP reads them."""

        if getattr(self, "rows_from", None) is not None:        # patches/0082: after a lean chunk
            return self.rows_from.fnormed[rows]
        return self.buf.fnormed[rows]

    def draft_hidden(self, row: int) -> torch.Tensor:
        """The MTP head's own output row a chained draft reads (after an MTP step): its shared_head.norm output."""

        return self.mbuf.fnormed[0:1]


# -- MTP drafts ---------------------------------------------------------------------------------------------------
def absorb(e: Engine, hidden: torch.Tensor, next_tokens: Sequence[int]) -> torch.Tensor:
    """The MTP cache takes positions (main-model hidden rows [n, D], the tokens after them); logits of the last.
    More rows than the head's buffers hold go in chunks (rows never depend on their chunk-mates)."""

    st = e.st
    if st.mtp_drafted:
        st.set_mtp_len(st.mtp_len - st.mtp_drafted)
        st.mtp_drafted = 0
    draftvocab.track(e.w, st, next_tokens)          # patches/0420: the fallback rule's window
    step = e.mbuf.rows
    logits = None
    for s0 in range(0, len(next_tokens), step):
        part = list(next_tokens[s0:s0 + step])
        logits = e.mtp(part, hidden[s0:s0 + step])
        st.set_mtp_len(st.mtp_len + len(part))
    return logits


def draft(e: Engine, hidden: torch.Tensor, next_tokens: Sequence[int], position: int, count: int,
          sampling: Sampling | None, confidence: float = 0.0, opt=None) -> list[int]:
    """Absorb the kept positions, then chain up to ``count`` drafts for positions position, position + 1, ...
    ``confidence`` > 0: stop once the product of the drafts' own probabilities falls below it (the first draft is
    always kept), so a verify row is spent only on a draft likely to be accepted. ``opt`` (patches/0071, a
    ``depth.DepthOptimizer``): instead, each draft joins the window and the chain goes on while that pays at the
    request's rate."""

    st = e.st
    logits = absorb(e, hidden, next_tokens)
    drafts: list[int] = []
    n = len(next_tokens)
    chain = 1.0
    if opt is not None:                                         # patches/0071
        opt.mtp_begin()
        for j in range(count):
            probs: list[float] = []
            d = e.sample(logits[:1], [position + j], sampling, draft=True, probs=probs)[0]
            take, more = opt.mtp_next(j, probs[0], count)
            if not take:
                break
            drafts.append(d)
            if not more:
                break
            prev = e.draft_hidden(n - 1 if j == 0 else 0)
            logits = e.mtp([d], prev)
            st.set_mtp_len(st.mtp_len + 1)
            st.mtp_drafted += 1
        return drafts
    for j in range(count):
        probs: list[float] = []
        d = e.sample(logits[:1], [position + j], sampling, draft=True, probs=probs if confidence > 0 else None)[0]
        if confidence > 0 and j > 0 and chain * probs[0] < confidence:
            break
        drafts.append(d)
        if confidence > 0:
            chain *= probs[0]
            if chain < confidence:            # a further draft could not pass either: skip its MTP step
                break
        if j + 1 < count:
            prev = e.draft_hidden(n - 1 if j == 0 else 0)
            logits = e.mtp([d], prev)
            st.set_mtp_len(st.mtp_len + 1)
            st.mtp_drafted += 1
    return drafts


# -- prefix snapshots ---------------------------------------------------------------------------------------------
@dataclass
class Snapshot:
    """The committed state after ``ids``, for a later prompt that extends them. The KDA states and conv windows are
    copied; the attention caches (the model's, the MTP head's, DFlash2's) stay where they are, because a request
    resumed from here writes only past ``len(ids)``. ``pending``: the MTP input rows of the last committed
    positions the head has not absorbed yet (their next tokens come from the new prompt). ``mtp_len`` and
    ``drafter_end``: how far the MTP and DFlash2 caches are valid, -1 when they are not usable. ``grid``
    (patches/0080): 0 for a state the row-invariant kernels built (resumable at any length), C for a fast-prefill
    snapshot on the chunk grid C (``fastpf``: only fast requests with the same grid resume from it). ``window``
    (patches/0110, with the session store on): the DFlash2 drafter's context rows in its sliding window
    (``sessions.capture_window``), so a session restored after other sessions ran drafts as it did."""

    ids: list[int]
    rec: torch.Tensor
    conv: torch.Tensor
    pending: torch.Tensor | None
    mtp_len: int
    drafter_end: int
    grid: int = 0
    window: dict | None = None
    kv: int = 0          # patches/0220: 1 when the attention rows below ``ids`` are FP8 latent rows (GLM53_TF_KV_DTYPE)


def _kv8(e) -> int:
    """patches/0220: 1 when ``e``'s latent caches hold FP8 rows."""

    meta = getattr(getattr(e, "w", None), "meta", None)
    return int(bool(meta.get("kv_fp8"))) if isinstance(meta, dict) else 0


def take_snapshot(e: Engine, ids: Sequence[int], pending: torch.Tensor | None, *, mtp: bool,
                  drafter=None, grid: int = 0, mtp_len: int | None = None) -> Snapshot:
    """``mtp_len`` (patches/0080): the MTP cache's valid length when it is not the committed one (a grid snapshot
    taken between prefill chunks, where the head already absorbed the pending row)."""

    st = e.st
    rec = st.rec[st.cur[0]].clone() if st.cur else st.rec[0].clone()
    if mtp_len is None:
        mtp_len = st.mtp_len - st.mtp_drafted
    capture = getattr(e, "snap_window", None)       # patches/0110: set by the session store
    return Snapshot(list(ids), rec, st.conv.clone(), pending.clone() if pending is not None else None,
                    mtp_len if mtp and pending is not None else -1,
                    drafter.context_end if drafter is not None else -1, grid,
                    capture(drafter) if capture is not None and drafter is not None else None,
                    _kv8(e))


def restore(e: Engine, snap: Snapshot, drafter=None) -> None:
    st = e.st
    if snap.kv != _kv8(e):     # patches/0220: never across latent row formats
        raise ValueError(f"a snapshot of {'fp8' if snap.kv else 'bf16'} latent rows cannot resume on this engine")
    if st.cur:
        st.rec[st.cur[0]].copy_(snap.rec)
    st.conv.copy_(snap.conv)
    st.set_pos(len(snap.ids))
    st.set_mtp_len(max(snap.mtp_len, 0))
    st.mtp_drafted = 0
    if drafter is not None:
        drafter.context_end = snap.drafter_end
        drafter.pos_dev.fill_(snap.drafter_end)


# -- prefill ----------------------------------------------------------------------------------------------------
@torch.no_grad()
def prefill(e: Engine, prompt: Sequence[int], sampling: Sampling | None, *, mtp: bool = True, drafter=None,
            resume: Snapshot | None = None) -> int:
    """``_prefill``; with GLM53_TF_PROFILE set, timed by component (``profile``: CUDA events only, same bits)."""

    if not profile.ON:
        return _prefill(e, prompt, sampling, mtp=mtp, drafter=drafter, resume=resume)
    profile.P.begin()
    try:
        first = _prefill(e, prompt, sampling, mtp=mtp, drafter=drafter, resume=resume)
    except BaseException:
        profile.P.active = False
        raise
    profile.P.end(len(prompt), len(resume.ids) if resume is not None else 0, e.w.rank)
    return first


@torch.no_grad()
def _prefill(e: Engine, prompt: Sequence[int], sampling: Sampling | None, *, mtp: bool = True, drafter=None,
             resume: Snapshot | None = None) -> int:
    """Commit the prompt in chains of up to ``prefill_rows`` rows (the MTP cache absorbing every position whose
    next token is known; a DFlash2 ``drafter`` taking every position's taps), sample the first output token, and
    keep the last hidden row for the first draft. ``resume``: start from that snapshot, whose ids begin the prompt
    (rows never depend on their chunk-mates, so the state ends with the bits of a prefill from the start).

    patches/0080, ``e.fast_prefill``: the main model's chunks run the fast kernels. patches/0085 (``pfgrid``): their
    bits do not depend on the chunking, so the chunks are ``pfgrid.plan``'s: from the resume point, C =
    ``pfgrid.chunk_rows(prefill_rows)`` rows (``prefill_rows`` may be ``pfgrid.AUTO``), cut at the session marks
    (``e.checkpoints``, snapshots in ``e.mark_snaps``) and at the snapshot point; ``e.fast_snap`` becomes the
    snapshot at the prompt's last multiple of ``e.snap_grid`` (or the last chunk's start, ``pfgrid``'s tail rule;
    None when that is 0), tagged by mode only, and ``resume`` must be a fast snapshot of the same mode at a multiple
    of 64 (``pfgrid``: why that keeps resumed == fresh whatever C either prefill used). ``e.fast_rows``: the C this
    prefill ran (0: exact).

    patches/0540, ``e.snap_before`` == len(prompt) (the caller's whole prompt ends here): the snapshot point is the last
    grid point strictly BEFORE n (``pfgrid.plan(before=True)``), so ``e.fast_snap`` is a strict prefix a resend of the
    same prompt resumes from; exact prefills then leave theirs in ``e.fast_snap`` too (None: none worth keeping)."""

    if not prompt:
        raise ValueError("prefill requires at least one token")
    w, st, b = e.w, e.st, e.buf
    use_mtp = mtp and w.mtp is not None
    fast = bool(getattr(e, "fast_prefill", False))
    # patches/0082: fast chunks through the lean set when there is one; an exact chunk never needs more than the main
    # buffers' rows (its rows never depend on their chunk-mates, so a smaller chunk gives the same bits)
    lean = getattr(e, "lean", None) if fast else None
    # patches/0083: an FP8 fast prefill runs other kernels, so its snapshots carry their own tag (G + 1)
    fp8pf.ON = fast and bool(getattr(e, "fp8_prefill", False))
    # patches/0085: tagged by mode (exact 0, fast G, FP8 G + 1), not by the chunk size
    grid = getattr(e, "snap_grid", None) or pfgrid.snapshot_grid()
    # patches/0190: + pfgrid.TC for GLM53_TF_LATENT_TC's arithmetic
    from . import b12xpf                  # patches/0240: + 4 x tf_knobs.b12x's effective bits

    tag = pfgrid.tag(fast, fp8pf.ON, grid, pfglue.latent_tc(getattr(e, "w", None)), b12xpf.tag_bits(getattr(e, "w", None)))
    cap = lean.rows if lean is not None else e.rows
    # patches/0190 (``tf_knobs.mtp_window``): the MTP head absorbs positions from ``mtp_lo`` on (a function of the
    # prompt length only); below it its caches are zeroed. Drafts only: replies never read the head
    mtp_lo = pfglue.mtp_start(len(prompt)) if use_mtp else 0
    e.fast_snap = None
    e.mark_snaps = []
    begin = 0
    if resume is None:
        e.reset()
        if drafter is not None:
            drafter.reset()
    else:
        begin = len(resume.ids)
        if begin >= len(prompt) or list(prompt[:begin]) != resume.ids:
            raise ValueError("a resumed prefill needs a snapshot of a strict prefix of the prompt")
        if (use_mtp and resume.mtp_len < 0) or (drafter is not None and resume.drafter_end != begin):
            raise ValueError("this snapshot's draft caches do not fit the request")
        if not pfgrid.resumable(resume.grid, tag, begin):            # patches/0085: any C, same mode, 64-aligned
            raise ValueError(f"a snapshot of grid {resume.grid} at {begin} cannot resume a prefill of grid {tag}")
        restore(e, resume, drafter)
        if use_mtp:
            k = resume.pending.shape[0]
            # patches/0190: rows below the MTP window's start are skipped (zeroed), drafts only
            pfglue.absorb(e, resume.pending, list(prompt[begin - k + 1:begin + 1]), begin - k, mtp_lo, absorb)
    prof = profile.ON
    if prof:
        profile.lap("resume")
    last = None
    n = len(prompt)
    # patches/0085: the chunk size (a speed choice: the bits do not depend on it), the chunks and the snapshot point
    step = pfgrid.chunk_rows(e.prefill_rows, n - begin, cap, fast)
    e.fast_rows = step if fast else 0
    # patches/0550: the selection scratch sized for this prefill's largest key block before its first chunk (DSA calls
    # of the lean sub-block's rows, else the chunk's), so no key block is allocated inside the chunks
    if getattr(st, "index", None) is not None and n - 1 >= getattr(w.cfg, "dense_limit", n):
        from . import sparse

        if sparse.SELECT == "blocked":
            sparse.reserve_for_prefill(begin, n, b.rows if lean is not None else step, st.index[0][2], w.device)
    # patches/0540 (``pfgrid.before_end``): the caller's whole-prompt prefill keeps its snapshot strictly before n
    # (``e.fast_snap``, exact prefills too), so the same prompt sent again resumes all but its last grid step
    before = bool(getattr(e, "snap_before", 0)) and int(e.snap_before) == len(prompt)
    if before:
        e.snap_before = 0                           # one prefill's (a stale value could only move a snapshot)
    pl = pfgrid.plan(begin, n, step, fast=fast, marks=getattr(e, "checkpoints", ()) or (), grid=grid,
                     tail=getattr(e, "snap_tail", None), before=before)
    snaps = fast or before
    if snaps and pl.snap == begin and resume is not None:
        e.fast_snap = resume                        # nothing new to keep: the resume point stays the prompt's state
    for start, stop in pl.spans:
        chunk = list(prompt[start:stop])
        R = len(chunk)
        if start in pl.marks:                       # patches/0085: extra snapshots asked for (patches/0110's marks)
            e.mark_snaps.append(take_snapshot(e, prompt[:start], e.last_hidden if use_mtp else None, mtp=use_mtp,
                                              drafter=drafter, grid=tag, mtp_len=start - 1))
        if snaps and start == pl.snap and start > begin:    # the state at the snapshot point, before [snap, ...)
            e.fast_snap = take_snapshot(e, prompt[:start], e.last_hidden if use_mtp else None, mtp=use_mtp,
                                        drafter=drafter, grid=tag, mtp_len=start - 1)
        if lean is not None:                        # patches/0082: the same bits in less memory
            staged = lean_mod.stage(w, st, lean, chunk)
        else:
            staged = stage(w, st, b, chunk)
        if prof:
            profile.lap("stage")
            profile.P.chunks += 1
            profile.P.rows += R
        if lean is not None:
            logits = lean_mod.compute(w, st, b, lean, staged, head=stop == n)
        else:
            logits = compute(w, st, b, staged, nch=chunks_for(st, R), host_pos=st.pos, fast=fast, head=stop == n)
        if hasattr(e, "rows_from"):
            e.rows_from = lean                      # patches/0082: main_hidden / tap_rows read the lean rows
        last = logits[-1:].clone()
        e.last_hidden = e.main_hidden(slice(R - 1, R)).clone()
        if prof:
            profile.lap("head")
        if use_mtp:
            nxt = list(prompt[start + 1:start + R + 1])
            if nxt:
                if prof:
                    profile.P.prefix = "mtp."
                pfglue.absorb(e, e.main_hidden(slice(0, len(nxt))), nxt, start, mtp_lo, absorb)   # patches/0190
                if prof:
                    profile.lap("other")
                    profile.P.prefix = ""
        if drafter is not None:
            drafter.add_taps(e.tap_rows(R))
            if prof:
                profile.lap("drafter")
        if dump_mod.DUMP is not None:               # patches/0430: the chunk's rows as training records (reads only)
            dump_mod.prefill_chunk(e, prompt, start, stop, begin)
        if lean is not None:
            lean_mod.commit(w, st, lean, R)
        else:
            commit(w, st, b, R, R)
        if prof:
            profile.lap("commit")
    if fast and pl.snap == n:                       # patches/0080: the prompt ends on the grid
        e.fast_snap = take_snapshot(e, prompt, e.last_hidden if use_mtp else None, mtp=use_mtp, drafter=drafter,
                                    grid=tag)
    first = e.sample(last, [len(prompt)], sampling)[0]
    if prof:
        profile.lap("sample")
    return first


def _spans(begin: int, n: int, step: int, marks: Sequence[int]):
    """Prefill chunks [start, stop) of up to ``step`` rows from ``begin``, also cut at every mark (patches/0110)."""

    start = begin
    for q in [*marks, n]:
        while start < q:
            stop = min(start + step, q)
            yield start, stop
            start = stop


# -- decode loops -----------------------------------------------------------------------------------------------
@dataclass
class DecodeResult:
    tokens: list[int]
    seconds: float
    rounds: int
    drafted: int = 0
    accepted: int = 0
    stages: dict[str, float] = field(default_factory=dict)
    depths: list[int] = field(default_factory=list)
    keeps: list[int] = field(default_factory=list)
    arms: str = ""                          # auto_decode: each round's drafter, "m" MTP, "f" DFlash2, "l" lookup
    pending: torch.Tensor | None = None     # auto_decode: MTP input rows of committed positions not absorbed yet

    @property
    def tokens_per_second(self) -> float:
        return (len(self.tokens) - 1) / self.seconds if self.seconds else 0.0


def _sync(w: Weights) -> None:
    torch.cuda.synchronize()


def _forward_sync() -> None:
    """After a verify forward: the host waits for it here unless GLM53_TF_DECODE_OVERLAP has ``sync`` (patches/0370;
    the sampler's readback then waits for it, and the stages' "forward" / "sample" split moves accordingly)."""

    if not dover.env().sync:
        torch.cuda.synchronize()


@torch.no_grad()
def serial_decode(e: Engine, pending: int, count: int, sampling: Sampling | None, *,
                  stop_eos: bool = False, on_tokens=None) -> DecodeResult:
    """One token a step through the same kernels and sampler; ``pending`` is the first sampled token."""

    w, st, b = e.w, e.st, e.buf
    out = [pending]
    stages = dict(forward=0.0, sample=0.0, commit=0.0)
    _sync(w)
    start = time.perf_counter()
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        t0 = time.perf_counter()
        logits = e.forward([out[-1]])
        _forward_sync()                     # patches/0370
        t1 = time.perf_counter()
        tok = e.sample(logits[:1], [st.pos + 1], sampling)[0]
        t2 = time.perf_counter()
        commit(w, st, b, 1, 1)
        t3 = time.perf_counter()
        stages["forward"] += t1 - t0
        stages["sample"] += t2 - t1
        stages["commit"] += t3 - t2
        out.append(tok)
        if on_tokens is not None:
            on_tokens([tok])
    _sync(w)
    return DecodeResult(out, time.perf_counter() - start, len(out) - 1, stages=stages)


class DepthPolicy:
    """Drafts a round: fixed, or from the running acceptance (the Flash Next engine's rule)."""

    def __init__(self, most: int = 3, fixed: bool = False, low: float = 0.8, high: float = 0.9,
                 confidence: float = 0.0) -> None:
        self.most, self.fixed, self.low, self.high = most, fixed, low, high
        self.confidence = confidence
        self.rate = 0.8

    def next(self, drafted: int, accepted: int) -> int:
        if self.fixed:
            return self.most
        if drafted:
            self.rate = 0.875 * self.rate + 0.125 * (accepted / drafted)
        return max(1, min(self.most, 1 if self.rate < self.low else 2 if self.rate < self.high else 3))


@torch.no_grad()
def mtp_decode(e: Engine, pending: int, count: int, sampling: Sampling | None, *, policy: DepthPolicy | None = None,
               stop_eos: bool = False, on_tokens=None) -> DecodeResult:
    """Verify the pending token and its MTP drafts in one window, keep up to the first mismatch, draft again.
    Starts from the state ``prefill`` left (the MTP cache holds every prompt position but the last)."""

    w, st, b = e.w, e.st, e.buf
    policy = policy or DepthPolicy()
    opt = getattr(e, "depth", None)         # patches/0071 (``om``): cost-derived depths, up to policy.most
    out = [pending]
    stages = dict(draft=0.0, forward=0.0, sample=0.0, commit=0.0)
    rounds = drafted = accepted = 0
    depths: list[int] = []
    keeps: list[int] = []
    _sync(w)
    start = time.perf_counter()
    t0 = time.perf_counter()
    depth = min(policy.next(0, 0), count - len(out))
    drafts = (draft(e, e.last_hidden, [pending], st.pos + 1, depth, sampling, policy.confidence, opt=opt)
              if depth > 0 else [])
    steps, backlog = 1 + st.mtp_drafted, 1
    stages["draft"] += time.perf_counter() - t0
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        t0 = time.perf_counter()
        tokens = [out[-1]] + drafts
        R = len(tokens)
        logits = e.forward(tokens)
        _forward_sync()                     # patches/0370
        t1 = time.perf_counter()
        sampled = e.sample(logits[:R], [st.pos + 1 + r for r in range(R)], sampling)
        keep = 1
        for i, d in enumerate(drafts):
            if sampled[i] != d or (stop_eos and sampled[i] in w.cfg.eos):
                break
            keep += 1
        t2 = time.perf_counter()
        commit(w, st, b, R, keep)
        t3 = time.perf_counter()
        if opt is not None:                 # patches/0071
            opt.record("m", R, steps, backlog, keep)
        rounds += 1
        drafted += len(drafts)
        accepted += keep - 1
        depths.append(len(drafts))
        keeps.append(keep)
        out.extend(sampled[:keep])
        if on_tokens is not None:
            on_tokens(sampled[:keep][:max(0, count - (len(out) - keep))])
        stages["forward"] += t1 - t0
        stages["sample"] += t2 - t1
        stages["commit"] += t3 - t2
        if len(out) >= count or (stop_eos and out[-1] in w.cfg.eos):
            break
        t4 = time.perf_counter()
        depth = min(policy.next(len(drafts), keep - 1), count - len(out))
        drafts = (draft(e, e.main_hidden(slice(0, keep)), sampled[:keep], st.pos + 1, depth, sampling,
                        policy.confidence, opt=opt) if depth > 0 else [])
        steps, backlog = 1 + st.mtp_drafted, keep
        stages["draft"] += time.perf_counter() - t4
    _sync(w)
    return DecodeResult(out[:count], time.perf_counter() - start, rounds, drafted, accepted, stages, depths, keeps)


@torch.no_grad()
def dflash_decode(e: Engine, drafter, pending: int, count: int, sampling: Sampling | None, *,
                  policy: DepthPolicy | None = None, stop_eos: bool = False, on_tokens=None) -> DecodeResult:
    """``mtp_decode`` with DFlash2 drafts: a round verifies the pending token and the drafter's chain for the
    positions after it, keeps up to the first mismatch, and the drafter takes the kept rows' taps. Starts from
    ``prefill(..., drafter=drafter)``."""

    w, st, b = e.w, e.st, e.buf
    policy = policy or DepthPolicy(3, fixed=True)
    opt = getattr(e, "depth", None)         # patches/0071 (``of``): cost-derived depths, up to policy.most
    backlog = 0
    out = [pending]
    stages = dict(draft=0.0, forward=0.0, sample=0.0, commit=0.0)
    rounds = drafted = accepted = 0
    depths: list[int] = []
    keeps: list[int] = []
    _sync(w)
    start = time.perf_counter()
    depth = min(policy.next(0, 0), count - len(out))
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        t0 = time.perf_counter()
        if opt is not None and depth > 0:   # patches/0071: every pick with its probability, then the best depth
            f_probs: list[float] = []
            drafts = drafter.propose(out[-1], depth, sampling, 0.0, probs=f_probs)
            if drafts:
                drafts = drafts[:opt.f_depth(f_probs[:len(drafts)])]
        else:
            drafts = drafter.propose(out[-1], depth, sampling, policy.confidence) if depth > 0 else []
        t1 = time.perf_counter()
        tokens = [out[-1]] + drafts
        R = len(tokens)
        logits = e.forward(tokens)
        _forward_sync()                     # patches/0370
        t2 = time.perf_counter()
        sampled = e.sample(logits[:R], [st.pos + 1 + r for r in range(R)], sampling)
        keep = 1
        for i, d in enumerate(drafts):
            if sampled[i] != d or (stop_eos and sampled[i] in w.cfg.eos):
                break
            keep += 1
        t3 = time.perf_counter()
        commit(w, st, b, R, keep)
        t4 = time.perf_counter()
        drafter.add_taps(e.tap_rows(keep))
        t5 = time.perf_counter()
        if opt is not None:                 # patches/0071
            opt.record("f", R, 0, backlog, keep)
            backlog = keep
        rounds += 1
        drafted += len(drafts)
        accepted += keep - 1
        depths.append(len(drafts))
        keeps.append(keep)
        out.extend(sampled[:keep])
        if on_tokens is not None:
            on_tokens(sampled[:keep][:max(0, count - (len(out) - keep))])
        stages["draft"] += (t1 - t0) + (t5 - t4)
        stages["forward"] += t2 - t1
        stages["sample"] += t3 - t2
        stages["commit"] += t4 - t3
        depth = min(policy.next(len(drafts), keep - 1), count - len(out))
    _sync(w)
    return DecodeResult(out[:count], time.perf_counter() - start, rounds, drafted, accepted, stages, depths, keeps)


# -- drafter chosen per request ------------------------------------------------------------------------------------
def _other(arm: str) -> str:
    return "f" if arm == "m" else "m"


class DrafterChoice:
    """Which drafter a round uses, MTP chains ("m") or DFlash2 blocks ("f"), from the tokens each has committed per
    millisecond in this request. The milliseconds come from ``costs`` (a verify window of R rows, an MTP step, a
    DFlash2 block, the catch-up of rows a drafter missed), timed at load and made the same on both ranks, so both
    ranks choose alike without exchanging anything; committed tokens are the same on both by construction.

    The first ``explore`` rounds use ``first``, the next ``explore`` the other drafter; then the one with the higher
    rate over its last ``window`` rounds, switching only for a rate ``margin`` higher, and one round of the other
    every ``every`` rounds so its rate stays current."""

    def __init__(self, costs: dict, *, first: str, explore: int = 2, every: int = 8, margin: float = 0.03,
                 window: int = 6) -> None:
        self.costs = costs
        self.first = first
        self.explore, self.every, self.margin, self.window = explore, every, margin, window
        self.rounds: list[tuple[str, int, float]] = []       # (drafter, tokens committed, model ms)
        self.choice = first
        self.run = 0

    def cost(self, arm: str, rows: int, steps: int, backlog: int) -> float:
        """Model ms of a round: its verify window, then MTP steps (``steps`` head runs) or a DFlash2 block, plus
        the catch-up of ``backlog`` rows."""

        c = self.costs
        verify = c["verify"][min(rows, len(c["verify"])) - 1]
        if arm == "m":
            return verify + c["mtp"] + c["mtp_step"] * max(steps - 1, 0) + c["mtp_row"] * max(backlog - 1, 0)
        return verify + c["block"] + c["taps_row"] * backlog

    def rate(self, arm: str) -> float | None:
        rs = [r for r in self.rounds if r[0] == arm][-self.window:]
        return sum(r[1] for r in rs) / sum(r[2] for r in rs) if rs else None

    def pick(self) -> str:
        n = len(self.rounds)
        if n < self.explore:
            return self.first
        if n < 2 * self.explore:
            return _other(self.first)
        cur = self.choice
        rc, ro = self.rate(cur), self.rate(_other(cur))
        need = 1.0 if n == 2 * self.explore else 1.0 + self.margin      # no bias at the first choice
        if ro is not None and (rc is None or ro > rc * need):
            self.choice = cur = _other(cur)
            self.run = 0
        if self.every and self.run >= self.every:
            self.run = 0
            return _other(cur)
        self.run += 1
        return cur

    def record(self, arm: str, rows: int, steps: int, backlog: int, keep: int) -> None:
        self.rounds.append((arm, keep, self.cost(arm, rows, steps, backlog)))


@torch.no_grad()
def auto_decode(e: Engine, drafter, pending: int, count: int, sampling: Sampling | None, *,
                choice: DrafterChoice | None,
                m_policy: DepthPolicy, f_policy: DepthPolicy, stop_eos: bool = False, on_tokens=None) -> DecodeResult:
    """Drafted decoding where ``choice`` picks the drafter of each round (MTP only when it is None). Each
    drafter keeps a backlog of the committed rows it has not taken (MTP input rows and their next tokens; DFlash2
    taps) and takes them when it next drafts, so switching costs one catch-up step. Drafts only propose, so the
    reply equals serial decoding whatever the choice. Starts from ``prefill(..., mtp=True, drafter=drafter)``."""

    w, st, b = e.w, e.st, e.buf
    cap = e.rows * 4
    m_rows = torch.empty((cap, w.cfg.hidden), dtype=torch.bfloat16, device=w.device)
    m_rows[:1].copy_(e.last_hidden)
    m_next: list[int] = [pending]
    f_taps = None
    n_f = 0
    if drafter is not None:
        f_taps = torch.empty((cap, len(b.taps) * w.cfg.hidden), dtype=torch.bfloat16, device=w.device)
    out = [pending]
    stages = dict(draft=0.0, forward=0.0, sample=0.0, commit=0.0)
    rounds = drafted = accepted = 0
    depths: list[int] = []
    keeps: list[int] = []
    arms: list[str] = []
    last = {"m": (0, 0), "f": (0, 0)}
    lookup = getattr(e, "lookup", None)     # patches/0020: prompt-lookup drafts (lookup.py), set per request
    timer = getattr(e, "calib", None)       # patches/0070: rank 0's online window costs (calib.py), or None
    opt = getattr(e, "depth", None)         # patches/0071: cost-derived depths (depth.py); None: the policies
    # patches/0370 (GLM53_TF_DECODE_OVERLAP ``sync``): no host wait after the forward; the online calibration reads
    # the forward's time from two CUDA events once the sampler's readback has synchronized
    no_sync = dover.env().sync
    fwd_timer = dover.ForwardTimer() if no_sync and timer is not None else None
    _sync(w)
    start = time.perf_counter()
    while len(out) < count and not (stop_eos and out[-1] in w.cfg.eos):
        room = count - len(out)
        t0 = time.perf_counter()
        look = lookup.plan(out, room) if lookup is not None else None      # patches/0020
        arm = "l" if look else choice.pick() if choice is not None else "m"
        if arm == "l":                      # patches/0020: the history's tokens, no drafter step
            backlog, depth, drafts, steps = 0, len(look), look, 0
        elif arm == "m":
            backlog = len(m_next)
            depth = max(1, min(m_policy.next(*last["m"]) if opt is None else opt.most_m, room))
            drafts = draft(e, m_rows[:backlog], m_next, st.pos + 1, depth, sampling, m_policy.confidence, opt=opt)
            steps = 1 + st.mtp_drafted
            m_next = []
        else:
            backlog = n_f
            if n_f:
                drafter.add_taps(f_taps[:n_f])
                n_f = 0
            if opt is None:
                depth = max(1, min(f_policy.next(*last["f"]), room))
                drafts = drafter.propose(out[-1], depth, sampling, f_policy.confidence)
            else:                           # patches/0071: every pick with its probability, then the best depth
                f_probs: list[float] = []
                drafts = drafter.propose(out[-1], max(1, min(opt.most_f, room)), sampling, 0.0, probs=f_probs)
                if drafts:
                    drafts = drafts[:opt.f_depth(f_probs[:len(drafts)])]
            steps = 0
        t1 = time.perf_counter()
        tokens = [out[-1]] + drafts
        R = len(tokens)
        if fwd_timer is not None:
            fwd_timer.start()
        logits = e.forward(tokens)
        if fwd_timer is not None:
            fwd_timer.stop()
        if not no_sync:
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        if timer is not None and fwd_timer is None:
            timer(R, (t2 - t1) * 1e3)
        sampled = e.sample(logits[:R], [st.pos + 1 + r for r in range(R)], sampling)
        if fwd_timer is not None:           # patches/0370: after the readback, the events are done
            timer(R, fwd_timer.ms())
        keep = 1
        for i, d in enumerate(drafts):
            if sampled[i] != d or (stop_eos and sampled[i] in w.cfg.eos):
                break
            keep += 1
        t3 = time.perf_counter()
        commit(w, st, b, R, keep)
        t4 = time.perf_counter()
        # the kept rows join both backlogs (a full backlog is taken first)
        if len(m_next) + keep > cap:
            absorb(e, m_rows[:len(m_next)], m_next)
            m_next = []
        m_rows[len(m_next):len(m_next) + keep].copy_(e.main_hidden(slice(0, keep)))
        m_next.extend(sampled[:keep])
        if drafter is not None:
            if n_f + keep > cap:
                drafter.add_taps(f_taps[:n_f])
                n_f = 0
            f_taps[n_f:n_f + keep].copy_(e.tap_rows(keep))
            n_f += keep
        t5 = time.perf_counter()
        if choice is not None and arm != "l":
            choice.record(arm, R, steps, backlog, keep)
        if lookup is not None:              # patches/0020
            lookup.record(arm, R, steps, backlog, keep)
        if opt is not None:                 # patches/0071
            opt.record(arm, R, steps, backlog, keep)
        last[arm] = (len(drafts), keep - 1)
        rounds += 1
        drafted += len(drafts)
        accepted += keep - 1
        depths.append(len(drafts))
        keeps.append(keep)
        arms.append(arm)
        out.extend(sampled[:keep])
        if on_tokens is not None:
            on_tokens(sampled[:keep][:max(0, count - (len(out) - keep))])
        stages["draft"] += (t1 - t0) + (t5 - t4)
        stages["forward"] += t2 - t1
        stages["sample"] += t3 - t2
        stages["commit"] += t4 - t3
    _sync(w)
    seconds = time.perf_counter() - start
    if drafter is not None and n_f:
        drafter.add_taps(f_taps[:n_f])          # DFlash2's context ends where the committed rows end
    return DecodeResult(out[:count], seconds, rounds, drafted, accepted, stages, depths, keeps, "".join(arms),
                        m_rows[:len(m_next)])
