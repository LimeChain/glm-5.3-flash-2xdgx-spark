"""Continuous batching for GLM-5.3-Flash's CUDA engine (``GLM53_TF_BATCH=N``, N >= 2; patches/0030, rebuilt on the
current engine by patches/0120): up to N requests decode together, and each round is ONE forward over every
decoding request's verify window.

Why it pays: decode reads the weights once a forward (about 5 GB a rank), and a row from a second sequence costs
what an extra draft row of one sequence costs (mostly the experts it adds). Rows never depend on their
window-mates (``forward.py``), so a round's rows can come from different sequences as long as the work that reads
per-sequence state is done per sequence.

Slots. One ``forward.State`` per slot (slot 0 is the engine's own ``e.st``), the engine's ``Buffers`` shared. Every
slot has the engine's context (latent KV with patches/0060, index rings with patches/0065), its own KDA states, its
own MTP-head cache and CUDA graphs, and its own DFlash2 context (``_drafter_view``: the drafter's weights and
buffers shared, its context caches, positions and graphs per slot). Slots other than 0 keep window-sized KDA
projection / replay rows (``engine.MAX_ROWS``: 8, or patches/0380's GLM53_TF_MAX_DRAFT_ROWS up to 16); a prefill on
such a slot borrows slot 0's (``_borrow``), which only hold the rows of one forward between that forward and its
commit.

A round (``_execute``, both ranks, from rank 0's plan: ``batchplan``):

1. cancels (a client went away; a background request steps aside for a waiting one and runs again later);
2. admissions: a queued request takes a free slot (the one whose kept snapshots resume the longest prefix of its
   prompt, else the least recently used);
3. at most one prefill piece: the next ``GLM53_TF_BATCH_PIECE`` tokens of one admitted prompt (fast prefill:
   rounded to its chunk grid), through ``decode.prefill`` itself (exact, fast, lean, FP8, pipelined: whatever the
   request's knobs say), resumed from the snapshot the previous piece left. Resumed == fresh holds for every
   prefill mode, so a prompt prefilled in pieces has the bits of one prefill. Pieces take at most
   ``GLM53_TF_BATCH_PREFILL_SHARE`` of the time while other requests decode (``batchplan.Fairness``), so a long
   prompt never stalls the others for more than one piece;
4. one verify forward over every decoding request's window (pending token + drafts), then per request, from its own
   rows: keyed sampling (all requests' candidates in ONE all-gather), the accept rule, ``forward.commit`` on its
   state, and its drafter's next drafts (``Stepper``: ``decode.auto_decode``'s round cut at the forward, so MTP,
   DFlash2, lookup, ``auto``'s choice and cost-derived depths run per request exactly as alone).

Exactness. A request's tokens are the keyed samples of its own logits at its own positions. Its logit rows are the
bits of its lone forward (row-local kernels; per-slot kernels see only the slot's state and rows), its state
advances by ``commit`` exactly as alone, its prefill is ``decode.prefill`` (in pieces: resumed == fresh), and the
accept rule only decides how many of its rows to keep; drafts only propose. So each batched reply is
byte-identical to the same request served alone, and to serial decoding, whatever shares its rounds.

Draft depths. ``auto`` / ``o`` requests with cost-derived depths (patches/0071) price a row at the slope of the
verify curve past the other requests' rows, against the aggregate rate of the shared rounds
(``batchplan.RoundCosts``): a draft row that slows every request must buy more than one that slows only its own.

CUDA graphs. Slot 0 alone replays the engine's own graphs. Other rounds replay graphs captured lazily per (slots,
window rows, per-slot context mode: dense or patches/0050's pool bucket), up to ``GLM53_TF_BATCH_MAX_GRAPHS``; the
KDA states of a graphed round are kept in buffer 0 (``_parity0``: a copy of the state when a commit left it in
buffer 1, 71 MB a slot) so parities do not multiply the keys. A key's first round runs eagerly through the
capturable code (its result) and is captured after (as 0050 does). Windows crossing 2,051 tokens run eager.
patches/0510 (``GLM53_TF_BATCH_GRAPHS=lone``): only rounds with one active slot use these graphs; rounds of 2+ slots
run eagerly (``0``'s path). Same kernels either way, so the same bits.

Tensor parallelism. Rank 0 alone decides each round's cancels, admissions (slot, resume length, request header)
and pieces and shares them before the round; rank 1 runs the same rounds in ``follow``.

Sessions (patches/0180, GLM53_TF_SESSION_GIB > 0 with GLM53_TF_BATCH_SESSIONS=1): patches/0110's store behind every
slot. At admission rank 0 looks the prompt up in the store (``SessionStore.plan``: the longest stored prefix of the
request's prefill mode, and the marks its prefill takes); when that resumes more than the free slots' own snapshots,
the request goes to the free slot already holding most of the entry's pages, and ``_admit`` copies the entry into
that slot's caches (restore = copy-in: the slot owns its copy, so later evictions never touch it) and resumes its
first piece from the entry's snapshot. Saves follow the lone engine's rules on the slot's live caches: marks (and a
piece end that is one) right after the piece that took them, the prompt snapshot after the last piece (exact: at the
prompt's end; fast: its last grid point; patches/0540, both: its last grid point strictly before the end), the reply
snapshot when a drafted exact request ends. Rank 0 ships each
admission's session plan (entry, marks, its store digest) right after the admission's prompt, and each save's
decision (stored / skipped / duplicate, evictions) when it happens, both inside the round both ranks run; rank 1
checks the digest and applies the decisions. The store's budget is set aside at load (slots are added while the
reserve AND the budget stay free) and at admission (``batchplan.admit_free``). A fork point is also a prefix shared
with a prompt in flight (running or admitted in the same round): sessions admitted together see no stored entry of
each other. Which side of a save decision a batcher is on is its role (``follow`` = rank 1), never ``g.rank``, as
for every other message of a round. patches/0250 (GLM53_TF_SESSION_DISK): an admission may resume from an entry on
disk instead; it is read into the chosen slot from the moment the plan is known (rank 0 in ``_plan``, rank 1 in
``follow``, when the slot is idle) and finished in ``_admit``, where both ranks agree on the outcome (a failed read on
either: both prefill the prompt from scratch).

Batched parallelism (patches/0200; each knob off by default, the ranks must agree):
``GLM53_TF_BATCH_CAPTURE_AFTER=N`` captures a multi-sequence graph key on its N-th sighting (``batchplan.Sightings``;
keys met once run eagerly through the same capturable code instead of paying a capture); ``GLM53_TF_BATCH_PAD=2,4,8``
pads each slot's window to the next listed size with its last token (``_pad``: fewer keys; padded rows are verified
like drafts that can never be accepted and ``commit`` gets the padded row count, so the kept bits are the unpadded
window's); ``GLM53_TF_BATCH_SHORT=N`` gives every prompt with at most N tokens left its piece in the round it is
admitted, several a round, outside the fair share (``batchplan.pick_pieces``); ``GLM53_TF_BATCH_PARITY_KEY=1`` puts
each slot's KDA parity in the graph key instead of copying a state left in buffer 1 back to buffer 0 before every
graphed round (71 MB a slot and round; every slot flips each round, so the keys only double: the graph bakes in
``rec[cur]`` / ``rec[1 - cur]`` as the engine's own (rows, parity) graphs do). Each request's stats carry
``round_kinds`` (alone / graph / eager / capture rounds, padded rows, and the wall ms of its verify rounds, of its
own drafting, and of other requests' prefill pieces it waited through). ``GLM53_TF_BATCH_MTP=1`` drafts the MTP
chains of all slots together (``MtpChains``: one head pass absorbs every slot's backlog, one pass per chained step
over the slots still drafting; the head's weights read once a pass instead of once a slot).
``GLM53_TF_BATCH_ROW_MS=F`` prices a row past the calibrated verify table at no less than F ms in the batch-aware
cost-derived depths (``batchplan.verify_ms``; the table's flattening top alone, ~4.5 ms, drafts too deep at 3-4
sequences where a row costs 6-10 ms).

Round buckets (patches/0280, off by default): ``GLM53_TF_BATCH_BUCKETS=4,8`` pads EVERY window of a multi-slot round
(graph path only) to the smallest listed size >= the round's longest window (``_bucket``; the same padding rules as
``GLM53_TF_BATCH_PAD``), so a round's key is (slots, bucket, modes, parities) instead of one row count a slot: with
every slot flipping its KDA parity each round, 4 decoding slots meet ~2 keys a bucket and nearly every round replays a
graph. ``GLM53_TF_BATCH_PAD_TIE`` (auto: on with buckets) makes a padded row route as its window's last real row
(``Buffers.route_src``: the row's router logits replaced by that row's before top-k), so padded rows add no expert
reads (the ~6 ms a row of a real row); padded rows are never kept, and every real row's bits are unchanged (its own
logits; row-independent kernels). The depth model keeps pricing the real rows only.

Per-slot drafter choice in shared rounds (patches/0340, off by default): ``GLM53_TF_BATCH_ADAPT=1`` makes each slot of
a request with cost-derived depths pick its round's drafter (MTP, DFlash2, or none: a one-row serial round with
``GLM53_TF_BATCH_ADAPT_SERIAL=1``) by the drafts' surplus at the SHARED rounds' rate and marginal row costs
(``adapt.SlotChoice``) instead of ``decode.DrafterChoice``'s tokens per lone-model ms; ``BatchDepth`` still picks the
depth. Alone, a slot's own choice decides as before. Drafts only: replies unchanged.

Multi-slot prefill (patches/0560, ``mpf.py``, GLM53_TF_MULTI_PREFILL=1, off by default): the round's pieces of fast
lean prefills with the same knobs go through ONE forward (``mpf.run``: each member's rows contiguous, its own
sub-blocks, KDA / DSA / carries / commit / snapshots / head per member, the routed experts once over every row), up to
GLM53_TF_MULTI_PREFILL_ROWS rows; each member's piece then ends in ``_piece`` as alone (``mpf.take``: the group's first
token instead of ``decode.prefill``). Same bits a member as alone (0085's row-independent fast kernels). Rank 0 also
takes more pieces a round when the fair share allows one (``mpf.more_pieces``) and, idle, waits a few ms for more
arrivals (``mpf.coalesce``).

Kept in step with ``forward.py``: ``compute_multi``'s layer loop, ``_kda`` and ``_dsa`` mirror ``forward.compute``,
``kda_block`` and ``dsa_block`` (and ``latent``'s pieces); a change there must be made here too (the tests compare
batched rows with lone ones bit for bit).
"""

from __future__ import annotations

import collections
import copy
import hashlib
import json
import os
import queue
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Sequence

import numpy as np
import torch

from tensorfold.engine.exact_sampling import MARGIN, choose_rows

from . import adapt as adapt_mod                                            # patches/0340
from . import cpupin, decode_overlap as dover                              # patches/0370
from . import draftvocab                                                   # patches/0420
from . import batchplan, comm as comm_mod, decode_v2 as dv2, glue, kda as kda_mod, kvpool, latent, qmm, sparse
from .attention import attention, kv_write
from . import l2pf                                                           # patches/0460
from . import vision as vision_mod                                           # patches/0500
from . import mpf                                                            # patches/0560
from .forward import State, _join, check_room, chunks_for, commit, mlp_block, moe_block, out_proj

BACKLOG = 32                     # committed rows a drafter may lag behind before they are taken early


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, "") or default)


from . import gpuround, gpusample                                          # noqa: E402  (patches/0450)
from .kda import SlotTables as _SlotTables                                 # noqa: E402  (patches/0450)


# -- the batched forward ------------------------------------------------------------------------------------------
def _starts(Rs: Sequence[int]) -> list[int]:
    offs, o = [], 0
    for R in Rs:
        offs.append(o)
        o += R
    return offs


def _kda(layer, w, sts: Sequence[State], b, Rs: Sequence[int], offs: Sequence[int], T: int) -> torch.Tensor:
    """``forward.kda_block`` over T rows of several sequences: the projections once, the chain per sequence."""

    if isinstance(getattr(b, "kda_slots", None), _SlotTables):     # patches/0450 (GLM53_TF_GPU_ROUND ``kda``)
        return _kda_slots(layer, w, sts, b, T)
    c = w.cfg
    k = layer.kda
    li = sts[0].kda_index[layer.index]
    p = b.bproj[:T]
    qmm.matmul(b.normed[:T], k.proj, b.xs[:T], out=p, part=b.sk)
    fa = p[:, k.fa_off:k.fa_off + 128]
    ga = p[:, k.ga_off:k.ga_off + 128]
    if not dv2.kda_gates(p, k, b, T):                  # patches/0130, as in ``forward.kda_block``
        qmm.matmul(fa, k.fb, qmm.group_sums(fa, b.xs_fa[:T]), out=b.ka[:T], part=b.sk)
        qmm.matmul(ga, k.gb, qmm.group_sums(ga, b.xs_ga[:T]), out=b.kg[:T], part=b.sk)
    l2pf.mark(w, layer, "o", T)         # patches/0460: o_proj into L2 while the slots' chains run
    for st, R, o in zip(sts, Rs, offs):
        sp = st.proj[li, :R]                 # the slot's window rows: its chain reads them, its commit shifts them
        sp.copy_(p[o:o + R])
        cur = st.cur[li]
        y = kda_mod.chain(sp, k.b_off, b.ka[o:o + R], b.kg[o:o + R], st.conv[li], k.conv, st.rec[cur, li], k.a_log,
                          k.dt_bias, k.norm, c.eps, c.lower, R, st.scratch[li], st.rec[1 - cur, li])
        b.bkout[o:o + R].copy_(y)
    out = b.bkout[:T]
    return out_proj(w, b, out, k.o, qmm.group_sums(out, b.kxs[:T]), T)


def _kda_slots(layer, w, sts: Sequence[State], b, T: int) -> torch.Tensor:
    """patches/0450 (GLM53_TF_GPU_ROUND ``kda``, resident rounds): ``_kda`` with every slot's chain in one launch
    (``kda.chain_slots``: the slot's previous window's kept rows replayed in its prologue, its real rows only, its
    window rows' conv channels stored for the conv shift); the projections and the output as ``_kda``'s."""

    c = w.cfg
    k = layer.kda
    li = sts[0].kda_index[layer.index]
    p = b.bproj[:T]
    qmm.matmul(b.normed[:T], k.proj, b.xs[:T], out=p, part=b.sk)
    fa = p[:, k.fa_off:k.fa_off + 128]
    ga = p[:, k.ga_off:k.ga_off + 128]
    if not dv2.kda_gates(p, k, b, T):
        qmm.matmul(fa, k.fb, qmm.group_sums(fa, b.xs_fa[:T]), out=b.ka[:T], part=b.sk)
        qmm.matmul(ga, k.gb, qmm.group_sums(ga, b.xs_ga[:T]), out=b.kg[:T], part=b.sk)
    b.kda_slots.chain(li, k, k.a_log.numel(), c.eps, c.lower)
    out = b.bkout[:T]
    return out_proj(w, b, out, k.o, qmm.group_sums(out, b.kxs[:T]), T)


def _dsa(layer, w, sts: Sequence[State], b, Rs: Sequence[int], offs: Sequence[int], T: int,
         nchs: Sequence[int] | None, host_pos: Sequence[int] | None, npbs: Sequence[int] | None,
         caches: Sequence[tuple] | None = None) -> torch.Tensor:
    """``forward.dsa_block`` over T rows of several sequences: projections once, cache and attention per sequence.
    Per sequence: ``host_pos`` (eager), or ``npbs`` (capturable: 0 dense, else patches/0050's pool bucket).
    ``caches`` (patches/0200, the MTP head): per sequence (k cache, v cache, device position, indexer caches or
    None) instead of the sequence's own layer caches."""

    c = w.cfg
    indexed = sts[0].index is not None

    def cache(i: int, st: State) -> tuple:
        if caches is not None:
            return caches[i]
        di = st.dsa_index[layer.index]
        return st.kc[di], st.vc[di], st.pos_dev, st.index[di] if indexed else None

    if latent.on(w):                     # GLM53_TF_LATENT_KV=1 (patches/0060): ``latent.dsa_block_latent`` per slot
        latent._project(layer, w, b, T)
        if indexed:
            latent._index_proj(layer, w, b, T)
        l2pf.mark(w, layer, "o", T)     # patches/0460: kv_v + o_proj into L2 while the slots attend
        for i, (st, R, o) in enumerate(zip(sts, Rs, offs)):
            kc, _, pos_dev, index = cache(i, st)
            latent._attend(layer, w, kc, pos_dev, b, o, R, nchs[i] if nchs is not None else None,
                           index, host_pos[i] if host_pos is not None else None,
                           (npbs[i] or None) if npbs is not None else None)
        return latent._output(layer, w, b, T)
    a = layer.dsa
    qmm.matmul(b.normed[:T], a.proj, b.xs[:T], out=b.dp[:T], part=b.sk)
    glue.rmsnorm(b.dp[:T, :c.q_lora], a.q_norm, c.eps, b.qr[:T], b.xs_qr[:T])
    glue.rmsnorm(b.dp[:T, c.q_lora:], a.kv_norm, c.eps, b.lat[:T], b.xs_lat[:T])
    HL = a.heads
    qmm.matmul(b.qr[:T], a.q_b, b.xs_qr[:T], out=b.q[:T].view(T, HL * c.qk_dim), part=b.sk)
    kn, vn = b.kn[:T].view(T, HL * c.qk_dim), b.vn[:T].view(T, HL * c.v_dim)
    if not dv2.kv_pair(b.lat[:T], b.xs_lat[:T], a.kv_k, a.kv_v, kn, vn, b.sk):     # patches/0130, as in forward.py
        qmm.matmul(b.lat[:T], a.kv_k, b.xs_lat[:T], out=kn, part=b.sk)
        qmm.matmul(b.lat[:T], a.kv_v, b.xs_lat[:T], out=vn, part=b.sk)
    ix = a.index
    if indexed:
        qmm.matmul(b.normed[:T], ix.kw, b.xs[:T], out=b.ikr[:T], part=b.sk)
        glue.router(b.normed[:T], ix.gate, b.igr[:T])
    l2pf.mark(w, layer, "o", T)         # patches/0460
    scale = c.qk_dim ** -0.5
    for i, (st, R, o) in enumerate(zip(sts, Rs, offs)):
        kc, vc, pos_dev, index = cache(i, st)
        kv_write(b.kn[o:o + R], b.vn[o:o + R], kc, vc, pos_dev)
        hp = host_pos[i] if host_pos is not None else None
        npb = (npbs[i] or None) if npbs is not None else None
        sparse_rows = indexed and hp is not None and hp + R - 1 >= c.dense_limit
        every_sparse = npb is not None or (sparse_rows and hp >= c.dense_limit)
        sparse_rows = sparse_rows or npb is not None
        if indexed:
            ik, ig, pk = index
            sparse.index_update(b.ikr[o:o + R, :c.index_dim], b.igr[o:o + R], ix.ln_w, ix.ln_b, ix.ape, ik, ig, pk,
                                pos_dev)
        out = b.attn.out[o:o + R]
        if not every_sparse:
            attention(b.q[o:o + R], kc, vc, pos_dev, b.attn, scale=scale,
                      nch=nchs[i] if nchs is not None else None, out=out)
        if sparse_rows:
            qmm.matmul(b.qr[o:o + R], ix.qb, b.xs_qr[o:o + R], out=b.qi[o:o + R], part=b.sk)
            if npb is not None:
                tokens, counts = sparse.select_tokens_dev(b.qi[o:o + R], b.ikr[o:o + R, c.index_dim:], pk,
                                                          pos_dev, R, npb, b.lc)
            else:
                np_max = pk.shape[0] - 2
                if w.meta.get("longctx_bound"):            # patches/0050: score only the pools that exist
                    np_max = sparse.pool_bucket(hp + R, np_max)
                tokens, counts = sparse.select_tokens(b.qi[o:o + R], b.ikr[o:o + R, c.index_dim:], pk, hp, R,
                                                      np_max, pos_dev)
            lc = b.lc if npb is not None or w.meta.get("longctx_bound") else None
            sparse.sparse_attention(b.q[o:o + R], kc, vc, tokens, counts, out, scale, lc)
    o_all = b.attn.out[:T].view(T, HL * c.v_dim)
    return out_proj(w, b, o_all, a.o, qmm.group_sums(o_all, b.xs_ao[:T]), T)


def compute_multi(w, sts: Sequence[State], b, Rs: Sequence[int], *, logits: bool = True,
                  nchs: Sequence[int] | None = None, host_pos: Sequence[int] | None = None,
                  npbs: Sequence[int] | None = None):
    """``forward.compute`` for staged rows of several sequences: sequence i's R_i rows follow sequence i-1's, at
    positions sts[i].pos .. + R_i - 1. Capturable (static buffers, device positions) with ``nchs``/``host_pos``
    None; ``npbs`` (capturable): per sequence 0 (every row dense) or patches/0050's pool bucket (every row past
    2,050). -> logits [sum R, V/world] (a view of b.logits)."""

    from . import hc_cuda                # patches/0520 (GLM53_TF_HC_CUDA): the boundaries as ``forward.layer_forward``

    c = w.cfg
    offs = _starts(Rs)
    T = sum(Rs)
    glue.embed(b.ids[:T], w.embed, c.hidden, c.streams, b.x[:T])
    for layer in w.layers:
        x = b.x[:T]
        h = layer.attn_hc
        if not (hc_cuda.ON and hc_cuda.take_ready(b, layer, T)):
            glue.hc_pre(x, h.fn, h.base, h.scale, layer.in_norm, b.normed[:T], b.xs[:T], b.post[:T], b.comb[:T],
                        b.hcpart[:T], c.eps, c.hc_eps, c.hc_iters)
        b.site = (layer.index, "a")      # patches/0460: the all-gather sites, as ``forward.layer_forward`` tags them
        if layer.kind == "kda":
            g = _kda(layer, w, sts, b, Rs, offs, T)
        else:
            g = _dsa(layer, w, sts, b, Rs, offs, T, nchs, host_pos, npbs)
        h = layer.ffn_hc
        if not (hc_cuda.ON and hc_cuda.post_pre(x, g, b.post[:T], b.comb[:T], h, layer.post_norm, b.normed[:T],
                                                b.xs[:T], b.hcpart[:T], c.eps, c.hc_eps, c.hc_iters)):
            glue.hc_post(x, x, g, b.post[:T], b.comb[:T])
            glue.hc_pre(x, h.fn, h.base, h.scale, layer.post_norm, b.normed[:T], b.xs[:T], b.post[:T], b.comb[:T],
                        b.hcpart[:T], c.eps, c.hc_eps, c.hc_iters)
        b.site = (layer.index, "f")
        g = mlp_block(layer, w, b, T) if layer.mlp is not None else moe_block(layer, w, b, T)
        nxt = hc_cuda.next_layer(w, layer) if hc_cuda.ON else None
        if nxt is not None and hc_cuda.post_pre(x, g, b.post[:T], b.comb[:T], nxt.attn_hc, nxt.in_norm,
                                                b.normed[:T], b.xs[:T], b.hcpart[:T], c.eps, c.hc_eps, c.hc_iters):
            hc_cuda.mark_ready(b, nxt, T)
        else:
            glue.hc_post(x, x, g, b.post[:T], b.comb[:T])
        for slot in b.tap_at.get(layer.index, ()):
            glue.stream_mean(b.x[:T], b.taps[slot][:T])
    glue.stream_mean(b.x[:T], b.hidden[:T])
    if not logits:
        _join(w)                         # patches/0460: no open prefetch fork past a forward (graphs)
        return None
    glue.rmsnorm(b.hidden[:T], w.norm, c.eps, b.fnormed[:T], b.fxs[:T])
    out = qmm.matmul(b.fnormed[:T], w.head, b.fxs[:T], out=b.logits[:T], part=b.sk)
    _join(w)
    return out


def stage_multi(w, sts: Sequence[State], b, windows: Sequence[Sequence[int]]) -> int:
    """``forward.stage`` for several sequences' windows (their token ids back to back)."""

    tokens = [int(t) for win in windows for t in win]
    T = len(tokens)
    if T > b.rows:
        raise ValueError(f"a batched round of {T} rows, buffers hold {b.rows}")
    for st, win in zip(sts, windows):
        check_room(w, st, len(win))
    b.staged.synchronize()
    b.ids_host[:T].numpy()[:] = tokens
    b.ids[:T].copy_(b.ids_host[:T], non_blocking=True)
    b.staged.record()
    return T


def sample_multi(w, logits: torch.Tensor, specs: Sequence[tuple[int, int, Sequence[int], Any]],
                 rider: torch.Tensor | None = None):
    """``decode.sample_rows`` for several sequences' rows (specs: (row offset, rows, positions, sampling)) with ONE
    all-gather of every row's candidates: each sequence's top-k is the call ``sample_rows`` makes, and the draw is
    the same keyed rule on the same gathered candidates.

    patches/0370: ``rider`` (int32 [P] on the device, the same P on both ranks) travels at the end of the exchange;
    then -> (tokens, rank 0's P words)."""

    parts, ks = [], []
    for off, R, _, sampling in specs:
        greedy = sampling is None or sampling.temperature <= 0
        k = 1 if greedy else min(logits.shape[1], int(sampling.top_k) + MARGIN)
        vals, ids = torch.topk(logits[off:off + R].float(), k, dim=-1)
        ids = (ids + w.vocab_offset).to(torch.int32)
        parts.append(torch.cat([vals, ids.view(torch.float32)], dim=1).reshape(-1))
        ks.append(k)
    if rider is not None:                               # patches/0370: bits, moved as they are
        parts.append(rider.view(torch.float32))
    flat = torch.cat(parts).contiguous()
    world = 1 if w.comm is None else w.world
    if gpuround.sample_on(w) and all(gpusample.fits(k, world, sp[3]) for sp, k in zip(specs, ks)):
        return _sample_multi_device(w, flat, specs, ks, world, rider)       # patches/0450
    if w.comm is None:
        got = flat.view(1, -1).cpu()
    else:
        buf = torch.empty((world * flat.numel(),), dtype=torch.float32, device=logits.device)
        comm_mod.fast_gather(w.comm, flat, buf)            # patches/0230
        got = buf.view(world, -1).cpu()
        comm_mod.check(w.comm)
    out, at = [], 0
    for (off, R, positions, sampling), k in zip(specs, ks):
        seg = got[:, at:at + R * 2 * k].reshape(world, R, 2 * k)
        at += R * 2 * k
        values = torch.cat([seg[r, :, :k] for r in range(world)], dim=1).numpy().astype(np.float32)
        tokens = torch.cat([seg[r, :, k:].contiguous().view(torch.int32) for r in range(world)], dim=1).numpy()
        tokens = tokens.astype(np.int64)
        if sampling is None or sampling.temperature <= 0:
            order = np.lexsort((tokens, -values), axis=-1)
            out.append([int(tokens[i, order[i, 0]]) for i in range(R)])
        else:
            out.append(choose_rows(values, tokens, list(positions), sampling))
    if rider is not None:                               # patches/0370: rank 0's words
        return out, got[0, at:at + rider.numel()].contiguous().view(torch.int32).tolist()
    return out


def _sample_multi_device(w, flat: torch.Tensor, specs, ks: Sequence[int], world: int, rider: torch.Tensor | None):
    """patches/0450 (GLM53_TF_GPU_ROUND ``sample``): ``sample_multi``'s draws on the GPU with the host's bits
    (``gpusample``); one readback of the tokens (and rank 0's rider words)."""

    if w.comm is None:
        buf = flat
    else:
        buf = torch.empty((world * flat.numel(),), dtype=torch.float32, device=flat.device)
        comm_mod.fast_gather(w.comm, flat, buf)            # patches/0230
    rows, at = [], 0
    for (off, R, positions, sampling), k in zip(specs, ks):
        rows += [(at + r * 2 * k, k, int(positions[r]), sampling, False) for r in range(R)]
        at += R * 2 * k
    tok, _ = gpusample.choose(buf, flat.numel(), world, rows)
    parts = [tok]
    if rider is not None:                               # rank 0's words: right after the candidates in its part
        parts.append(buf[at:at + rider.numel()].view(torch.int32))
    host = torch.cat(parts).cpu().tolist()
    comm_mod.check(w.comm)
    out, i = [], 0
    for _, R, _, _ in specs:
        out.append([int(t) for t in host[i:i + R]])
        i += R
    if rider is not None:
        return out, [int(v) for v in host[i:i + rider.numel()]]
    return out


def _sample_drafts_device(w, parts: list, specs, ks: Sequence[int], world: int) -> list[tuple[int, float | None]]:
    """patches/0450: ``sample_drafts``' draws and probabilities on the GPU."""

    flat = torch.cat(parts).contiguous()
    if w.comm is None:
        buf = flat
    else:
        buf = torch.empty((world * flat.numel(),), dtype=torch.float32, device=flat.device)
        comm_mod.fast_gather(w.comm, flat, buf)            # patches/0230
    rows, at = [], 0
    for (_, pos, sampling, want), k in zip(specs, ks):
        rows.append((at, k, int(pos), sampling, bool(want)))
        at += 2 * k
    want_any = any(r[4] for r in rows)
    tok, prob = gpusample.choose(buf, flat.numel(), world, rows, want=want_any)
    n = len(rows)
    host = (torch.cat([tok.to(torch.float64), prob]) if want_any else tok).cpu().tolist()
    comm_mod.check(w.comm)
    return [(int(host[i]), float(host[n + i]) if rows[i][4] else None) for i in range(n)]


# -- patches/0200: MTP drafts of several slots in one pass --------------------------------------------------------------
def sample_drafts(w, logits: torch.Tensor, specs: Sequence[tuple[int, int, Any, bool]]) -> list[tuple[int, float | None]]:
    """``decode.sample_rows`` for one row each of several sequences' draft logits (specs: (row, position, sampling,
    with probability)) with ONE all-gather: per row the same candidates (top-k, or 20 + MARGIN for a greedy draft
    whose probability is wanted), the same draw and ``_probability``. -> [(token, probability or None)]."""

    from .decode import _probability

    parts, ks = [], []
    V = logits.shape[1]
    for row, _, sampling, want in specs:
        greedy = sampling is None or sampling.temperature <= 0
        k = 1 if greedy else min(V, int(sampling.top_k) + MARGIN)
        if want and greedy:
            k = min(V, 20 + MARGIN)
        vals, ids = torch.topk(logits[row:row + 1].float(), k, dim=-1)
        ids = draftvocab.ids_of(w, ids, V)             # patches/0420: listed rows' columns -> token ids
        parts.append(torch.cat([vals, ids.view(torch.float32)], dim=1).reshape(-1))
        ks.append(k)
    world = 1 if w.comm is None else w.world
    if gpuround.sample_on(w) and all(gpusample.fits(k, world, sp[2], sp[3]) for sp, k in zip(specs, ks)):
        return _sample_drafts_device(w, parts, specs, ks, world)            # patches/0450
    flat = torch.cat(parts).contiguous()
    if w.comm is None:
        world = 1
        got = flat.view(1, -1).cpu()
    else:
        world = w.world
        buf = torch.empty((world * flat.numel(),), dtype=torch.float32, device=logits.device)
        comm_mod.fast_gather(w.comm, flat, buf)            # patches/0230
        got = buf.view(world, -1).cpu()
        comm_mod.check(w.comm)
    out, at = [], 0
    for (row, pos, sampling, want), k in zip(specs, ks):
        seg = got[:, at:at + 2 * k].reshape(world, 1, 2 * k)
        at += 2 * k
        values = torch.cat([seg[r, :, :k] for r in range(world)], dim=1).numpy().astype(np.float32)
        tokens = torch.cat([seg[r, :, k:].contiguous().view(torch.int32) for r in range(world)], dim=1).numpy()
        tokens = tokens.astype(np.int64)
        if sampling is None or sampling.temperature <= 0:
            order = np.lexsort((tokens, -values), axis=-1)
            chosen = [int(tokens[0, order[0, 0]])]
        else:
            chosen = choose_rows(values, tokens, [pos], sampling)
        out.append((int(chosen[0]), _probability(values, tokens, chosen, sampling)[0] if want else None))
    return out


def mtp_multi(w, sts: Sequence[State], b, tokens: Sequence[Sequence[int]], hidden: Sequence[torch.Tensor],
              full: bool = False) -> torch.Tensor:
    """``mtp.mtp_stage`` + ``mtp_compute`` (eager, last row of each sequence) for several sequences' MTP rows in ONE
    head pass: sequence i's rows (``hidden[i]`` [n_i, D], the tokens after them) at its head-cache slots mtp_len..;
    the head's weights (its half of the vocabulary head, the MoE layer) are read once for all of them. Row-local
    kernels over all rows; the DSA cache write and attention per sequence on its own head cache (``_dsa`` with
    ``caches``). -> logits [N, draft_n] (row i: sequence i's last row), the head's output rows in b.fnormed[:N] (the
    chained drafts' hidden). The caller advances each ``mtp_len``."""

    from .attention import CHUNK
    from .forward import _join

    c = w.cfg
    m = w.mtp
    D = c.hidden
    ns = [len(t) for t in tokens]
    offs = _starts(ns)
    T = sum(ns)
    N = len(sts)
    if T > b.rows:
        raise ValueError(f"MTP rows of {N} sequences: {T}, the head's buffers hold {b.rows}")
    for st, n in zip(sts, ns):
        check_room(w, st, n, pos=st.mtp_len)
    zero = [o for st, o in zip(sts, offs) if st.mtp_len == 0]      # ``mtp_stage``'s zero_first, per sequence
    b.staged.synchronize()
    b.ids_host[:T].numpy()[:] = [int(t) for tt in tokens for t in tt]
    b.ids[:T].copy_(b.ids_host[:T], non_blocking=True)
    for h, o, n in zip(hidden, offs, ns):
        if h.data_ptr() != b.hin[o:o + n].data_ptr():
            b.hin[o:o + n].copy_(h)
    b.staged.record()
    glue.embed(b.ids[:T], w.embed, D, 1, b.me[:T])
    for o in zero:
        b.me[o].zero_()
    glue.rmsnorm(b.me[:T], m.enorm, c.eps, b.mcat[:T, :D])
    glue.rmsnorm(b.hin[:T], m.hnorm, c.eps, b.mcat[:T, D:])
    qmm.matmul(b.mcat[:T], m.eh, qmm.group_sums(b.mcat[:T], b.mxs[:T]), out=b.mx[:T], part=b.sk)
    layer = m.layer
    glue.rmsnorm(b.mx[:T], layer.in_norm, c.eps, b.normed[:T], b.xs[:T])
    b.site = ("mtp", "a")
    caches = [(st.mtp_kc, st.mtp_vc, st.mtp_pos_dev, st.index[-1] if st.index is not None else None) for st in sts]
    g = _dsa(layer, w, sts, b, ns, offs, T, [-(-(st.mtp_len + n) // CHUNK) for st, n in zip(sts, ns)],
             [st.mtp_len for st in sts], None, caches=caches)
    glue.residual_add(b.mx[:T], b.mx[:T], g)
    glue.rmsnorm(b.mx[:T], layer.post_norm, c.eps, b.normed[:T], b.xs[:T])
    b.site = ("mtp", "f")
    g = moe_block(layer, w, b, T)
    glue.residual_add(b.mx[:T], b.mx[:T], g)
    for i, (o, n) in enumerate(zip(offs, ns)):                     # each sequence's last row, as ``last_only``
        glue.rmsnorm(b.mx[o + n - 1:o + n], m.norm, c.eps, b.fnormed[i:i + 1], b.fxs[i:i + 1])
    head = draftvocab.head_for(w, "mtp", full)       # patches/0420: the listed rows unless ``full``
    out = qmm.matmul(b.fnormed[:N], head, b.fxs[:N], out=b.logits[:N, :head.n], part=b.sk)
    _join(w)
    return out


class MtpChains:
    """patches/0200 (GLM53_TF_BATCH_MTP=1): the MTP drafts of every slot whose next window is an MTP one, chained
    together: one head pass absorbs every slot's backlog, then one head pass per chained step over the slots still
    drafting. Each slot's chain follows ``decode.draft`` step for step (its own sampling, confidence or cost-depth
    stop, positions, head cache), and each row is the bits of the slot's own head pass (row-local kernels, attention
    per slot), so the drafts are those the slot would draft alone. Drafts only propose: replies never depend on them."""

    def __init__(self) -> None:
        self.items: list[SimpleNamespace] = []

    def add(self, stepper: "Stepper", st: State, backlog: int, depth: int) -> None:
        self.items.append(SimpleNamespace(stepper=stepper, st=st, backlog=backlog, depth=depth))

    def _alone(self, bat: "Batcher", it) -> None:
        from .decode import draft

        q = it.stepper
        with bat._on(q.slot) as e:
            drafts = draft(e, q.m_rows[:it.backlog], q.m_next, e.st.pos + 1, it.depth, q.job.sampling,
                           q.m_policy.confidence, opt=q.opt)
        self._done(it, drafts)

    @staticmethod
    def _done(it, drafts: list[int]) -> None:
        q = it.stepper
        q.arm, q.drafts, q.steps, q.backlog = "m", list(drafts), 1 + it.st.mtp_drafted, it.backlog
        q.m_next = []

    def run(self, bat: "Batcher") -> bool:
        """Draft every added chain. -> whether they ran together (else each alone)."""

        items = self.items
        self.items = []
        if not items:
            return False
        e = bat.g.e
        w, b = e.w, e.mbuf
        if len(items) < 2 or sum(it.backlog for it in items) > b.rows:
            for it in items:                        # one slot (its own head graphs), or more rows than the buffers
                self._alone(bat, it)
            return False
        for it in items:                            # ``decode.absorb``: the chained entries of the last round go
            st = it.st
            if st.mtp_drafted:
                st.set_mtp_len(st.mtp_len - st.mtp_drafted)
                st.mtp_drafted = 0
            it.n, it.j, it.chain, it.drafts, it.pos = len(it.stepper.m_next), 0, 1.0, [], st.pos + 1
            it.opt = it.stepper.opt
            it.conf = it.stepper.m_policy.confidence
            if it.opt is not None:
                it.opt.mtp_begin()
        sts = [it.st for it in items]
        for it in items:                            # patches/0420: the fallback rule's window, as ``decode.absorb``
            draftvocab.track(w, it.st, it.stepper.m_next)
        # patches/0420: one head for the pass: the whole vocabulary when any slot's rule asks for it
        full = {"full": True} if any(draftvocab.full(w, st, "mtp") for st in sts) else {}
        logits = mtp_multi(w, sts, b, [it.stepper.m_next for it in items],
                           [it.stepper.m_rows[:it.backlog] for it in items], **full)[:, :e.draft_n]
        for it in items:
            it.st.set_mtp_len(it.st.mtp_len + it.n)
        live = list(range(len(items)))              # items still drafting, in the order of the logits rows
        from .decode import Engine

        # the engine samples with ``sample_rows`` (not a stand-in): every live row's draft in one all-gather
        one_gather = type(e).sample is Engine.sample and "sample" not in vars(e)
        while live:
            step = []                               # (item, draft) that take one more head step
            if one_gather:
                drawn = sample_drafts(w, logits, [(r, items[i].pos + items[i].j, items[i].stepper.job.sampling,
                                                   items[i].opt is not None or items[i].conf > 0)
                                                  for r, i in enumerate(live)])
            for r, i in enumerate(live):
                it = items[i]
                q = it.stepper
                want = it.opt is not None or it.conf > 0
                probs: list[float] = []
                if one_gather:
                    d, p = drawn[r]
                    if want:
                        probs.append(p)
                else:
                    d = e.sample(logits[r:r + 1], [it.pos + it.j], q.job.sampling, draft=True,
                                 probs=probs if want else None)[0]
                if it.opt is not None:              # ``decode.draft`` with ``opt`` (patches/0071)
                    take, more = it.opt.mtp_next(it.j, probs[0], it.depth)
                    if not take:
                        continue
                    it.drafts.append(d)
                    if more:
                        step.append((i, d, r))
                    continue
                if it.conf > 0 and it.j > 0 and it.chain * probs[0] < it.conf:
                    continue
                it.drafts.append(d)
                if it.conf > 0:
                    it.chain *= probs[0]
                    if it.chain < it.conf:
                        continue
                if it.j + 1 < it.depth:
                    step.append((i, d, r))
            if not step:
                break
            # the head's output rows of this pass feed the next (``Engine.draft_hidden``); copied out first, the
            # pass overwrites b.fnormed
            prev = [b.fnormed[r:r + 1].clone() for _, _, r in step]
            logits = mtp_multi(w, [items[i].st for i, _, _ in step], b, [[d] for _, d, _ in step], prev,
                               **full)[:, :e.draft_n]
            live = []
            for i, _, _ in step:
                it = items[i]
                it.st.set_mtp_len(it.st.mtp_len + 1)
                it.st.mtp_drafted += 1
                it.j += 1
                if it.j < it.depth:                 # ``for j in range(count)``: the chain ends at count
                    live.append(i)
        for it in items:
            self._done(it, it.drafts)
        return True


# -- per-slot DFlash2 contexts --------------------------------------------------------------------------------------
def _drafter_view(d):
    """The DFlash2 drafter with its own context: the weights and per-pass buffers shared with ``d`` (one pass runs at
    a time), the context caches, positions and CUDA graphs its own (``Drafter.capture``: both ranks, at load)."""

    v = copy.copy(d)
    v.kc = [torch.zeros_like(t) for t in d.kc]
    v.vc = [torch.zeros_like(t) for t in d.vc]
    v.pos_dev = torch.zeros_like(d.pos_dev)
    v.lo_dev = torch.zeros_like(d.lo_dev)
    v._end = v._hi = v._lo = 0
    v.packed = v.proj = None
    v.pool = None
    v.block_graph = None
    v.tap_graphs = {}
    if d.block_graph is not None:
        v.capture()
    return v


# -- requests and running sequences -------------------------------------------------------------------------------
@dataclass
class Job:
    prompt: list[int]
    max_tokens: int
    sampling: Any
    stop_eos: bool
    draft: bool
    code: list[int]
    spec: str = ""
    cost: int = 0                                 # patches/0071: cost-derived depths for auto
    values: dict = field(default_factory=dict)    # patches/0090: the request's knobs (every ``knobs.HEADER`` key)
    background: bool = False                      # steps aside for a waiting request, runs again later
    out: "queue.SimpleQueue | None" = None        # rank 0: token lists, then None (or an exception)
    cancel: bool = False
    sent: int = 0                                 # tokens delivered to the caller (a rerun skips them)
    submitted: float = 0.0
    echo: dict = field(default_factory=dict)      # the response's ``tf_knobs``
    stats: dict = field(default_factory=dict)
    vision: Any = None                            # patches/0500: the prompt's image rows (``vision.Table``) or None
    sess: Any = None                              # patches/0180: the admission's ``sessions.Plan`` (store on)
    lp_sink: Any = None                           # patches/9001: rank 0: a list the delivered tokens' logprobs extend


class Always:
    """``depth.Always``: a drafter choice that always picks one arm (om / of / fN / N policies)."""

    def __init__(self, arm: str) -> None:
        self.arm = arm

    def pick(self) -> str:
        return self.arm

    def record(self, *args) -> None:
        pass


class Stepper:
    """One request's decode loop (``decode.auto_decode``; ``mtp_decode``, ``dflash_decode`` and ``serial_decode`` are
    its special cases) cut at the forward: ``propose`` the next window's drafts, then ``accept`` the round's sampled
    rows. Same drafters, policies, backlogs and records as alone, so the same drafts."""

    def __init__(self, bat: "Batcher", slot: int, job: Job, first: int, last_hidden: torch.Tensor) -> None:
        from . import depth as depth_mod
        from .decode import DepthPolicy, DrafterChoice
        from .engine import MAX_ROWS, decode_policy
        from .lookup import LOOKUP_KIND, lookup_for

        g = bat.g
        code = job.code
        self.bat, self.slot, self.job = bat, slot, job
        auto, use_mtp, use_dflash = g._drafters(code)
        self.use_mtp = use_mtp
        self.drafter = bat.drafters[slot] if use_dflash else None
        if self.drafter is not None:            # patches/0420: the fallback rule reads the slot's MTP window
            self.drafter.dv_st = bat.states[slot]
        greedy = job.sampling is None or job.sampling.temperature <= 0
        self.policy = policy = decode_policy(code)
        costs = bat.costs
        self.lookup = lookup_for(code, job.prompt, costs, eos=bat.eos, stop_eos=job.stop_eos)
        self.opt = None
        f_most = int(job.values.get("auto_fdrafts", g.f_most))
        if policy is not None and (code[0] == depth_mod.OPT_KIND or (auto and job.cost)):
            most_m = code[1] if code[0] == depth_mod.OPT_KIND else MAX_ROWS - 1
            most_f = code[1] if code[0] == depth_mod.OPT_KIND else f_most
            self.opt = BatchDepth(bat.round_costs, costs, most_m=most_m, most_f=most_f)
            if self.lookup is not None:
                self.lookup.opt = self.opt
        m_std = DepthPolicy(3, fixed=True, confidence=0.35) if greedy else DepthPolicy(3, low=0.6, high=0.85)
        self.choice = None
        self.m_policy = self.f_policy = m_std
        if policy is None or code[0] == LOOKUP_KIND:
            pass                                  # serial; lN: lookup rounds, MTP rounds without a match
        elif auto:
            _, explore, every, margin, sampled_too = policy
            if self.drafter is not None and (greedy or sampled_too):
                self.choice = DrafterChoice(costs, first="f" if greedy else "m", explore=explore, every=every,
                                            margin=margin)
            self.f_policy = DepthPolicy(f_most, fixed=True, confidence=0.3)
        elif use_dflash:
            self.choice = Always("f")
            self.f_policy = policy
        else:
            self.m_policy = policy
        # patches/0340 (GLM53_TF_BATCH_ADAPT): shared rounds' drafter chosen at the round's margin; alone: ``choice``
        self.adapt = None
        if getattr(bat, "adapt", False) and self.opt is not None and policy is not None and code[0] != LOOKUP_KIND:
            arms = "mf" if isinstance(self.choice, DrafterChoice) else getattr(self.choice, "arm", "m")
            self.adapt = adapt_mod.SlotChoice(costs, arms, self.choice, every=bat.adapt_every,
                                              window=bat.adapt_window, serial=bat.adapt_serial)
        self.out = [first]
        self.m_rows = bat.m_rows[slot]
        self.m_next: list[int] = []
        if self.use_mtp:
            self.m_rows[:1].copy_(last_hidden)
            self.m_next = [first]
        self.f_taps = bat.f_taps[slot] if self.drafter is not None else None
        self.n_f = 0
        self.last = {"m": (0, 0), "f": (0, 0)}
        self.arm, self.drafts, self.steps, self.backlog = "s", [], 0, 0
        self.rounds = self.shared = self.drafted = self.accepted = 0
        self.depths: list[int] = []
        self.keeps: list[int] = []
        self.arms: list[str] = []
        self.start = time.perf_counter()

    def done(self, eos) -> bool:
        return len(self.out) >= self.job.max_tokens or (self.job.stop_eos and self.out[-1] in eos)

    def propose(self, e, chains: "MtpChains | None" = None) -> None:
        """The next window's drafts (``e``: the engine on this slot). ``chains`` (patches/0200): an MTP window's
        chain is left to ``chains``, which drafts every slot's MTP chain together (after this call returns)."""

        from .decode import draft

        job, st = self.job, e.st
        out = self.out
        room = job.max_tokens - len(out)
        if self.policy is None or room <= 0:
            self.arm, self.drafts, self.steps, self.backlog = "s", [], 0, 0
            return
        look = self.lookup.plan(out, room) if self.lookup is not None else None
        if look:
            arm = "l"
        elif self.adapt is not None:            # patches/0340: the shared rounds' rate, None alone
            arm = self.adapt.pick(self.bat.round_costs.rate())
        else:
            arm = self.choice.pick() if self.choice is not None else "m"
        if arm == adapt_mod.SERIAL:             # patches/0340: a serial round (the pending token alone)
            self.arm, self.drafts, self.steps, self.backlog = arm, [], 0, 0
            return
        if arm == "l":
            backlog, drafts, steps = 0, look, 0
        elif arm == "m":
            backlog = len(self.m_next)
            depth = max(1, min(self.m_policy.next(*self.last["m"]) if self.opt is None else self.opt.most_m, room))
            if chains is not None:
                self.arm, self.drafts, self.steps, self.backlog = "m", [], 0, backlog
                chains.add(self, st, backlog, depth)
                return
            drafts = draft(e, self.m_rows[:backlog], self.m_next, st.pos + 1, depth, job.sampling,
                           self.m_policy.confidence, opt=self.opt)
            steps = 1 + st.mtp_drafted
            self.m_next = []
        else:
            d = self.drafter
            backlog = self.n_f
            if self.n_f:
                d.add_taps(self.f_taps[:self.n_f])
                self.n_f = 0
            if self.opt is None:
                depth = max(1, min(self.f_policy.next(*self.last["f"]), room))
                drafts = d.propose(out[-1], depth, job.sampling, self.f_policy.confidence)
            else:
                f_probs: list[float] = []
                drafts = d.propose(out[-1], max(1, min(self.opt.most_f, room)), job.sampling, 0.0, probs=f_probs)
                if drafts:
                    drafts = drafts[:self.opt.f_depth(f_probs[:len(drafts)])]
            steps = 0
        self.arm, self.drafts, self.steps, self.backlog = arm, list(drafts), steps, backlog

    def window(self) -> list[int]:
        return [self.out[-1]] + self.drafts

    def accept(self, e, sampled: list[int], off: int, shared: bool, rows: int | None = None) -> int:
        """The round's sampled rows (this request's, at ``off`` of the forward): keep up to the first mismatch,
        commit, and hand the kept rows to the drafters' backlogs. ``rows`` (patches/0200): the rows the forward ran
        for this request when its window was padded (the padded rows are never kept). -> rows kept."""

        from .decode import absorb

        job, bat = self.job, self.bat
        w, st, b = e.w, e.st, e.buf
        drafts = self.drafts
        R = 1 + len(drafts)
        keep = 1
        for i, d in enumerate(drafts):
            if sampled[i] != d or (job.stop_eos and sampled[i] in bat.eos):
                break
            keep += 1
        commit(w, st, b, max(R, int(rows or R)), keep)
        if self.policy is not None:
            if self.use_mtp:
                if len(self.m_next) + keep > BACKLOG:
                    absorb(e, self.m_rows[:len(self.m_next)], self.m_next)
                    self.m_next = []
                n = len(self.m_next)
                self.m_rows[n:n + keep].copy_(e.main_hidden(slice(off, off + keep)))
                self.m_next.extend(sampled[:keep])
            if self.drafter is not None:
                if self.n_f + keep > BACKLOG:
                    self.drafter.add_taps(self.f_taps[:self.n_f])
                    self.n_f = 0
                self.f_taps[self.n_f:self.n_f + keep].copy_(torch.cat([t[off:off + keep] for t in b.taps], dim=1))
                self.n_f += keep
            arm = self.arm
            if self.adapt is not None:          # patches/0340 (records the base choice's drafted rounds too)
                if arm != "l":
                    self.adapt.record(arm, R, self.steps, self.backlog, keep, self.opt.verify, shared)
            elif self.choice is not None and arm != "l":
                self.choice.record(arm, R, self.steps, self.backlog, keep)
            if self.lookup is not None and arm != adapt_mod.SERIAL:
                self.lookup.record(arm, R, self.steps, self.backlog, keep)
            if self.opt is not None:
                self.opt.record(arm, R, self.steps, self.backlog, keep)
            if arm in self.last:
                self.last[arm] = (len(drafts), keep - 1)
        self.rounds += 1
        self.shared += int(shared)
        self.drafted += len(drafts)
        self.accepted += keep - 1
        self.depths.append(len(drafts))
        self.keeps.append(keep)
        self.arms.append(self.arm)
        self.out.extend(sampled[:keep])
        return keep

    def part(self) -> tuple[str, int, int, int]:
        """(arm, rows, MTP steps, backlog) of the window just proposed (``batchplan.RoundCosts``)."""

        return self.arm, 1 + len(self.drafts), self.steps, self.backlog

    def close(self) -> torch.Tensor | None:
        """At the end: DFlash2's context takes the rows it has not (it ends where the committed rows end); -> the MTP
        input rows of committed positions the head has not absorbed (a reply snapshot's ``pending``)."""

        if self.drafter is not None and self.n_f:
            self.drafter.add_taps(self.f_taps[:self.n_f])
            self.n_f = 0
        return self.m_rows[:len(self.m_next)] if self.use_mtp else None


def _depth_optimizer():
    from .depth import DepthOptimizer

    return DepthOptimizer


class BatchDepth(_depth_optimizer()):
    """``depth.DepthOptimizer`` in shared rounds: the running rate is the rounds' aggregate one while other requests
    share them (``RoundCosts.rate``), and ``verify`` is the window-cost table past the other requests' rows (set
    before each proposal). Alone it is the lone optimizer."""

    def __init__(self, round_costs: batchplan.RoundCosts, costs: dict, **kw) -> None:
        self.round_costs = round_costs
        super().__init__(costs, **kw)

    def rate(self) -> float:
        shared = self.round_costs.rate() if self.round_costs is not None else None
        return shared if shared is not None else super().rate()

    def record(self, arm: str, rows: int, steps: int, backlog: int, keep: int) -> None:
        if arm == adapt_mod.SERIAL:             # patches/0340: a serial round, priced as a one-row window
            from .depth import RATE_WINDOW

            self.used = []
            self.rounds.append((keep, float(self.costs["verify"][0])))
            del self.rounds[:-RATE_WINDOW]
            return
        super().record(arm, rows, steps, backlog, keep)


@dataclass
class Seq:
    job: Job
    slot: int
    admitted: int                                 # the round it was admitted in
    resume: Any = None                            # decode.Snapshot the next prefill piece resumes from
    done: int = 0                                 # prompt tokens prefilled
    grid: int = 0                                 # patches/0080: the request's snapshot grid (0: exact)
    stepper: Stepper | None = None                # None while prefilling
    t0: float = 0.0
    pieces: int = 0
    solo: int = 0                                 # patches/0335: pieces cut at GLM53_TF_SOLO_PIECE (alone in the batch)
    emitted: int = 0                              # tokens produced by this run (a rerun re-produces the sent ones)
    marks: tuple = ()                             # patches/0180: the session store's marks for this prompt
    kinds: collections.Counter = field(default_factory=collections.Counter)   # patches/0200: its rounds by kind


class Batcher:
    """Up to ``n`` concurrent requests on a ``GlmEngine`` (both ranks)."""

    short = 0                                     # patches/0200 defaults (a batcher built without ``__init__``)
    solo = 0                                      # patches/0335: GLM53_TF_SOLO_PIECE (off)
    parity_key = False
    batch_mtp = False
    last_kind = "alone"
    buckets = 0                                   # patches/0280 defaults
    tie = False
    adapt = False                                 # patches/0340 defaults
    adapt_every, adapt_window, adapt_serial = adapt_mod.EVERY, adapt_mod.WINDOW, False
    src_host = src_dev = None
    kvp = None                                    # patches/0290: the KV page pool (None: every slot owns its caches)
    pool_check = False
    mpf_rows = 0                                  # patches/0560: GLM53_TF_MULTI_PREFILL's group rows (0: off)
    mpf_wait = 0.0                                # patches/0560: rank 0's idle wait for more arrivals (seconds)
    _spills: Sequence[int] = ()                   # patches/0290: this round's spilled slots (rank 0's plan)
    overlap = dover.Overlap()                     # patches/0370 defaults (off)
    _ahead: list[int] | None = None               # patches/0370: the next round's plan, received in this round
    _held: list | None = None                     # patches/0370: rank 0's tokens not handed to the HTTP threads yet
    rider_n = 0
    rider_host = rider_dev = None
    resident = None                               # patches/0450 (GLM53_TF_GPU_ROUND ``resident``): a Resident
    m_all = f_all = None                          # patches/0450: every slot's drafter backlog in one tensor each
    _resident_go: list[int] | None = None         # patches/0450: the slots to run resident after this round
    emit_first = True                             # patches/0540: GLM53_TF_EMIT_FIRST (a piece's first token at once)
    admit_policy = "free"                         # patches/0550 (``__init__``: GLM53_TF_ADMIT_MEM)
    admit_log = None                              # patches/0550: memsafe.AdmitLog (rank 0)
    trimmer = None                                # patches/0550: memsafe.Trimmer (GLM53_TF_ALLOC_TRIM_GB)

    def __init__(self, g, n: int) -> None:
        from .engine import GRAPH_ROWS, MAX_ROWS

        self.g = g
        e, w = g.e, g.w
        self.max_rows = MAX_ROWS
        if n * MAX_ROWS > e.rows:
            raise ValueError(f"GLM53_TF_BATCH={n}: {n} windows of up to {MAX_ROWS} rows need {n * MAX_ROWS} buffer "
                             f"rows, the engine has {e.rows} (raise GLM53_TF_PREFILL_ROWS)")
        self.graph_rows = _env_int("GLM53_TF_BATCH_GRAPH_ROWS", MAX_ROWS)
        self.max_graphs = _env_int("GLM53_TF_BATCH_MAX_GRAPHS", 256)
        self.piece = _env_int("GLM53_TF_BATCH_PIECE", 2048)
        # patches/0335 (GLM53_TF_SOLO_PIECE=N, 0 = off): a fast prefill piece of a request that is ALONE in the batch
        # (no other slot holds a request when its piece is cut) is N tokens instead of GLM53_TF_BATCH_PIECE; the next
        # piece is cut at the batch piece again as soon as another request is admitted (a piece boundary). Bigger
        # pieces run as bigger fast chunks when the lean set holds them (GLM53_TF_PREFILL_ROWS_MAX >= N), so the
        # routed experts read their weights once per N rows instead of once per piece. Same bits: pieces end on the
        # 64-token snapshot grid and resumed == fresh for any cut (patches/0085).
        self.solo = _env_int("GLM53_TF_SOLO_PIECE", 0)
        if self.solo < 0 or (self.solo and self.solo < 64):
            raise ValueError(f"GLM53_TF_SOLO_PIECE={self.solo}: expected 0 (off) or at least 64 tokens")
        both = g._gather_ints([self.solo])              # both ranks cut the same pieces
        if both[0] != both[1]:
            raise RuntimeError(f"the two ranks were started with different GLM53_TF_SOLO_PIECE: rank 0 {both[0]}, "
                               f"rank 1 {both[1]}")
        # patches/0510: GLM53_TF_BATCH_GRAPHS=1 (every round) | 0 (none) | lone (rounds with one active slot)
        self.graph_policy = batchplan.graph_policy(os.environ.get("GLM53_TF_BATCH_GRAPHS"))
        self.use_graphs = e.graphs is not None and self.graph_policy != batchplan.GRAPHS_OFF
        # patches/0200 (each off by default): graph keys captured on their N-th sighting, windows padded to fewer
        # graph keys, short prompts prefilled without waiting for the fair share
        self.sightings = batchplan.Sightings(_env_int("GLM53_TF_BATCH_CAPTURE_AFTER", 1))
        self.pad = batchplan.pad_mask(os.environ.get("GLM53_TF_BATCH_PAD", ""), MAX_ROWS)
        self.short = max(0, _env_int("GLM53_TF_BATCH_SHORT", 0))
        # patches/0540: a prompt's first token goes to its caller when its last piece ends, not after the round's
        # other pieces and verify forward (rank 0 only; scheduling only)
        self.emit_first = _env_int("GLM53_TF_EMIT_FIRST", 1) != 0
        self.parity_key = _env_int("GLM53_TF_BATCH_PARITY_KEY", 0) == 1
        # patches/0280: every window of a multi-slot round padded to one bucket; padded rows routed as real ones
        self.buckets = batchplan.bucket_mask(os.environ.get("GLM53_TF_BATCH_BUCKETS", ""), MAX_ROWS)
        tie = os.environ.get("GLM53_TF_BATCH_PAD_TIE", "auto").strip().lower()
        self.tie = bool(self.buckets) if tie in ("", "auto") else tie not in ("0", "false", "no", "off")
        self.batch_mtp = _env_int("GLM53_TF_BATCH_MTP", 0) == 1 and g.w.mtp is not None
        self.row_ms = float(os.environ.get("GLM53_TF_BATCH_ROW_MS", "") or 0.0)
        # patches/0340: shared rounds' drafter per slot at the round's margin (adapt.py), off by default
        self.adapt = adapt_mod.env_on()
        self.adapt_every, self.adapt_window = adapt_mod.env_every(), adapt_mod.env_window()
        self.adapt_serial = adapt_mod.env_serial()
        self.last_rows: list[int] = []                 # the rows each slot had in the last ``_forward`` (padded)
        both = g._gather_ints([n, int(self.use_graphs) * self.graph_policy,     # patches/0510: the policy
                               self.graph_rows, self.max_graphs, self.piece,
                               self.sightings.after, self.pad, self.short, int(self.parity_key),
                               int(self.batch_mtp), int(round(self.row_ms * 1000)), self.buckets, int(self.tie),
                               int(self.adapt), self.adapt_every, self.adapt_window, int(self.adapt_serial)])
        if both[0] != both[1]:
            raise RuntimeError(f"the two ranks were started with different batch settings: rank 0 {both[0]}, "
                               f"rank 1 {both[1]} (GLM53_TF_BATCH, GLM53_TF_BATCH_GRAPHS, GLM53_TF_BATCH_GRAPH_ROWS, "
                               "GLM53_TF_BATCH_MAX_GRAPHS, GLM53_TF_BATCH_PIECE, GLM53_TF_BATCH_CAPTURE_AFTER, "
                               "GLM53_TF_BATCH_PAD, GLM53_TF_BATCH_SHORT, GLM53_TF_BATCH_PARITY_KEY, "
                               "GLM53_TF_BATCH_MTP, GLM53_TF_BATCH_ROW_MS, GLM53_TF_BATCH_BUCKETS, "
                               "GLM53_TF_BATCH_PAD_TIE, GLM53_TF_BATCH_ADAPT, GLM53_TF_BATCH_ADAPT_EVERY, "
                               "GLM53_TF_BATCH_ADAPT_WINDOW, GLM53_TF_BATCH_ADAPT_SERIAL)")
        # patches/0370 (GLM53_TF_DECODE_OVERLAP): host work off the round's critical path; ``plan`` changes what
        # the sampler's all-gather carries, so both ranks must agree on it
        self.overlap = dover.env()
        both = g._gather_ints([int(self.overlap.plan)])
        if both[0] != both[1]:
            raise RuntimeError(f"the two ranks were started with different GLM53_TF_DECODE_OVERLAP 'plan' parts: rank 0 "
                               f"{both[0]}, rank 1 {both[1]}")
        dover.apply_gil(self.overlap)
        # patches/0560 (GLM53_TF_MULTI_PREFILL): a round's pieces in one forward; both ranks form the same groups
        lean_rows = int(getattr(getattr(e, "lean", None), "rows", 0) or 0)
        mine = mpf.settings(lean_rows)
        both = g._gather_ints(mine)
        if both[0] != both[1]:
            raise RuntimeError(f"the two ranks were started with different GLM53_TF_MULTI_PREFILL / "
                               f"GLM53_TF_MULTI_PREFILL_ROWS: rank 0 {both[0]}, rank 1 {both[1]}")
        self.mpf_rows = mine[1] if mine[0] and lean_rows else 0
        self.mpf_wait = mpf.wait_ms() / 1e3 if self.mpf_rows else 0.0
        if self.mpf_rows and g.rank == 0:
            print(f"[tensorfold] multi prefill (patches/0560): a round's pieces in one forward, up to {self.mpf_rows} "
                  f"rows ({int(self.mpf_wait * 1e3)} ms wait for more arrivals when idle)", flush=True)
        # rank 0 only: scheduling (its decisions travel in the round plans)
        self.fair = batchplan.Fairness(float(os.environ.get("GLM53_TF_BATCH_PREFILL_SHARE", "") or 0.5))
        # memory kept free: at load, for the slots (prefill transients, lazily captured graphs); at admission, while
        # other requests run (a request waits for one to end when less is free)
        self.reserve = float(os.environ.get("GLM53_TF_BATCH_RESERVE_GB", "") or 4.0) * 2**30
        self.admit_min = float(os.environ.get("GLM53_TF_BATCH_ADMIT_GB", "") or 1.0) * 2**30
        # patches/0550: admission counts reclaimable page cache (GLM53_TF_ADMIT_MEM, ``memsafe``) and says when it
        # waits for memory (rank 0); the caching allocator's unused cache is trimmed before a prefill piece
        # (GLM53_TF_ALLOC_TRIM_GB, both ranks, each for itself)
        from . import memsafe

        self.admit_policy = memsafe.admit_policy()
        self.admit_log = memsafe.AdmitLog()
        self.trimmer = memsafe.Trimmer()
        self.last_verify = False
        # patches/0180: the session store behind the slots (``attach_store``, after the batcher); its budget is set
        # aside now, so the slots never take the memory the store may fill later
        from . import sessions

        self.store = None
        self.store_budget = sessions.budget_bytes() if sessions.batch_enabled() else 0
        self.following = False                    # patches/0180: True in ``follow`` (rank 1's side of every message)
        # -- slots: what fits, the same count on both ranks ------------------------------------------------------
        b = e.buf
        self.states: list[State] = [e.st]
        self.graphs = [e.graphs]
        self.drafters = [g.drafter]
        free0 = self._free()
        per = self._slot_estimate()
        for _ in range(1, n):
            # both ranks decide each slot together: making one captures graphs with collectives in them
            before = self._free()
            ok = g._gather_ints([int(before - per - self.store_budget >= self.reserve)])
            if not (ok[0][0] and ok[1][0]):
                break
            st = self._new_state()
            self.states.append(st)
            self.graphs.append(self._slot_graphs(st, GRAPH_ROWS))
            self.drafters.append(_drafter_view(g.drafter) if g.drafter is not None else None)
            torch.cuda.synchronize()
            per = max(per, before - self._free())
        fit = len(self.states)
        if fit < n and g.rank == 0:
            print(f"[tensorfold] GLM53_TF_BATCH={n}: only {fit} sequence(s) fit with GLM53_TF_BATCH_RESERVE_GB="
                  f"{self.reserve / 2**30:g} GB kept free ({per / 2**30:.2f} GB a sequence at {e.st.capacity} cache "
                  "slots)" + (f" and the session store's {self.store_budget / 2**30:g} GiB (GLM53_TF_SESSION_GIB)"
                              if self.store_budget else "") + "; serving that many", flush=True)
        self.n = n = fit
        # patches/0290: with the KV pool, slots hold pages only while (and after) a request uses them; admission
        # reserves a request's pages (``_plan``), so every slot starts empty and unreserved
        self.kvp = w.meta.get("kv_pool")
        self.pool_check = self.kvp is not None and kvpool.check_on()
        self._spills = []
        if self.kvp is not None:
            for st in self.states:
                st.pages.truncate(0)
                st.pages.reserve(0)
        kda_layers = [l for l in w.layers if l.kind == "kda"]
        rows = min(b.rows, n * MAX_ROWS)
        if kda_layers:
            b.bproj = torch.empty((rows, kda_layers[0].kda.proj.n), dtype=torch.bfloat16, device=w.device)
            b.bkout = torch.empty((rows, e.st.scratch_set.out.shape[2]), dtype=torch.bfloat16, device=w.device)
        # drafter backlogs per slot (auto_decode's m_rows / f_taps, BACKLOG rows)
        if gpuround.env().resident:                 # patches/0450: one tensor each (+ a row the device may discard)
            self.m_all = torch.zeros((n, BACKLOG + 1, w.cfg.hidden), dtype=torch.bfloat16, device=w.device)
            self.m_rows = [self.m_all[i, :BACKLOG] for i in range(n)]
            self.f_all = (torch.zeros((n, BACKLOG + 1, len(b.taps) * w.cfg.hidden), dtype=torch.bfloat16,
                                      device=w.device) if b.taps else None)
            self.f_taps = [self.f_all[i, :BACKLOG] if self.f_all is not None else None for i in range(n)]
        else:
            self.m_rows = [torch.zeros((BACKLOG, w.cfg.hidden), dtype=torch.bfloat16, device=w.device)
                           for _ in range(n)]
            self.f_taps = [torch.zeros((BACKLOG, len(b.taps) * w.cfg.hidden), dtype=torch.bfloat16, device=w.device)
                           if b.taps else None for _ in range(n)]
        self.pool = torch.cuda.graph_pool_handle() if self.use_graphs else None
        if self.overlap.plan:                          # patches/0370: the plan rider's staging (same size both ranks)
            self.rider_n = dover.rider_size(n)
            self.rider_host = torch.zeros((self.rider_n,), dtype=torch.int32).pin_memory()
            self.rider_dev = torch.zeros((self.rider_n,), dtype=torch.int32, device=w.device)
        # patches/0280: the route sources of a padded round (host staging + the device table graphs read)
        self.src_host = self.src_dev = None
        if self.tie:
            self.src_host = torch.zeros((rows,), dtype=torch.int64).pin_memory()
            self.src_dev = torch.arange(rows, dtype=torch.int64, device=w.device)
        self.multi: dict[tuple, torch.cuda.CUDAGraph] = {}
        self.counts = collections.Counter()                        # graph / capture / eager rounds, pieces
        self.caches: list[list] = [[] for _ in range(n)]           # decode.Snapshot entries per slot
        self.used = [0] * n                                        # round of each slot's last admission
        self.seqs: list[Seq | None] = [None] * n
        self.round = 0
        self.eos = tuple(w.cfg.eos)
        self.costs = g.costs
        self.round_costs = batchplan.RoundCosts(self.costs, self.row_ms)
        self.defaults = g._knob_state()                            # the load-time knob values (rank 0's requests)
        self.log: collections.deque = collections.deque(maxlen=256)    # finished requests (both ranks; tests)
        self.trace: collections.deque = collections.deque(maxlen=4096)  # (round, pieces, verified slots)
        self.last_piece_s = 0.0
        self.last_kind = "alone"
        if g.rank == 0 and torch.cuda.is_available():
            torch.cuda.synchronize()
            extra = (free0 - self._free()) / 2**30
            print(f"[tensorfold] batching {n} requests: {n - 1} extra sequence(s), {extra:.2f} GB on this rank "
                  f"({e.st.capacity} cache slots each; prefill pieces of {self.piece} tokens, at most "
                  f"{self.fair.share:g} of the time while others decode)", flush=True)
            if self.graph_policy == batchplan.GRAPHS_LONE:  # patches/0510
                print("[tensorfold] batch graphs (patches/0510): lone rounds only (GLM53_TF_BATCH_GRAPHS=lone; rounds "
                      "of 2+ slots run eagerly)", flush=True)
            if self.solo:                               # patches/0335
                most = int(getattr(e, "prefill_max", e.rows))
                print(f"[tensorfold] GLM53_TF_SOLO_PIECE={self.solo}: a fast prompt alone in the batch prefills in "
                      f"pieces of {self.solo} tokens (chunks of at most {most} rows"
                      + ("" if most >= self.solo else "; raise GLM53_TF_PREFILL_ROWS_MAX for chunks as big as the "
                         "piece") + ")", flush=True)
            print(f"[tensorfold] memory safety (patches/0550): selection scratch {sparse.SCRATCH} (key blocks up to "
                  f"{sparse.SELECT_MB} MiB), admission {self.admit_policy} (GLM53_TF_ADMIT_MEM"
                  + (f"; page cache counted above max(Mapped, {memsafe.cache_keep_bytes() / 2**30:g} GiB) less dirty, "
                     f"at least {min(memsafe.free_floor_bytes(), int(self.admit_min)) / 2**30:g} GiB free now"
                     if self.admit_policy == "available" else "") + "), allocator trim "
                  + (f"over {self.trimmer.limit / 2**30:g} GiB unused" if self.trimmer.limit > 0 else "off"),
                  flush=True)
            if self.overlap.on:                        # patches/0370
                print(f"[tensorfold] decode overlap (patches/0370): {self.overlap.describe()}"
                      + (f", switch interval {self.overlap.switch_us} us" if self.overlap.gil else ""), flush=True)
            if self.kvp is not None:
                kp = self.kvp
                print(f"[tensorfold] KV pool (patches/0290): {kp.npages * kp.page} tokens in {kp.npages} pages of "
                      f"{kp.page}, {kp.nbytes() / 2**30:.2f} GiB on this rank, shared by the {n} slots (each up to "
                      f"{e.st.capacity} tokens); a request reserves prompt + max_tokens + {kp.slack} tokens of "
                      "pages at admission, idle slots' pages are spilled when short, else it waits", flush=True)
        self.cv = threading.Condition()
        self.stopping = False
        self.queue: collections.deque[Job] = collections.deque()
        self.error: BaseException | None = None
        self.thread = None
        # patches/0450 (GLM53_TF_GPU_ROUND ``resident``): steady decode as GPU-resident rounds; its drain flag rides on
        # the sampler exchange as 0370's plan does, so it needs the ``plan`` part (both ranks agree on both knobs)
        if gpuround.env().resident:
            if not self.overlap.plan:
                if g.rank == 0:
                    print("[tensorfold] GLM53_TF_GPU_ROUND 'resident' needs GLM53_TF_DECODE_OVERLAP's 'plan' part: "
                          "resident rounds are off", flush=True)
            else:
                from .resident import Resident

                self.resident = Resident(self)
                if g.rank == 0:
                    print(f"[tensorfold] GPU-resident rounds (patches/0450): {gpuround.env().describe()}", flush=True)
        if g.rank == 0:
            self.thread = threading.Thread(target=self._serve, name="glm-batch", daemon=True)
            self.thread.start()

    # -- patches/0180: the session store behind the slots ----------------------------------------------------------
    def attach_store(self, store) -> None:
        """Serve the slots from ``store`` (``sessions.SessionStore`` built on slot 0's state): a live map per slot; the
        store keeps at least the admission minimum free when it grows."""

        store.bind_slots(self.states)
        store.reserve = max(store.reserve, int(self.admit_min))
        self.store = store

    def _headroom(self) -> int:
        return self.store.headroom() if self.store is not None else 0

    def _session_plan(self, job: Job, slot: int, cached: int, free: list[int],
                      others: list[list[int]]) -> tuple[int, int]:
        """Rank 0: the store's plan for an admission (``job.sess``); when a stored entry resumes more than the slot's
        own snapshots, the free slot to restore it into and its length. ``others``: the prompts in flight beside it
        (fork points too: their entries are not stored yet). -> (slot, resume length)."""

        g = self.g
        _, need_mtp, need_f = g._drafters(job.code)
        sess = self.store.plan(job.prompt, self._grid(job.values), need_mtp, need_f and g.drafter is not None,
                               cached, job.draft, others)
        job.sess = sess
        e = sess.entry if sess.entry is not None else sess.disk     # patches/0250: or an entry on disk
        if e is None:
            return slot, cached
        slot = batchplan.place_entry(free, {s: self.store.overlap(e, s) for s in free},
                                     {s: self.used[s] for s in free})
        if sess.disk is not None and self.kvp is None:  # patches/0290: with the pool the read starts in ``_admit``
            self._prefetch(slot, sess)                  # (after this round's spills and the slot's reservation)
        return slot, len(e.ids)

    def _share_wait(self, job: Job, admits: list) -> bool:
        """patches/0310 (GLM53_TF_PREFIX_SHARE_WAIT, rank 0): whether ``job`` should wait a round because a request in
        flight (prefilling, or admitted earlier this round) of the same prefill mode and drafters will soon store a
        mark inside the prefix they share, at least ``fork_min`` past what ``job`` can resume now. It then resumes
        there instead of prefilling the same tokens beside it. Never waits on a request that is not prefilling (its
        marks are stored), so it ends when the partner passes the mark (stored or not), finishes, or is cancelled."""

        share = getattr(self.store, "share", None) if self.store is not None else None
        if share is None or not share.wait or not job.draft:
            return False
        from .sessions import common_prefix

        g = self.g
        grid = self._grid(job.values)
        _, m, f = g._drafters(job.code)
        partners = [(s.job, s.done, s.marks) for s in self.seqs if s is not None and s.stepper is None
                    and not s.job.cancel]
        partners += [(j, c, tuple(j.sess.marks) if j.sess is not None else ()) for _, c, j in admits]
        have = None
        for pj, done, marks in partners:
            if not marks or not pj.draft or self._grid(pj.values) != grid:
                continue
            _, pm, pf = g._drafters(pj.code)
            if (m and not pm) or (f and not pf):         # its snapshots lack a draft cache this request needs
                continue
            ahead = [q for q in marks if done < q < len(job.prompt)]
            if not ahead:
                continue
            d = common_prefix(job.prompt, pj.prompt)
            ahead = [q for q in ahead if q <= d]
            if not ahead:
                continue
            if have is None:
                have = self._resumable(job, grid, m, f and g.drafter is not None)
            if max(ahead) >= have + self.store.fork_min:
                return True
        return False

    def _resumable(self, job: Job, grid: int, need_mtp: bool, need_f: bool) -> int:
        """patches/0310: the longest prefix ``job`` could resume now (any slot's snapshot, the store, its disk)."""

        best = 0
        for s in range(self.n):
            snap = self._resume(s, job.prompt, job.code, grid)
            if snap is not None:
                best = max(best, len(snap.ids))
        e = self.store.index.find(job.prompt, grid, need_mtp, need_f)
        if e is not None:
            best = max(best, len(e.ids))
        disk = getattr(self.store, "disk", None)
        if disk is not None:
            d = disk.index.find(job.prompt, grid, need_mtp, need_f)
            if d is not None:
                best = max(best, len(d.ids))
        return best

    def _prefetch(self, slot: int, sess) -> None:
        """patches/0250: start reading a disk entry into ``slot`` when the slot is idle (a slot freed by a cancel in
        this round is read in ``_admit`` instead); its own snapshots no longer match its caches."""

        if self.seqs[slot] is None:
            self.caches[slot] = []
            with self.store.on(slot):
                self.store.prefetch(sess)

    def _save(self, slot: int, snaps: list) -> None:
        """Store snapshots of ``slot``'s live state (both ranks, in the round's order: rank 0 decides and sends its
        decisions, the follower applies them). The side is the batcher's role (``follow``), as for the round plans,
        not ``g.rank``."""

        if self.store is None:
            return
        snaps = [x for x in snaps if x is not None]
        if not snaps:
            return
        with self.store.on(slot):
            if self.following:
                self.store.save_all(snaps, forced=self.g._share(None))
            else:
                self.g._share(self.store.save_all(snaps))

    def _mem_ok(self, job: Job, admits: list) -> bool:
        """patches/0550 (rank 0): whether ``job`` may start beside running (or this round's) requests: the usable
        memory (``memsafe.view``: free + the allocator's cache, + the page cache credit with GLM53_TF_ADMIT_MEM=
        available) less the store's unused budget covers GLM53_TF_BATCH_ADMIT_GB and the selection scratch the
        prompt still needs (the largest of this round's), and something is free right now. Logs a wait."""

        from . import memsafe

        need = self.admit_min + max([self._scratch_need(j) for j in [job] + [a[2] for a in admits]])
        v = memsafe.view(self._free(), 0, self.admit_policy)
        headroom = self._headroom()
        if memsafe.admit_ok(v, headroom, need, min(memsafe.free_floor_bytes(), int(self.admit_min))):
            return True
        self.counts["mem_deferred"] += 1
        if self.admit_log is not None:
            self.admit_log.waiting(v, headroom, need, len(self.queue))
        return False

    def _scratch_need(self, job: Job) -> int:
        """patches/0550: bytes the prefill selection's scratch still grows by for ``job``'s prompt (0: none)."""

        try:
            e, st = self.g.e, self.states[0]
            if st.index is None or sparse.SCRATCH == "off" or sparse.SELECT != "blocked" \
                    or len(job.prompt) <= self.g.w.cfg.dense_limit or not torch.cuda.is_available():
                return 0
            from . import memsafe

            return memsafe.scratch_growth(len(job.prompt), sparse.scratch_bytes(self.g.w.device), e.buf.rows,
                                          st.index[0][2].shape[0] - 2, sparse.SELECT_MB, sparse.SCRATCH)
        except Exception:  # noqa: BLE001 - host fakes without these parts: nothing to reserve
            return 0

    # -- slots ----------------------------------------------------------------------------------------------------
    @staticmethod
    def _free() -> int:
        """Free device memory, counting what the caching allocator holds but does not use."""

        if not torch.cuda.is_available():
            return 0
        return torch.cuda.mem_get_info()[0] + torch.cuda.memory_reserved() - torch.cuda.memory_allocated()

    def _slot_estimate(self) -> int:
        """Bytes of one more slot before the first is made: the state's caches (window rows excepted), its DFlash2
        context, and some room for its graphs; measured on the first slot after that."""

        st, seen, total = self.g.e.st, set(), 0
        tensors = list(st.kc) + list(st.vc) + [st.rec, st.conv]
        for name in ("mtp_kc", "mtp_vc", "iring"):
            if hasattr(st, name):
                tensors.append(getattr(st, name))
        if st.index is not None:
            tensors += [t for trio in st.index for t in trio]
        for t in tensors:
            if kvpool.is_paged(t):              # patches/0290: in the pool, allocated once for every slot
                continue
            if t.data_ptr() not in seen:
                seen.add(t.data_ptr())
                total += t.numel() * t.element_size()
        d = self.g.drafter
        if d is not None:
            total += sum(t.numel() * t.element_size() for t in list(d.kc) + list(d.vc))
        return total + (64 << 20)

    def _new_state(self) -> State:
        """A slot's State: the engine's capacity and rings, window-sized KDA projection / replay rows (a prefill
        borrows slot 0's, ``_borrow``)."""

        from .engine import MAX_ROWS

        g = self.g
        e, w = g.e, g.w
        st = State(w, e.st.capacity, min(MAX_ROWS, e.rows))
        ring = sparse.index_ring_rows(e.rows)
        if st.index_ring and ring > st.index_ring:
            # patches/0065: the index rings hold a prefill chunk of the engine's rows (a ring as long as the capacity
            # or longer is a full cache: positions never wrap); the committed tail slot does not depend on the ring
            n, dim = st._tail_n, st.iring.shape[3]
            st.iring = torch.zeros((2, n, ring, dim), dtype=st.iring.dtype, device=st.iring.device)
            st.index_ring = ring
            st.index[:n] = [(st.iring[0, i], st.iring[1, i], st.index[i][2]) for i in range(n)]
        return st

    def _slot_graphs(self, st: State, rows):
        """The MTP head's graphs for another slot's state (patches/0050's long-context ones too, lazily); its
        main-model rounds go through ``compute_multi``."""

        e = self.g.e
        if e.graphs is None:
            return None
        from .graphs import Graphs

        return Graphs(SimpleNamespace(w=e.w, st=st, buf=e.buf, mbuf=e.mbuf), main_rows=(), mtp_rows=rows,
                      long_rows=e.long_rows)

    @contextmanager
    def _on(self, slot: int):
        """The engine as slot ``slot``'s sequence: its State and its graphs (prefill, drafts, snapshots)."""

        e = self.g.e
        saved = e.st, e.graphs
        e.st, e.graphs = self.states[slot], self.graphs[slot]
        try:
            yield e
        finally:
            e.st, e.graphs = saved

    @contextmanager
    def _borrow(self, slot: int):
        """Slot 0's KDA projection and replay rows for a prefill on ``slot`` (they only hold one forward's rows
        between that forward and its commit, and no round is in flight during a prefill piece)."""

        st, big = self.states[slot], self.states[0]
        if st is big or st.proj.shape[1] >= big.proj.shape[1]:
            yield
            return
        saved = st.proj, st.scratch_set, st.scratch
        st.proj, st.scratch_set, st.scratch = big.proj, big.scratch_set, big.scratch
        try:
            yield
        finally:
            st.proj, st.scratch_set, st.scratch = saved

    # -- rank 0: requests in ---------------------------------------------------------------------------------------
    def _job(self, prompt, max_tokens, sampling, draft: bool, spec: str | None, stop_eos: bool, knobs=None,
             background: bool = False) -> Job:
        """A request as ``GlmEngine.generate`` resolves it (policy code, cost flag, lookup settings, knobs), on rank
        0's load-time defaults."""

        from . import depth as depth_mod, knobs as knobs_mod
        from .engine import DFLASH_POLICY, encode_policy
        from .lookup import env_settings, with_lookup

        g = self.g
        asked = g.parse_knobs(knobs)
        prompt = [int(t) for t in prompt]
        if not prompt:
            raise ValueError("an empty prompt")
        if len(prompt) >= g.limit:
            raise ValueError(f"prompt of {len(prompt)} tokens: this engine serves contexts up to {g.limit}")
        max_tokens = max(1, min(int(max_tokens), g.limit - len(prompt)))
        spec = "0" if not draft or g.serial_only else (spec or g.policy)
        code = encode_policy(spec)
        depth_cost = asked["depth"] == "cost" if "depth" in asked else depth_mod.env_cost()
        cost = int(str(spec).strip() == "o" or code[0] == depth_mod.OPT_KIND or (code[0] in (4, 5) and depth_cost))
        code = g._effective(code)
        if cost:
            code = depth_mod.as_cost(code)
        code = with_lookup(code, g.w.mtp is not None, encode_policy(DFLASH_POLICY))
        look = list(env_settings())
        look = [int(asked.get("lookup", look[0])), int(asked.get("lookup_min", look[1]))]
        if code[0] in (4, 5) and len(code) >= 6:
            code = code[:4] + look + code[6:]
        values = dict(self.defaults, **{k: v for k, v in asked.items() if k in knobs_mod.HEADER})
        if not g.longctx_buffers:
            values["longctx_graphs"] = 0
        values["calib_online"] = 0                # no online cost table in shared rounds
        job = Job(prompt, max_tokens, sampling, bool(stop_eos), bool(draft), list(code), str(spec), cost, values,
                  bool(background), out=queue.SimpleQueue(), submitted=time.perf_counter())
        job.echo = dict(values, lookup=look[0], lookup_min=look[1], depth="cost" if depth_cost else "threshold")
        return job

    def _submit(self, jobs: list[Job]) -> None:
        if self.error is not None:
            raise RuntimeError("the batch loop stopped") from self.error
        with self.cv:
            self.queue.extend(jobs)
            self.cv.notify()

    def _collect(self, job: Job, on_tokens: Callable | None) -> list[int]:
        got: list[int] = []
        listening = on_tokens is not None
        while True:
            item = job.out.get()
            if item is None:
                return got
            if isinstance(item, ValueError):
                raise item
            if isinstance(item, BaseException):
                raise RuntimeError("the batch loop failed") from item
            got.extend(item)
            if listening and on_tokens(item):
                job.cancel = True               # the client went away: the next round ends the request
                listening = False
                with self.cv:
                    self.cv.notify()

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft: bool = True, vis=None) -> dict[str, Any]:
        """``GlmEngine.generate`` in batch mode (the calling thread's policy, stop-at-EOS, knobs and priority;
        patches/0500: ``vis``, the prompt's image rows, encoded by the calling thread before the job is queued)."""

        req = self.g.request
        job = self._job(prompt, max_tokens, sampling, draft, getattr(req, "policy", None),
                        bool(getattr(req, "stop_eos", True)), getattr(req, "knobs", None),
                        bool(getattr(req, "background", False)))
        job.vision = vis
        lp = int(getattr(req, "logprobs", 0) or 0)            # patches/9001: rank 1 learns it from the header
        if lp > 0:
            job.values["logprobs"] = lp
            job.lp_sink = getattr(req, "lp_sink", None)
        if vis is not None and self.g.vision is not None:
            job.stats["vision"] = dict(self.g.vision.last)
        self._submit([job])
        self._collect(job, on_tokens)
        job.stats.update(policy=job.spec, drafts=job.draft, tf_knobs=job.echo)
        return job.stats

    def generate_batch(self, requests: list[dict]) -> list[tuple[list[int], dict]]:
        """Requests queued together (so they share rounds from the first): dicts with prompt, max_tokens and
        optionally sampling, policy, draft, stop_eos (False), knobs, background, on_tokens. -> [(tokens, stats)]."""

        jobs = [self._job(r["prompt"], r["max_tokens"], r.get("sampling"), r.get("draft", True), r.get("policy"),
                          r.get("stop_eos", False), r.get("knobs"), r.get("background", False)) for r in requests]
        self._submit(jobs)
        out = []
        for r, job in zip(requests, jobs):
            tokens = self._collect(job, r.get("on_tokens"))
            job.stats.update(policy=job.spec, drafts=job.draft, tf_knobs=job.echo)
            out.append((tokens, job.stats))
        return out

    def stop(self) -> None:
        """Rank 0: end the loop thread once nothing runs or waits (tests; a server runs until the process ends)."""

        with self.cv:
            self.stopping = True
            self.cv.notify()
        if self.thread is not None:
            self.thread.join(timeout=60)

    # -- the round loop ------------------------------------------------------------------------------------------------
    def _serve(self) -> None:
        """Rank 0's loop thread: plan a round (shared with rank 1), run it."""

        torch.cuda.set_device(0)
        cpupin.serving()                        # patches/0370 (GLM53_TF_CPU_PIN): this thread on its core
        try:
            with torch.no_grad():
                while True:
                    cpupin.tick()
                    plan = self._take_ahead()   # patches/0370: decided and shared inside the last round
                    if plan is None:
                        plan = self._plan()
                    if plan is None:
                        return
                    cancels, admits, pieces = plan
                    t0 = time.perf_counter()
                    self.last_piece_s = 0.0
                    self._execute(cancels, admits, pieces)
                    self._go_resident()         # patches/0450
                    decode_s = max(time.perf_counter() - t0 - self.last_piece_s, 0.0) if self.last_verify else 0.0
                    prefilling = any(s is not None and s.stepper is None for s in self.seqs)
                    self.fair.after(self.last_piece_s, decode_s, prefilling)
        except BaseException as exc:            # noqa: BLE001 - every waiting request must hear about it
            traceback.print_exc()
            self.error = exc
            self._flush()                       # patches/0370: tokens decided before the failure first
            with self.cv:
                jobs = [s.job for s in self.seqs if s is not None] + list(self.queue)
                self.queue.clear()
            for job in jobs:
                if job.out is not None:
                    job.out.put(exc)

    def follow(self) -> None:
        """Rank 1: run every round rank 0 plans, forever."""

        self.following = True                   # patches/0180: the save decisions come from rank 0 too
        cpupin.serving()                        # patches/0370 (GLM53_TF_CPU_PIN)
        with torch.no_grad():
            while True:
                cpupin.tick()
                ahead = self._take_ahead()      # patches/0370: rank 0's plan came with the last round's sampler
                if ahead is not None:
                    self._execute(*ahead)       # cancels only: no spills, no admissions, no pieces
                    self._go_resident()         # patches/0450
                    continue
                cancels, admits, pieces = batchplan.parse_plan(self.g._share(None))
                if self.kvp is not None:                # patches/0290: the slots rank 0 spills this round
                    got = self.g._share(None)
                    self._spills = list(got[1:1 + got[0]])
                jobs = []
                for slot, cached, header in admits:
                    job = self._job_from(header, self.g._share(None))
                    job.vision = vision_mod.exchange(self.g, job.prompt, None)     # patches/0500: rank 0's image rows
                    if self.store is not None:          # patches/0180: rank 0's session plan, after its digest check
                        job.sess = self.store.follow(self.g._share(None))
                        if job.sess.disk is not None and self.kvp is None:   # patches/0250: rank 1's read starts now
                            self._prefetch(slot, job.sess)
                    jobs.append((slot, cached, job))
                self._execute(cancels, jobs, pieces)
                self._go_resident()             # patches/0450

    def _plan(self) -> tuple[list[int], list[tuple[int, int, Job]], list[int]] | None:
        """Rank 0: this round's cancels, admissions and prefill piece, shared with rank 1 (None: ``stop``)."""

        self._flush()                           # patches/0370: nothing is launched before this plan is shared
        with self.cv:
            while not self.queue and not any(s is not None for s in self.seqs):
                if self.stopping:
                    return None
                self.cv.wait()
            if self.mpf_wait > 0:                       # patches/0560: idle, a few ms for more arrivals
                mpf.coalesce(self)
            for job in [j for j in self.queue if j.cancel]:          # gone before it started
                self.queue.remove(job)
                job.stats.update(cancelled=True)
                job.out.put(None)
            cancels = [i for i, s in enumerate(self.seqs) if s is not None and s.job.cancel]
            free = [i for i in range(self.n) if self.seqs[i] is None or i in cancels]
            requeue = []
            fg_waiting = any(not j.background for j in self.queue)
            v = batchplan.victim([(i, s.job.background, s.admitted) for i, s in enumerate(self.seqs)
                                  if s is not None and i not in cancels], fg_waiting, len(free))
            if v is not None:
                cancels.append(v)
                free.append(v)
                requeue.append(self.seqs[v].job)
            admits: list[tuple[int, int, Job]] = []
            spills: list[int] = []                     # patches/0290: idle slots whose pages go back to the pool
            avail = self.kvp.available() if self.kvp is not None else 0
            others = any(s is not None and i not in cancels for i, s in enumerate(self.seqs))
            # patches/0180: the prompts in flight (fork partners whose entries are not stored yet)
            flying = [s.job.prompt for i, s in enumerate(self.seqs) if s is not None and i not in cancels]
            waiting = list(self.queue)
            for i in batchplan.admission_order([j.background for j in waiting]):
                job = waiting[i]
                if not free:
                    break
                if self._share_wait(job, admits):          # patches/0310: a partner prefills its prefix
                    self.counts["prefix_wait"] += 1
                    job.stats["prefix_wait"] = job.stats.get("prefix_wait", 0) + 1
                    continue
                if (others or admits) and not self._mem_ok(job, admits):
                    self.counts["deferred"] += 1               # short of memory: it waits for a request to end
                    job.stats["mem_waits"] = job.stats.get("mem_waits", 0) + 1     # patches/0550
                    break
                slot, cached = self._place(job, free, spills)
                if self.kvp is not None:                       # patches/0290: its pages, reserved now
                    need = self._need_pages(job)
                    if need > self.kvp.npages:                 # more than the whole pool: it can never run
                        self.queue.remove(job)
                        self.counts["pool_refused"] += 1
                        if job.out is not None:
                            job.out.put(ValueError(
                                f"prompt ({len(job.prompt)}) + max_tokens ({job.max_tokens}) needs {need} KV pages of "
                                f"{self.kvp.page} tokens; the pool (GLM53_TF_KV_POOL_TOKENS) has {self.kvp.npages}"))
                        continue
                    got = batchplan.pool_spills(need, avail, [(s, self.states[s].pages.held, self.used[s])
                                                              for s in free if s not in spills], prefer=slot)
                    if got is None:                            # short even with every idle slot's pages: it waits
                        self.counts["pool_wait"] += 1
                        break
                    spills += got
                    avail += sum(self.states[s].pages.held for s in got) - need
                    if slot in got:                            # its own kept prefix went with its pages
                        slot, cached = self._place(job, free, spills)
                if self.store is not None:                     # patches/0180: a stored session may resume more
                    slot, cached = self._session_plan(job, slot, cached, free,
                                                      flying + [j.prompt for _, _, j in admits])
                free.remove(slot)
                self.queue.remove(job)
                admits.append((slot, cached, job))
                if self.admit_log is not None:                 # patches/0550: ends a wait for memory (if any)
                    self.admit_log.admitted()
            self.queue.extend(requeue)
        # the prefill piece: one prompt's next piece, when the others have had their share of the time
        admitted = {slot for slot, _, _ in admits}
        prefilling, decoding = [], False
        for i, s in enumerate(self.seqs):
            if s is None or i in cancels or i in admitted:
                continue
            if s.stepper is None:
                prefilling.append((i, len(s.job.prompt) - s.done, s.admitted))
            else:
                decoding = True
        for slot, cached, job in admits:
            prefilling.append((slot, len(job.prompt) - cached, self.round + 1))
        allow = self.fair.allow(decoding) if prefilling else False
        pieces = batchplan.pick_pieces(prefilling, allow, self.short) if prefilling else []
        if self.mpf_rows and prefilling:                # patches/0560: more pieces for one forward
            pieces = mpf.more_pieces(pieces, prefilling, allow, self.piece, self.mpf_rows)
        plan = batchplan.encode_plan(cancels, [(slot, cached, self._header(job)) for slot, cached, job in admits],
                                     pieces)
        self.g._share(plan)
        if self.kvp is not None:                        # patches/0290: this round's spills, before the admissions
            self._spills = list(spills)
            self.g._share([len(spills), *spills])
        for _, _, job in admits:
            self.g._share(job.prompt)
            job.vision = vision_mod.exchange(self.g, job.prompt, job.vision)    # patches/0500 (no image rows: nothing)
            if self.store is not None:                  # patches/0180: the plan, with rank 0's store digest
                self.g._share(job.sess.encode(self.store.index.digest()))
        return cancels, admits, pieces

    @staticmethod
    def _header(job: Job) -> list[int]:
        s = job.sampling
        return batchplan.encode_header(batchplan.Header(
            job.max_tokens, job.stop_eos, job.draft, s.seed if s else 0, s.temperature if s else 0.0,
            int(s.top_k) if s else 0, s.top_p if s else 1.0, job.cost, job.values, job.code))

    @staticmethod
    def _job_from(ints: list[int], prompt: list[int]) -> Job:
        from tensorfold.engine.exact_sampling import Sampling

        h = batchplan.decode_header(ints)
        sampling = Sampling(h.seed, h.temperature, h.top_k, h.top_p) if h.temperature > 0 else None
        return Job(list(prompt), h.max_tokens, sampling, h.stop_eos, h.draft, list(h.code), cost=h.cost,
                   values=dict(h.values))

    def _resume(self, slot: int, prompt: list[int], code: list[int], grid: int):
        """``GlmEngine._resume`` on ``slot``: the longest snapshot of a strict prefix of ``prompt`` whose draft caches
        fit the request's drafters and whose prefill mode (``grid``) is the request's."""

        _, mtp, dflash = self.g._drafters(code)
        best = None
        for snap in self.caches[slot]:
            fits = (not dflash or snap.drafter_end == len(snap.ids)) and (not mtp or snap.mtp_len >= 0) \
                and snap.grid == grid
            if fits and len(snap.ids) < len(prompt) and prompt[:len(snap.ids)] == snap.ids and (
                    best is None or len(snap.ids) > len(best.ids)):
                best = snap
        return best

    def _grid(self, values: dict) -> int:
        return self.g._grid(bool(values["fast_prefill"]), int(values["prefill_rows"]), bool(values["fp8_prefill"]),
                            int(values.get("b12x", 0)))                  # patches/0240

    def _place(self, job: Job, free: list[int], spilled: Sequence[int] = ()) -> tuple[int, int]:
        """(slot, resume length): the free slot resuming the longest prefix; else the least recently admitted one
        (its kept states are the likeliest to be stale). ``spilled`` (patches/0290): slots whose pages go back to the
        pool this round (their kept snapshots with them)."""

        best, length = None, 0
        if job.draft:
            grid = self._grid(job.values)
            for s in free:
                if s in spilled:
                    continue
                snap = self._resume(s, job.prompt, job.code, grid)
                if snap is not None and len(snap.ids) > length:
                    best, length = s, len(snap.ids)
        if best is None:
            best = min(free, key=lambda s: (self.used[s], s))
        return best, length

    # -- patches/0290: the KV page pool -------------------------------------------------------------------------------
    def _need_pages(self, job: Job) -> int:
        """Pages a request may write (prompt + max_tokens + the pool's slack, at most a slot's capacity): what its
        admission reserves, so it can always finish."""

        kp = self.kvp
        return kvpool.pages_for(kvpool.need_tokens(len(job.prompt), job.max_tokens, self.states[0].capacity,
                                                   kp.slack), kp.page)

    def _spill(self, slot: int) -> None:
        """An idle slot's pages back to the pool (both ranks, in the round's order). Its sessions are in the session
        store already (0180 saves every prompt, mark and reply snapshot when it is made; with 0250's write-through tier
        also on NVMe), so a later turn resumes from there; its own kept snapshots and live map go with the pages."""

        if self.seqs[slot] is not None:
            raise RuntimeError(f"KV pool: slot {slot} is busy, it cannot be spilled")
        self.caches[slot] = []
        if self.store is not None:
            self.store.forget(slot)
        st = self.states[slot]
        self.counts["pool_spilled_pages"] += st.pages.truncate(0)
        self.counts["pool_spills"] += 1
        st.set_pos(0)                           # it holds nothing now (the next request resets or restores it)
        st.set_mtp_len(0)

    def _release(self, slot: int, keep: int) -> None:
        """A request left the slot: pages past ``keep`` go back, its reservation ends (the slot is idle)."""

        if self.kvp is None:
            return
        sp = self.states[slot].pages
        sp.truncate(keep)
        sp.reserve(0)

    def _remember(self, slot: int, snap) -> None:
        cache = [c for c in self.caches[slot] if len(c.ids) < len(snap.ids) and snap.ids[:len(c.ids)] == c.ids]
        self.caches[slot] = cache[-1:] + [snap]

    def _emit(self, seq: Seq, tokens: list[int], lps: list | None = None) -> None:
        """Deliver tokens to the caller; a background request running again skips the ones it already sent.
        patches/9001: ``lps`` (one entry a token) go to the job's logprob sink before the tokens are delivered."""

        start = seq.emitted
        seq.emitted += len(tokens)
        job = seq.job
        if job.out is None or not tokens:
            return
        new = tokens[max(0, job.sent - start):]
        if new and job.lp_sink is not None:     # never raises here: a gap fails only that request (server side)
            ok = lps is not None and len(lps) == len(tokens)
            job.lp_sink.extend(lps[max(0, job.sent - start):] if ok else [None] * len(new))
        if new:
            if self.overlap.emit:               # patches/0370: after the next forward's launch (``_flush``)
                if self._held is None:
                    self._held = []
                self._held.append((job, list(new)))
            else:
                job.out.put(list(new))
            job.sent = start + len(tokens)

    def _flush(self) -> None:
        """patches/0370 (``emit``): the held tokens to their callers, in order."""

        held, self._held = self._held, None
        for job, tokens in held or ():
            job.out.put(tokens)

    # -- patches/0370 (``plan``): the next round's plan inside this round's sampler exchange ---------------------------
    def _plan_ahead(self) -> list[int] | None:
        """Rank 0, while this round's verify forward runs: the next round's plan when it can only be cancels (every
        slot in flight decodes, nothing waits, not stopping), else None (the loop plans at the top of the next
        round, as before). A slot that ends in this round and is cancelled in the next is skipped by ``_execute``."""

        with self.cv:
            if self.queue or self.stopping:
                return None
            if any(s is not None and s.stepper is None for s in self.seqs):
                return None                     # a prompt prefills: its piece is the loop's decision
            cancels = [i for i, s in enumerate(self.seqs) if s is not None and s.job.cancel]
        return batchplan.encode_plan(cancels, [], [])

    def _rider(self) -> torch.Tensor:
        plan = None if self.following else self._plan_ahead()
        self.rider_host.numpy()[:] = dover.rider_encode(plan, self.rider_n)
        self.rider_dev.copy_(self.rider_host, non_blocking=True)
        return self.rider_dev

    def _take_ahead(self):
        plan, self._ahead = self._ahead, None
        if plan is None:
            return None
        cancels, admits, pieces = batchplan.parse_plan(plan)
        if admits or pieces:
            raise RuntimeError("a round plan rider held admissions or pieces: both ranks must run the same patched "
                               "TensorFold")
        return cancels, [], []

    # -- a round (both ranks) --------------------------------------------------------------------------------------------
    def _execute(self, cancels: list[int], admits: list[tuple[int, int, Job]], pieces: list[int]) -> None:
        self.round += 1
        self.last_verify = False
        for s in cancels:
            if self.seqs[s] is not None:
                self._finish(s, cancelled=True)
        spills, self._spills = list(self._spills), []
        for s in spills:                        # patches/0290: before the admissions that need their pages
            self._spill(s)
        for slot, cached, job in admits:
            self._admit(slot, cached, job)
        waiting = [q for q in self.seqs if q is not None and q.stepper is not None] if pieces else []
        for grp in mpf.groups(self, pieces) if self.mpf_rows else [[s] for s in pieces]:     # patches/0560
            t0 = time.perf_counter()
            if len(grp) > 1:
                mpf.run(self, grp)              # one forward for the group; each piece then ends in ``_piece``
            for s in grp:
                self._piece(s)
            self.last_piece_s += time.perf_counter() - t0
        for q in waiting:                       # patches/0200: decode time lost to other requests' pieces
            q.kinds["piece_ms"] += self.last_piece_s * 1e3
        active = [s for s in range(self.n) if self.seqs[s] is not None and self.seqs[s].stepper is not None]
        self.trace.append((self.round, tuple(pieces), tuple(active)))
        if active:
            self.last_verify = True
            self._verify(active)
        if self.pool_check and not self.kvp.null_clean():     # patches/0290, GLM53_TF_KV_POOL_CHECK=1
            raise RuntimeError(f"KV pool: the null page was written in round {self.round} (an unmapped row)")

    def _admit(self, slot: int, cached: int, job: Job) -> None:
        if self.seqs[slot] is not None:
            raise RuntimeError(f"slot {slot} is busy")
        if self.kvp is not None:                # patches/0290: the request's reservation; rows past its resume are dead
            sp = self.states[slot].pages
            sp.reserve(self._need_pages(job))
            sp.truncate(cached)
            job.stats["kv_pages"] = sp.quota
        hit = None
        sess = job.sess if self.store is not None else None
        entry = sess.entry if sess is not None else None
        if entry is not None:                   # patches/0180: a stored session, copied into this slot's caches
            if len(entry.ids) != cached or job.prompt[:cached] != entry.ids:
                raise RuntimeError(f"session entry {entry.id} does not hold the {cached} tokens slot {slot} resumes "
                                   "from")
            _, _, use_dflash = self.g._drafters(job.code)
            with self.store.on(slot):
                hit = self.store.restore(entry, self.drafters[slot] if use_dflash else None)
            self.caches[slot] = []              # the slot's own snapshots no longer match its caches
        elif sess is not None and sess.disk is not None:        # patches/0250: read from disk into this slot
            if len(sess.disk.ids) != cached or job.prompt[:cached] != sess.disk.ids:
                raise RuntimeError(f"session disk entry {sess.disk.id} does not hold the {cached} tokens slot {slot} "
                                   "resumes from")
            _, _, use_dflash = self.g._drafters(job.code)
            self.caches[slot] = []
            with self.store.on(slot):
                hit = self.store.restore_disk(sess, self.drafters[slot] if use_dflash else None)
            job.stats["disk"] = dict(self.store.disk.last)
            if hit is None:                     # a rank failed to read it: both prefill from scratch
                cached = 0
            else:
                job.stats["restored_disk"] = sess.disk.id
        elif cached:
            hit = next((c for c in self.caches[slot] if len(c.ids) == cached and job.prompt[:cached] == c.ids), None)
            if hit is None:
                raise RuntimeError(f"no snapshot of {cached} tokens on slot {slot} to resume from")
        # the request writes the slot's attention caches from its resume point on: longer snapshots go
        self.caches[slot] = [c for c in self.caches[slot] if len(c.ids) <= cached]
        if self.store is not None:
            with self.store.on(slot):
                self.store.begin(hit, cached)
        self.used[slot] = self.round
        self.seqs[slot] = Seq(job, slot, self.round, resume=hit, done=cached, grid=self._grid(job.values),
                              t0=time.perf_counter(), marks=tuple(sess.marks) if sess is not None and job.draft
                              else ())
        job.stats.update(cached=cached, slot=slot, queued_s=round(time.perf_counter() - job.submitted, 3)
                         if job.submitted else 0.0)
        job.stats["marks"] = len(self.seqs[slot].marks)          # patches/0300: the store's checkpoints it takes
        if self.kvp is not None:
            job.stats["kv_free"] = self.kvp.available()       # patches/0300: pool pages left after its reservation
        if entry is not None:
            job.stats["restored"] = entry.id

    def _piece(self, slot: int) -> None:
        """The next piece of the slot's prompt through ``decode.prefill`` (the request's knobs, resumed from the
        previous piece's snapshot); after the last one, the first token and the first drafts."""

        from . import fastpf, pfgrid
        from .decode import prefill, take_snapshot

        g = self.g
        seq = self.seqs[slot]
        job = seq.job
        _, use_mtp, use_dflash = g._drafters(job.code)
        drafter = self.drafters[slot] if use_dflash else None
        n = len(job.prompt)
        self.counts["pieces"] += 1
        from . import memsafe

        if memsafe.trim_torch(self.trimmer):            # patches/0550: the allocator's unused cache back to the device
            self.counts["trims"] += 1
        try:
            with self._on(slot) as e, self._borrow(slot), g._knobs(job.values):
                # patches/0180: pieces end on the chunk grid and on the snapshot grid (the snapshot the next resumes)
                grid = batchplan.piece_grid(fastpf.grid(e.prefill_rows), int(getattr(e, "snap_grid", 0) or 0)) \
                    if e.fast_prefill else 0
                # patches/0540: the piece that ends at the prompt's end keeps the prompt's snapshot strictly before
                # it (a resend resumes all but its last grid step); earlier pieces keep theirs at their end, which the
                # next piece resumes (``decode._prefill`` applies it to a prefill of exactly ``e.snap_before`` tokens)
                e.snap_before = n if job.draft and pfgrid.before_end() else 0
                # patches/0335: alone in the batch -> the solo piece (fast prefills only; both ranks see the same slots)
                solo = batchplan.solo_piece(self.solo, self.piece, grid,
                                            [i for i, q in enumerate(self.seqs) if q is not None and i != slot])
                if solo != self.piece:
                    seq.solo += 1
                end = batchplan.piece_end(seq.done, n, solo, grid)
                e.checkpoints = seq.marks           # patches/0180: the store's marks inside this piece
                e.lp_want, e.lp_last = int(job.values.get("logprobs", 0) or 0), None     # patches/9001
                try:
                    with vision_mod.active(job.vision):     # patches/0500: the prompt's image rows (None: none)
                        first = (mpf.take(self, slot) or prefill)(e, job.prompt[:end], job.sampling, mtp=use_mtp,
                                                                  drafter=drafter, resume=seq.resume)   # 0560
                finally:
                    e.checkpoints = ()
                    first_lp, e.lp_want, e.lp_last = e.lp_last, 0, None                  # patches/9001
                saves = list(getattr(e, "mark_snaps", ()) or ()) if job.draft else []
                before, e.snap_before = job.draft and end == n and pfgrid.before_end(), 0     # patches/0540
                e.mark_snaps = []
                seq.pieces += 1
                if end < n:
                    if grid:
                        snap = e.fast_snap          # the state at ``end`` (on the grid), resumable by fast pieces
                        if snap is None or len(snap.ids) != end:
                            raise RuntimeError(f"fast prefill piece to {end} left no grid snapshot there")
                    else:
                        snap = take_snapshot(e, job.prompt[:end], e.last_hidden if use_mtp else None, mtp=use_mtp,
                                             drafter=drafter)
                    seq.resume, seq.done = snap, end
                    if job.draft and end in seq.marks:      # a mark on the piece bound: the piece's own snapshot
                        saves.append(snap)
                    self._save(slot, saves)
                    return
                seq.done = n
                seq.resume = None
                tag = g._grid()                     # patches/0080: 0, or the fast chunk grid this prefill ran on
                if job.draft and (tag or before):
                    if e.fast_snap is not None:     # the state at the prompt's last grid point (0540: before its end)
                        self._remember(slot, e.fast_snap)
                        saves.append(e.fast_snap)
                elif job.draft:
                    snap = take_snapshot(e, job.prompt, e.last_hidden if use_mtp else None, mtp=use_mtp,
                                         drafter=drafter)
                    self._remember(slot, snap)
                    saves.append(snap)
                self._save(slot, saves)             # patches/0180: marks, then the prompt snapshot
                last_hidden = e.last_hidden
        except ValueError as exc:                   # the request's own problem (the same on both ranks)
            self.seqs[slot] = None
            self.caches[slot] = []
            if self.store is not None:
                self.store.forget(slot)
            self._release(slot, 0)                  # patches/0290
            if job.out is not None:
                job.out.put(exc)
            return
        job.stats.update(prefill_s=round(time.perf_counter() - seq.t0, 4), pieces=seq.pieces)
        if seq.solo:                                    # patches/0335
            job.stats["solo_pieces"] = seq.solo
        if tag:
            job.stats["fast_prefill"] = tag - tag % 64
            from . import pfgrid

            if pfgrid.is_fp8(tag):                                  # patches/0240: not every offset is FP8
                job.stats["fp8_prefill"] = 1
            if tag % 64 // pfgrid.B12X:
                job.stats["b12x"] = tag % 64 // pfgrid.B12X
        self._emit(seq, [first], [first_lp[0]] if first_lp else None)     # patches/9001: its logprobs if asked
        if self.emit_first and self._held:          # patches/0540: not held behind the round's other pieces
            self._flush()
        seq.stepper = Stepper(self, slot, job, first, last_hidden)
        if seq.stepper.done(self.eos):
            self._finish(slot)
            return
        with self._on(slot) as e:
            seq.stepper.propose(e)

    def _parity0(self, st: State) -> None:
        """Keep a slot's KDA states in buffer 0 for a graphed round (the same values, moved)."""

        if st.cur and st.cur[0]:
            st.rec[0].copy_(st.rec[1])
            st.cur = [0] * len(st.cur)

    def _go_resident(self) -> None:
        """patches/0450: run resident rounds when the round just run said so (both ranks, the same point)."""

        go, self._resident_go = self._resident_go, None
        if go and self.resident is not None:
            self._ahead = None                  # the plan that rode was empty: resident rounds take its place
            self.resident.run(go)
            self.counts["resident_runs"] += 1

    def _mode_at(self, st: State, pos: int, R: int) -> int | None:
        """``_mode`` of a window starting at ``pos`` (patches/0450: a position the host knows a bound of)."""

        saved = st.pos
        st.pos = pos
        try:
            return self._mode(st, R)
        finally:
            st.pos = saved

    def _mode(self, st: State, R: int) -> int | None:
        """A window's context mode for the graph key: 0 (every row dense), patches/0050's pool bucket (every row
        past 2,050, long-context graphs on), or None (eager: a window crossing 2,051, or no long graphs)."""

        e = self.g.e
        c = self.g.w.cfg
        if st.pos + R <= c.dense_limit:
            return 0
        if e.longctx and st.index is not None and st.pos >= c.dense_limit and e.buf.lc is not None \
                and R <= e.buf.lc.rows:
            return sparse.pool_bucket(st.pos + R, st.index[0][2].shape[0] - 2)
        return None

    def _pad(self, sts: Sequence[State], windows: list[list[int]], modes: list) -> tuple[list[list[int]], list]:
        """patches/0200 (GLM53_TF_BATCH_PAD): each window padded up to the next listed size with its last token, where
        that stays in the cache, in the slot's window rows and in the same context mode (``batchplan.padded``)."""

        c = self.g.w.cfg
        out_w, out_m = [], []
        for st, win, m in zip(sts, windows, modes):
            R = len(win)
            Rp = batchplan.padded(R, self.pad)
            if self._can_pad(st, R, Rp, m):
                win = list(win) + [win[-1]] * (Rp - R)
                self.counts["pad_rows"] += Rp - R
            out_w.append(win)
            out_m.append(m)
        return out_w, out_m

    def _can_pad(self, st: State, R: int, Rp: int, m) -> bool:
        """A window of R rows may run as Rp: it stays in the cache, in the slot's window rows, under the graph rows and
        in the same context mode (patches/0200's rule for GLM53_TF_BATCH_PAD, also 0280's buckets)."""

        c = self.g.w.cfg
        return Rp > R and m is not None and Rp <= self.graph_rows and Rp <= st.proj.shape[1] \
            and st.pos + Rp <= st.capacity and (st.index is not None or st.pos + Rp <= c.dense_limit) \
            and self._mode(st, Rp) == m

    def _bucket(self, sts: Sequence[State], windows: list[list[int]], modes: list) -> list[list[int]]:
        """patches/0280 (GLM53_TF_BATCH_BUCKETS): every window padded with its last token to the round's bucket
        (``batchplan.bucket_rows``) where ``_can_pad`` allows; a window that cannot stays as it is (its own key)."""

        Rp = batchplan.bucket_rows([len(x) for x in windows], self.buckets)
        out = []
        for st, win, m in zip(sts, windows, modes):
            R = len(win)
            if self._can_pad(st, R, Rp, m):
                win = list(win) + [win[-1]] * (Rp - R)
                self.counts["pad_rows"] += Rp - R
                self.counts["bucket_rows"] += Rp - R
            out.append(win)
        return out

    def graphs_on(self, n_active: int) -> bool:
        """patches/0510: may a round (or a batched drafter pass) over ``n_active`` slots use the batcher's graphs?"""

        return self.use_graphs and batchplan.graphs_for(self.graph_policy, n_active)

    def _forward(self, active: list[int], windows: list[list[int]]) -> torch.Tensor:
        """One forward over the active slots' windows -> their logits rows. ``last_rows``: the rows each slot had in
        it (its window's length, or more with GLM53_TF_BATCH_PAD: its rows start at the running sum of these)."""

        g = self.g
        e, w, b = g.e, g.w, g.e.buf
        e.rows_from = None                      # patches/0082: rows come from the window buffers again
        self.last_rows = [len(x) for x in windows]
        if active == [0]:
            with self._on(0):
                self.counts["alone"] += 1
                self.last_kind = "alone"
                return e.forward(windows[0])    # the engine's own graphs
        sts = [self.states[s] for s in active]
        real = [len(x) for x in windows]
        modes = [self._mode(st, len(x)) for st, x in zip(sts, windows)]
        graphs = self.graphs_on(len(active))    # patches/0510
        if self.pad and graphs and all(m is not None for m in modes):
            windows, modes = self._pad(sts, windows, modes)
        if self.buckets and graphs and len(active) > 1 and all(m is not None for m in modes):
            windows = self._bucket(sts, windows, modes)          # patches/0280
        self.last_rows = [len(x) for x in windows]
        T = stage_multi(w, sts, b, windows)
        Rs = [len(x) for x in windows]
        src = None
        if self.tie:                            # patches/0280: padded rows route as their window's last real row
            self.src_host[:T].numpy()[:] = batchplan.route_sources(real, Rs)
            self.src_dev[:T].copy_(self.src_host[:T], non_blocking=True)
            src = self.src_dev
        if graphs and all(m is not None for m in modes) and max(Rs) <= self.graph_rows:
            if self.parity_key:                 # patches/0200: the parities in the key, no state copies
                key = (tuple(active), tuple(Rs), tuple(modes), tuple(st.cur[0] if st.cur else 0 for st in sts))
            else:
                for st in sts:
                    self._parity0(st)
                key = (tuple(active), tuple(Rs), tuple(modes))
            graph = self.multi.get(key)
            if graph is not None:
                self.counts["graph"] += 1
                self.last_kind = "graph"
                graph.replay()
                return b.logits[:T]

            def run() -> None:
                b.route_src = src               # patches/0280 (None: off); a graph holds the table's address
                try:
                    compute_multi(w, sts, b, Rs, npbs=modes)
                finally:
                    b.route_src = None

            run()                               # the round's result, through the code the graph holds
            self.counts["eager"] += 1
            self.last_kind = "eager"
            # patches/0200: a key is captured on its N-th sighting (GLM53_TF_BATCH_CAPTURE_AFTER; 1 = the first)
            if len(self.multi) < self.max_graphs and self.sightings.capture(key):
                self.last_kind = "capture"
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                try:
                    with torch.cuda.graph(graph, pool=self.pool, capture_error_mode="thread_local"):
                        run()
                except Exception as exc:  # noqa: BLE001 - keep serving eagerly
                    self.use_graphs = False
                    print(f"[tensorfold] batched round CUDA graph capture failed ({type(exc).__name__}: {exc}); "
                          "multi-sequence rounds run eagerly", file=sys.stderr, flush=True)
                    torch.cuda.synchronize()
                else:
                    self.multi[key] = graph
                    self.counts["capture"] += 1
            return b.logits[:T]
        self.counts["eager"] += 1
        self.last_kind = "eager"
        return compute_multi(w, sts, b, Rs, nchs=[chunks_for(st, R) for st, R in zip(sts, Rs)],
                             host_pos=[st.pos for st in sts])

    def _verify(self, active: list[int]) -> None:
        g = self.g
        w = g.w
        seqs = [self.seqs[s] for s in active]
        t_round = time.perf_counter()
        windows = [q.stepper.window() for q in seqs]
        self.last_rows = [len(x) for x in windows]
        logits = self._forward(active, windows)
        ov = self.overlap
        if ov.emit:                             # patches/0370: the GPU is busy now; the HTTP threads may run
            self._flush()
        ran = list(self.last_rows)              # patches/0200: rows per slot in the forward (padded or not)
        # patches/0280: tied padded rows read no experts of their own; the depth model prices the real rows
        priced = [len(x) for x in windows] if self.tie else ran
        parts = [(a, n, st_, bl) for (a, _, st_, bl), n in zip((q.stepper.part() for q in seqs), priced)]
        if not ov.sync:                         # patches/0370: timing only; the sampler's readback waits anyway
            torch.cuda.synchronize()
        specs, off = [], 0
        for s, q, win, n in zip(active, seqs, windows, ran):
            pos = self.states[s].pos
            specs.append((off, len(win), [pos + 1 + r for r in range(len(win))], q.job.sampling))
            off += n
            q.kinds[self.last_kind] += 1
            q.kinds["pad_rows"] += n - len(win)
        if ov.plan:                             # patches/0370: the next round's plan rides on this exchange
            sampled, words = sample_multi(w, logits, specs, rider=self._rider())
            self._ahead = dover.rider_decode(words)
        else:
            sampled = sample_multi(w, logits, specs)
        # patches/9001: the logprobs of the rows of requests that asked for them (both ranks, one exchange)
        lp_of: dict[int, list] = {}
        lp_rows = [(i, spec) for i, (q, spec) in enumerate(zip(seqs, specs)) if q.job.values.get("logprobs", 0)]
        if lp_rows:
            from .decode import logprob_rows

            got = logprob_rows(w, torch.cat([logits[off:off + R] for _, (off, R, _, _) in lp_rows]),
                               [t for i, _ in lp_rows for t in sampled[i]])
            at = 0
            for i, (_, R, _, _) in lp_rows:
                lp_of[i] = got[at:at + R]
                at += R
        shared = len(active) > 1
        tokens = 0
        from . import dump as dump_mod          # patches/0430
        dumped = [] if dump_mod.DUMP is not None else None
        for idx, (s, q, (off, R, _, _), rows, n, win) in enumerate(zip(active, seqs, specs, sampled, ran, windows)):
            pos = self.states[s].pos
            with self._on(s) as e:
                keep = q.stepper.accept(e, rows, off, shared, n) if n != R else q.stepper.accept(e, rows, off, shared)
            if dumped is not None:
                dumped.append((self.states[s], off, keep, win[:keep], pos))
            tokens += keep
            out = q.stepper.out
            bound = max(0, q.job.max_tokens - (len(out) - keep))
            lps = lp_of.get(idx)
            self._emit(q, rows[:keep][:bound], lps[:keep][:bound] if lps is not None else None)
        if dumped:                              # patches/0430: the kept rows as training records (reads only)
            dump_mod.decode_round(g.e, logits, dumped)
        self.round_costs.record(parts, tokens)
        verify_ms = (time.perf_counter() - t_round) * 1e3      # patches/0200: forward, sampling, accept, commit
        for q in seqs:
            q.kinds["verify_ms"] += verify_ms
        going = []
        for s, q in zip(active, seqs):
            if q.stepper.done(self.eos):
                self._finish(s)
            else:
                going.append((s, q))
        rows_of = {s: len(q.stepper.window()) for s, q in zip(active, seqs)}
        # patches/0200 (GLM53_TF_BATCH_MTP=1): the MTP chains of all slots drafted together after the loop
        chains = MtpChains() if self.batch_mtp and len(going) > 1 else None
        for s, q in going:
            st = q.stepper
            if st.opt is not None:              # batch-aware depths: this request's rows come after the others'
                others = sum(r for o, r in rows_of.items() if o != s and self.seqs[o] is not None)
                st.opt.verify = self.round_costs.table(others)
            t_draft = time.perf_counter()
            with self._on(s) as e:
                st.propose(e, chains) if chains is not None else st.propose(e)
            q.kinds["draft_ms"] += (time.perf_counter() - t_draft) * 1e3
        if chains is not None and chains.items:
            t_draft = time.perf_counter()
            mine = [it.stepper for it in chains.items]
            together = chains.run(self)
            share = (time.perf_counter() - t_draft) * 1e3 / len(mine)
            for s, q in going:
                if q.stepper in mine:
                    q.kinds["draft_ms"] += share
                    q.kinds["mtp_batched"] += int(together)
        # patches/0450: resident rounds follow when the next round's plan rode empty (rank 0: nothing waits, no cancel)
        # and every slot can run them; both ranks decide from the same shared values
        self._resident_go = None
        if self.resident is not None and ov.plan and self._ahead is not None and going:
            if batchplan.parse_plan(self._ahead) == ([], [], []):
                slots = [s for s, _ in going]
                why = self.resident.eligible(slots)
                if why is None:
                    self._resident_go = slots
                else:
                    self.counts[f"resident_no_{why}"] += 1

    def _finish(self, slot: int, *, cancelled: bool = False) -> None:
        from .decode import take_snapshot

        self._flush()                           # patches/0370: its tokens before its end
        q = self.seqs[slot]
        self.seqs[slot] = None
        job = q.job
        st = q.stepper
        preempted = cancelled and job.background and not job.cancel and job.out is not None
        if st is not None and not cancelled:
            with self._on(slot) as e:
                pending = st.close()
                if job.draft and st.policy is not None and st.keeps and not q.grid:
                    committed = list(job.prompt) + st.out[:e.st.pos - len(job.prompt)]
                    snap = take_snapshot(e, committed, pending, mtp=st.use_mtp, drafter=st.drafter)
                    self._remember(slot, snap)
                    self._save(slot, [snap])            # patches/0180: the reply snapshot (exact requests)
                elif st.policy is None:         # the reply's rows are not kept
                    self.caches[slot] = [c for c in self.caches[slot] if len(c.ids) <= len(job.prompt)]
        else:
            self.caches[slot] = [c for c in self.caches[slot] if len(c.ids) <= len(job.prompt)]
        # patches/0290: the slot keeps its committed rows (its snapshots' and the store's live pages), not the rest
        self._release(slot, self.states[slot].pos)
        tokens = st.out[:job.max_tokens] if st is not None else []
        sha = hashlib.sha256(json.dumps(tokens).encode()).hexdigest()[:16]
        self.log.append(dict(slot=slot, sha256=sha, tokens=len(tokens), keeps=list(st.keeps) if st else [],
                             arms="".join(st.arms) if st else "", cancelled=cancelled))
        if preempted:                           # rank 0 queued it again; its caller keeps waiting
            job.stats["preempted"] = job.stats.get("preempted", 0) + 1
            return
        if st is not None:
            job.stats.update(decode_s=round(time.perf_counter() - st.start, 4), rounds=st.rounds,
                             batched_rounds=st.shared, min_rows=1 + min(st.depths, default=0),
                             tokens_per_round=round((len(tokens) - 1) / max(st.rounds, 1), 3), sha256=sha,
                             keeps=st.keeps, drafters="".join(st.arms), depths=st.depths,     # 0380: drafts a round
                             round_kinds={k: round(v, 1) for k, v in q.kinds.items()})
            if getattr(st, "adapt", None) is not None:  # patches/0340
                job.stats["adapt"] = dict(serial=st.adapt.serial, probes=st.adapt.probes)
        job.stats.update(cancelled=cancelled)
        if self.store is not None:                  # patches/0180
            job.stats["sessions"] = self.store.describe()
        if job.out is not None:
            job.out.put(None)
