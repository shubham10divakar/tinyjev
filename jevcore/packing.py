"""Render packed examples into token rows, and build the attention masks (design §3.3–3.8).

Rendered sequence (Micro-Jev, bidirectional):

    [CLS] <state> {header} <seg> [1] {title}: {text} <seg> [2] ... [SEP]
    <dec> <q> {question} <opt> {option 1} <opt> {option 2} ... [<ref> x |targets|]
    <dec> <q> {question} <opt> ...

Per-token bookkeeping (§3.5):
    ex     example index inside the row (several short examples can share a row)
    blk    0 for state tokens, d >= 1 for decision block d (unique within the row)
    prt    state: 0 = header/CLS/SEP/<state>, i+1 = segment i
           decision: 0 = question span, j+1 = option j, REF_BASE+i = the <ref> for segment i
    pos    state 0..S-1; each decision block restarts at S (§3.6)
    valid  0 for padding

With the block mask and the position restart, a decision's outputs are identical to running
it alone (invariance, §6.4).

Tiny-Jev (causal=True, design doc 15 §3): no [CLS] / [SEP], an optional sink token first, and
end markers read out what the causal model has seen:

    [sink] <state> {header} <seg> [1] ... <dec> <q> {question} <qe> <opt> {option} <oe> ...

The anchor is <qe> (or <ref>), option j is read at its <oe>. `render_state` and
`append_decisions` split a row into the cached prefix and the per-call decision part.
"""

from dataclasses import dataclass, field

import torch

from .schema import IGNORE

REF_BASE = 1_000_000

# Marker tokens and the words whose mean embedding initialises each one (§3.3).
MARKERS = {
    "state": ("<state>", ["context"]),
    "seg": ("<seg>", ["passage"]),
    "dec": ("<dec>", ["decide"]),
    "q": ("<q>", ["question"]),
    "opt": ("<opt>", ["option"]),
    "ref": ("<ref>", ["passage", "relevant"]),
    "qe": ("<qe>", ["answer"]),        # Tiny-Jev only (causal readout)
    "oe": ("<oe>", ["option"]),        # Tiny-Jev only
}
MICRO_MARKERS = ("state", "seg", "dec", "q", "opt", "ref")
HEADER_MAX = 256       # §3.7 step 1
MIN_SEG_TOKENS = 32    # §3.7 step 2


class PackOverflow(ValueError):
    """The decisions plus a minimally truncated state don't fit in max_len."""


@dataclass
class PackConfig:
    """Rendering / masking switches. Defaults are the main model; others are ablations (§7)."""
    max_len: int = 2048
    isolated: bool = True            # False: options see sibling options (A2)
    position_restart: bool = True    # False: plain sequential positions (A7; A1 with mask="full")
    mask: str = "block"              # "block" | "full" (A1: no isolation mask)
    ref_view: str = "own"            # "own" | "all" (A6)
    decision_global: bool = True     # False: decision tokens keep the +-64 window (A3)
    half_window: int = 64            # ModernBERT local_attention // 2
    causal: bool = False             # Tiny-Jev
    sink_id: int | None = None       # Tiny-Jev: BOS-like attention-sink token before <state> (§3)
    echo: bool = False               # Tiny-Jev T-A7: segments rendered twice (causal asymmetry fix)
    drop_untargeted: bool = True     # §3.7 step 3a

    @classmethod
    def from_model_cfg(cls, model_cfg: dict, max_len: int) -> "PackConfig":
        return cls(max_len=max_len,
                   isolated=model_cfg.get("option_mode", "isolated") == "isolated",
                   position_restart=model_cfg.get("position_restart", True),
                   mask=model_cfg.get("mask", "block"),
                   ref_view=model_cfg.get("ref_view", "own"),
                   decision_global=model_cfg.get("decision_global", True))


@dataclass
class Group:
    """One softmax group. Token indices are relative to the start of its row."""
    name: str
    anchor: int                  # <q> (global decision) or <ref> (segment-scope)
    options: list[int]           # <opt> token of each option
    spans: list[tuple[int, int]] # [start, end) of each option span (for mean readout, A4)
    label: int = IGNORE
    seg: int = -1                # original segment index for segment-scope decisions
    ex: int = 0                  # example index inside the row
    dec: int = 0                 # decision index inside its example


@dataclass
class Rendered:
    """One example rendered on its own; positions start at 0."""
    ids: list[int] = field(default_factory=list)
    pos: list[int] = field(default_factory=list)
    blk: list[int] = field(default_factory=list)
    prt: list[int] = field(default_factory=list)
    groups: list[Group] = field(default_factory=list)
    state_len: int = 0
    kept_segments: list[int] = field(default_factory=list)  # original indices, in pack order
    truncated: bool = False

    def __len__(self):
        return len(self.ids)

    def push(self, ids, blk: int, prt: int, pos_start: int | None = None) -> int:
        """Append tokens; return the index of the first. Positions continue sequentially,
        from pos_start when given."""
        i0 = len(self.ids)
        p0 = pos_start if pos_start is not None else (self.pos[-1] + 1 if self.pos else 0)
        self.ids += ids
        self.pos += range(p0, p0 + len(ids))
        self.blk += [blk] * len(ids)
        self.prt += [prt] * len(ids)
        return i0


# ------------------------------------------------------------------------------ tokenizer

def add_markers(tok, names=MICRO_MARKERS) -> dict[str, int]:
    """Add the marker tokens to the tokenizer; return {marker name: token id}."""
    toks = [MARKERS[n][0] for n in names]
    tok.add_special_tokens({"additional_special_tokens": toks})
    return {n: tok.convert_tokens_to_ids(MARKERS[n][0]) for n in names}


def marker_ids(tok, names=MICRO_MARKERS) -> dict[str, int]:
    """Marker ids from a tokenizer that already has them (e.g. a saved checkpoint)."""
    ids = {n: tok.convert_tokens_to_ids(MARKERS[n][0]) for n in names}
    missing = [n for n, i in ids.items() if i is None or i == tok.unk_token_id]
    if missing:
        raise ValueError(f"tokenizer lacks markers {missing}; call add_markers first")
    return ids


# ------------------------------------------------------------------------------ truncation

def _head_tail(ids: list[int], n: int) -> list[int]:
    if len(ids) <= n:
        return ids
    return ids[: n - n // 2] + ids[len(ids) - n // 2:]


def _cap_for(lengths: list[int], budget: int) -> int | None:
    """Largest per-segment cap c >= MIN_SEG_TOKENS with sum(min(l, c)) <= budget, or None."""
    if sum(lengths) <= budget:
        return max(lengths, default=0)
    lo, hi = MIN_SEG_TOKENS, max(lengths)
    if sum(min(n, lo) for n in lengths) > budget:
        return None
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(min(n, mid) for n in lengths) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def truncate_state(header_ids: list[int], seg_ids: list[list[int]], targeted: set[int],
                   budget: int, drop_untargeted: bool = True):
    """Fit the state into `budget` tokens (§3.7). Decisions are never truncated.

    seg_ids[i] already includes its leading <seg> marker. Returns
    (header_ids, [(original index, ids)], truncated). Raises PackOverflow if even the minimal
    state doesn't fit (the caller then splits the segments into several passes).
    """
    header = _head_tail(header_ids, HEADER_MAX)
    truncated = len(header) < len(header_ids)
    keep = list(range(len(seg_ids)))
    room = budget - len(header)
    cap = _cap_for([len(seg_ids[i]) for i in keep], room)
    if cap is None and drop_untargeted:
        keep = [i for i in keep if i in targeted]
        truncated = True
        cap = _cap_for([len(seg_ids[i]) for i in keep], room)
    if cap is None or room < 0:
        raise PackOverflow(f"state needs > {budget} tokens even at {MIN_SEG_TOKENS} per segment")
    segs = [(i, seg_ids[i][:cap]) for i in keep]
    truncated |= any(len(s) < len(seg_ids[i]) for i, s in segs)
    return header, segs, truncated


# ------------------------------------------------------------------------------ rendering

def _encoder(tok):
    return lambda s: tok(s, add_special_tokens=False)["input_ids"]


def segment_text(i: int, seg: dict) -> str:
    title = seg.get("title") or ""
    return f"[{i + 1}] {title}: {seg['text']}" if title else f"[{i + 1}] {seg['text']}"


def decision_blocks(decisions: list[dict], enc, M: dict[str, int], causal: bool):
    """Token ids of each decision block: [(question ids, [option ids], n_ref)], and the total."""
    blocks = []
    for dec in decisions:
        q_ids = [M["dec"], M["q"]] + enc(dec["question"]) + ([M["qe"]] if causal else [])
        o_ids = [[M["opt"]] + enc(o) + ([M["oe"]] if causal else []) for o in dec["options"]]
        n_ref = len(dec.get("targets", [])) if dec["scope"] == "segment" else 0
        blocks.append((q_ids, o_ids, n_ref))
    return blocks, sum(len(q) + sum(map(len, o)) + n for q, o, n in blocks)


def state_overhead(cfg: PackConfig) -> int:
    """Tokens of the state that aren't header or segments."""
    if cfg.causal:                                    # [sink] <state> ...
        return 1 + (cfg.sink_id is not None)
    return 3                                          # [CLS] <state> ... [SEP]


def segment_ids(i: int, seg: dict, enc, M: dict[str, int]) -> list[int]:
    return [M["seg"]] + enc(segment_text(i, seg))


def render_state(state: dict, tok, M: dict[str, int], cfg: PackConfig, budget: int,
                 targeted: set[int] = frozenset(), enc=None) -> Rendered:
    """The state part only (blk 0, positions from 0), truncated to `budget` header + segment
    tokens (§3.7). Tiny-Jev's Session prefills exactly these tokens into the cache."""
    enc = enc or _encoder(tok)
    causal = cfg.causal
    header_ids = enc(state.get("header", ""))
    seg_ids = [segment_ids(i, s, enc, M) for i, s in enumerate(state.get("segments", []))]
    header, segs, truncated = truncate_state(header_ids, seg_ids, targeted, budget,
                                             cfg.drop_untargeted)
    if cfg.echo:      # segments count twice: shrink their budget until header + 2x fits
        b = budget
        while len(header) + 2 * sum(len(x) for _, x in segs) > budget:
            b -= len(header) + 2 * sum(len(x) for _, x in segs) - budget
            header, segs, truncated = truncate_state(header_ids, seg_ids, targeted, b,
                                                     cfg.drop_untargeted)
            truncated = True
    # Re-number kept segments so the "[i]" labels match the pack order.
    if len(segs) < len(seg_ids):
        segs = [(i, segment_ids(k, state["segments"][i], enc, M)[: len(ids)])
                for k, (i, ids) in enumerate(segs)]

    r = Rendered(truncated=truncated)
    if causal:
        first = ([cfg.sink_id] if cfg.sink_id is not None else []) + [M["state"]]
    else:
        first = [tok.cls_token_id, M["state"]]
    r.push(first + header, blk=0, prt=0)
    for i, ids in segs:
        r.push(ids, blk=0, prt=i + 1)
        r.kept_segments.append(i)
    if cfg.echo:      # second copy: each segment now also sees every other segment (causal)
        for i, ids in segs:
            r.push(ids, blk=0, prt=i + 1)
    if not causal:
        r.push([tok.sep_token_id], blk=0, prt=0)
    r.state_len = len(r)
    return r


def render(example: dict, tok, M: dict[str, int], cfg: PackConfig, enc=None) -> Rendered:
    """Render one packed example (§3.9). Raises PackOverflow if it can't fit in cfg.max_len."""
    enc = enc or _encoder(tok)
    decisions = example["decisions"]
    # Decision blocks first: their size sets the state's budget.
    blocks, dec_tokens = decision_blocks(decisions, enc, M, cfg.causal)
    budget = cfg.max_len - dec_tokens - state_overhead(cfg)
    targeted = {t for d in decisions if d["scope"] == "segment" for t in d["targets"]}
    r = render_state(example["state"], tok, M, cfg, budget, targeted, enc)
    append_decisions(r, decisions, blocks, M, cfg)
    if len(r) > cfg.max_len:      # only possible if the decisions alone exceed max_len
        raise PackOverflow(f"decision blocks alone need {dec_tokens} tokens > {cfg.max_len}")
    return r


def append_decisions(r: Rendered, decisions: list[dict], blocks, M: dict[str, int],
                     cfg: PackConfig) -> Rendered:
    """Append decision blocks after the state in `r` (positions restart at r.state_len).
    Segment targets refer to the state's original segment indices."""
    causal = cfg.causal
    S = r.state_len
    restart = cfg.position_restart

    for d, (dec, (q_ids, o_ids, _)) in enumerate(zip(decisions, blocks)):
        blk = d + 1
        q0 = r.push(q_ids, blk=blk, prt=0, pos_start=S if restart else None)
        Lq = len(q_ids)
        anchor = q0 + 1 if not causal else q0 + Lq - 1          # <q> (Micro) or <qe> (Tiny)
        opt_tok, spans, nxt = [], [], S + Lq
        for j, ids in enumerate(o_ids):
            start = (S + Lq if cfg.isolated else nxt) if restart else None
            o0 = r.push(ids, blk=blk, prt=j + 1, pos_start=start)
            opt_tok.append(o0 if not causal else o0 + len(ids) - 1)   # <opt> or <oe>
            spans.append((o0, o0 + len(ids)))
            nxt = r.pos[-1] + 1
        if dec["scope"] == "global":
            r.groups.append(Group(dec["name"], anchor, opt_tok, spans,
                                  dec.get("label", IGNORE), seg=-1, dec=d))
        else:
            labels = dec.get("labels") or [IGNORE] * len(dec["targets"])
            for t, lab in zip(dec["targets"], labels):
                ref = r.push([M["ref"]], blk=blk, prt=REF_BASE + t,
                             pos_start=S + Lq if restart else None)
                r.groups.append(Group(dec["name"], ref, opt_tok, spans, lab, seg=t, dec=d))
    return r


@dataclass
class Row:
    """Several rendered examples concatenated into one row of the batch."""
    ids: list[int] = field(default_factory=list)
    pos: list[int] = field(default_factory=list)
    ex: list[int] = field(default_factory=list)
    blk: list[int] = field(default_factory=list)
    prt: list[int] = field(default_factory=list)
    groups: list[Group] = field(default_factory=list)
    examples: list[int] = field(default_factory=list)   # caller's ids of the packed examples

    def __len__(self):
        return len(self.ids)


def pack_row(rendered: list[Rendered], example_ids=None) -> Row:
    """Concatenate rendered examples; block ids and group indices are offset to stay unique."""
    row = Row()
    blk_off = 0
    for e, r in enumerate(rendered):
        off = len(row.ids)
        row.ids += r.ids
        row.pos += r.pos
        row.ex += [e] * len(r)
        row.blk += [b + blk_off if b else 0 for b in r.blk]
        row.prt += r.prt
        for g in r.groups:
            row.groups.append(Group(g.name, g.anchor + off, [o + off for o in g.options],
                                    [(a + off, b + off) for a, b in g.spans],
                                    g.label, g.seg, e, g.dec))
        blk_off += max(r.blk, default=0)
        row.examples.append(example_ids[e] if example_ids is not None else e)
    return row


# ------------------------------------------------------------------------------ masks

def allowed(ex, blk, prt, valid, isolated=True, causal=False, ref_view="own", q_start=0):
    """Bool [T - q_start, T] for one row: True where query token q may attend key token k (§3.6).

    Rows are the query tokens q_start..T-1; the default gives the full [T, T] mask. Tiny-Jev's
    cache path asks only for the new tokens' rows (`allowed_with_state`, §4.4)."""
    Q = lambda t: t[q_start:, None]  # noqa: E731
    K = lambda t: t[None, :]  # noqa: E731
    T = blk.shape[0]
    idx = torch.arange(T, device=blk.device)
    same_ex = Q(ex) == K(ex)
    q_state, k_state = Q(blk) == 0, K(blk) == 0
    q_ref = Q(prt) >= REF_BASE
    ref_seg = Q(prt) - REF_BASE + 1                       # state segment i has prt = i + 1
    if ref_view == "own":
        see_state = k_state & (q_state | ~q_ref | (K(prt) == 0) | (K(prt) == ref_seg))
    else:                                                 # A6: <ref> sees the whole state
        see_state = k_state.expand(T - q_start, T)
    same_dec = ~q_state & (Q(blk) == K(blk))
    k_is_opt = (K(prt) > 0) & (K(prt) < REF_BASE)
    if isolated:
        k_opt = k_is_opt & (K(prt) == Q(prt))
    else:                                                 # A2: option tokens see sibling options
        k_opt = k_is_opt & (Q(prt) > 0) & (Q(prt) < REF_BASE)
    k_self_ref = (K(prt) >= REF_BASE) & (K(prt) == Q(prt))
    see_dec = same_dec & ((K(prt) == 0) | k_opt | k_self_ref)
    v = valid.bool()
    m = same_ex & (see_state | see_dec) & Q(v) & K(v)
    if causal:
        m &= K(idx) <= Q(idx)
    return m | (K(idx) == Q(idx))


def allowed_full(ex, valid):
    """A1: no isolation, only example and padding boundaries."""
    v = valid.bool()
    m = (ex[:, None] == ex[None, :]) & v[:, None] & v[None, :]
    return m | torch.eye(ex.shape[0], dtype=torch.bool, device=ex.device)


def local_from_global(m, blk, half_window=64, decision_global=True):
    """ModernBERT sliding layers: state tokens keep their +-half_window window; decision tokens
    stay global (§4.3) unless decision_global=False (A3)."""
    idx = torch.arange(blk.shape[0], device=blk.device)
    near = (idx[:, None] - idx[None, :]).abs() <= half_window
    if decision_global:
        near = near | (blk[:, None] != 0)
    return m & near | torch.eye(blk.shape[0], dtype=torch.bool, device=blk.device)


def row_masks(ex, blk, prt, valid, cfg: PackConfig):
    """(full, local) bool [T, T] masks for one row."""
    if cfg.mask == "full":
        full = allowed_full(ex, valid)
        if cfg.causal:
            full = torch.tril(full)
    else:
        full = allowed(ex, blk, prt, valid, cfg.isolated, cfg.causal, cfg.ref_view)
    return full, local_from_global(full, blk, cfg.half_window, cfg.decision_global)
