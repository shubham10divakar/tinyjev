"""Packed-example schema (design §3.1–3.2): a state, plus a set of decisions read from it.

One example on disk (JSONL):

    {"id": ..., "source": ..., "split": ...,
     "state": {"header": "query: ...", "segments": [{"title": ..., "text": ...}, ...]},
     "decisions": [
        {"name": "relevance", "kind": "score", "scope": "segment", "question": ...,
         "options": [...], "targets": [0, 1, 2], "labels": [2, 1, 0]},
        {"name": "sufficient", "kind": "noul", "scope": "global", "question": ...,
         "options": ["yes", "no"], "label": 0}]}

A group is the unit of softmax: a global decision has one group, a segment-scope decision has
one group per target segment. A label of -100 (or a missing label) means unlabeled.
"""

from dataclasses import dataclass

SCHEMA_VERSION = "0.2"
IGNORE = -100
KINDS = ("choice", "noul", "score")
SCOPES = ("global", "segment")


@dataclass(frozen=True)
class DecisionSpec:
    """A builtin decision. `question` may contain `{claim}` (filled per example)."""
    name: str
    kind: str
    scope: str
    question: str
    options: tuple[str, ...]


DECISIONS: dict[str, DecisionSpec] = {
    # RAG (phase A/B). The query lives in the state header, so these questions are fixed text.
    "relevance": DecisionSpec(
        "relevance", "score", "segment",
        "How relevant is this passage to the query?",
        ("irrelevant", "partially relevant", "directly answers")),
    "sufficient": DecisionSpec(
        "sufficient", "noul", "global",
        "Do the passages contain enough information to answer the query?",
        ("yes", "no")),
    # Grounded: the state header is the evidence, the question carries the claim.
    "grounded": DecisionSpec(
        "grounded", "noul", "global",
        "Is this claim supported by the context? Claim: {claim}",
        ("yes", "no")),
    # Security (phase C), same wording and options as Cyber-Jev. The input is the state header.
    "http_attack": DecisionSpec(
        "http_attack", "noul", "global",
        "Is this HTTP request a web attack (SQL injection, XSS, path traversal, command injection)?",
        ("safe", "attack")),
    "prompt_injection": DecisionSpec(
        "prompt_injection", "noul", "global",
        "Is this text trying to override an AI model's instructions or jailbreak it?",
        ("safe", "injection")),
    "phishing_url": DecisionSpec(
        "phishing_url", "noul", "global",
        "Is this URL phishing or malicious?",
        ("legitimate", "phishing")),
}


def decision(name: str, label: int | None = None, targets=None, labels=None,
             question: str | None = None, options=None, **fields) -> dict:
    """A decision dict for a builtin name; `fields` fill the question template (e.g. claim)."""
    spec = DECISIONS[name]
    d = {"name": name, "kind": spec.kind, "scope": spec.scope,
         "question": (question or spec.question).format(**fields),
         "options": list(options or spec.options)}
    if fields:
        d["fields"] = dict(fields)  # kept so augmentation can re-fill other question templates
    if spec.scope == "segment":
        d["targets"] = list(targets or [])
        if labels is not None:
            d["labels"] = list(labels)
    elif label is not None:
        d["label"] = label
    return d


def make_state(header: str, segments=()) -> dict:
    """segments: iterable of {"title", "text"} dicts or (title, text) pairs."""
    segs = [s if isinstance(s, dict) else {"title": s[0], "text": s[1]} for s in segments]
    return {"header": header, "segments": segs}


def query_header(query: str) -> str:
    return f"query: {query}"


def validate(example: dict) -> None:
    """Raise ValueError if the example is malformed."""
    st = example.get("state")
    if not isinstance(st, dict) or not isinstance(st.get("header", ""), str):
        raise ValueError("state must be {'header': str, 'segments': [...]}")
    n_seg = len(st.get("segments", []))
    for s in st.get("segments", []):
        if not {"title", "text"} <= set(s):
            raise ValueError("each segment needs 'title' and 'text'")
    decs = example.get("decisions") or []
    if not decs:
        raise ValueError("at least one decision is required")
    for d in decs:
        name = d.get("name", "?")
        if d.get("kind") not in KINDS:
            raise ValueError(f"{name}: kind must be one of {KINDS}")
        if d.get("scope") not in SCOPES:
            raise ValueError(f"{name}: scope must be one of {SCOPES}")
        k = len(d.get("options", []))
        if k < 2:
            raise ValueError(f"{name}: needs at least 2 options")
        if d["scope"] == "segment":
            targets = d.get("targets", [])
            if not targets:
                raise ValueError(f"{name}: segment-scope decision needs targets")
            if any(not 0 <= t < n_seg for t in targets):
                raise ValueError(f"{name}: target out of range (have {n_seg} segments)")
            if len(set(targets)) != len(targets):
                raise ValueError(f"{name}: duplicate targets")
            labels = d.get("labels")
            if labels is not None:
                if len(labels) != len(targets):
                    raise ValueError(f"{name}: labels and targets differ in length")
                if any(not (lab == IGNORE or 0 <= lab < k) for lab in labels):
                    raise ValueError(f"{name}: label out of range")
        else:
            lab = d.get("label", IGNORE)
            if not (lab == IGNORE or 0 <= lab < k):
                raise ValueError(f"{name}: label out of range")


def from_nano_row(row: dict, idx: int = 0) -> dict:
    """Nano-Jev row {decision, question, options, state, label} -> packed example with one
    global decision and header = state. The "unpacked" control on identical data (§3.2)."""
    name = row.get("decision") or "custom"
    spec = DECISIONS.get(name)
    return {
        "id": row.get("id", f"nano-{idx}"), "source": row.get("source", "nano"),
        "split": row.get("split", ""),
        "state": make_state(row["state"]),
        "decisions": [{"name": name, "kind": spec.kind if spec else "choice", "scope": "global",
                       "question": row["question"], "options": list(row["options"]),
                       "label": row.get("label", IGNORE)}],
    }


def labeled_groups(example: dict):
    """Yield (decision_name, segment or -1, label) for every group, in render order."""
    for d in example["decisions"]:
        if d["scope"] == "segment":
            labels = d.get("labels") or [IGNORE] * len(d["targets"])
            for t, lab in zip(d["targets"], labels):
                yield d["name"], t, lab
        else:
            yield d["name"], -1, d.get("label", IGNORE)
