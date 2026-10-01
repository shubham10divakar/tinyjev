"""Format-robustness family (design doc 15 §5.2, 3%): rule-generated decisions about the
state's own form ("Does the text contain a URL / a date / code?").

It teaches that the *question* decides the answer, not the data source: the same text gets
different answers for different features. Labels come from regex detectors run on the final
text, so they are correct even when a base sentence already contains the feature.
"""

import random
import re

from .tasks import canonical

# feature -> (phrasings for {feature}, detector, generator of an instance)
_WORDS = ("alpha beta river stone cloud paper music garden window yellow quick silver "
          "morning train letter market forest bridge orange table winter").split()


def _url(r):
    return f"https://www.{r.choice(_WORDS)}{r.choice(_WORDS)}.{r.choice(['com', 'org', 'net', 'io'])}/{r.choice(_WORDS)}"


def _email(r):
    return f"{r.choice(_WORDS)}.{r.choice(_WORDS)}@{r.choice(_WORDS)}mail.com"


def _date(r):
    months = ["January", "March", "June", "September", "November"]
    return r.choice([f"{r.randint(1, 28)} {r.choice(months)} {r.randint(1950, 2030)}",
                     f"{r.randint(1990, 2030)}-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}",
                     f"{r.choice(months)} {r.randint(1, 28)}, {r.randint(1950, 2030)}"])


def _code(r):
    v = r.choice(_WORDS)
    return r.choice([f"def {v}(x): return x + 1", f"for (int i = 0; i < n; i++) {{ {v}[i] = 0; }}",
                     f"SELECT * FROM {v} WHERE id = 3;", f"const {v} = () => {{ return null; }};"])


def _phone(r):
    return r.choice([f"+1 {r.randint(200, 999)}-{r.randint(200, 999)}-{r.randint(1000, 9999)}",
                     f"({r.randint(200, 999)}) {r.randint(200, 999)}-{r.randint(1000, 9999)}"])


def _money(r):
    return r.choice([f"${r.randint(1, 999)}.{r.randint(0, 99):02d}", f"€{r.randint(1, 5000)}",
                     f"{r.randint(1, 900)} dollars"])


def _hashtag(r):
    return f"#{r.choice(_WORDS)}{r.choice(_WORDS).capitalize()}"


FEATURES = {
    "url": (["a URL", "a web link", "a link to a website"],
            re.compile(r"https?://|www\.\w", re.I), _url),
    "email": (["an email address", "an e-mail address"],
              re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), _email),
    "date": (["a date", "a calendar date"],
             re.compile(r"\b(1[5-9]|20)\d\d-\d\d-\d\d\b|\b\d{1,2} (January|February|March|April|May|June|"
                        r"July|August|September|October|November|December) \d{4}\b|\b(January|February|"
                        r"March|April|May|June|July|August|September|October|November|December) \d{1,2}, "
                        r"\d{4}\b"), _date),
    "code": (["source code", "a code snippet", "programming code"],
             re.compile(r"\bdef \w+\(|\bfor \(int |\bSELECT \* FROM\b|\bconst \w+ = \(\)"), _code),
    "phone": (["a phone number", "a telephone number"],
              re.compile(r"\+1 \d{3}-\d{3}-\d{4}|\(\d{3}\) \d{3}-\d{4}"), _phone),
    "money": (["an amount of money", "a price"],
              re.compile(r"[$€]\d|\b\d+ dollars\b"), _money),
    "hashtag": (["a hashtag"], re.compile(r"(?<![\w&])#[A-Za-z]\w+"), _hashtag),
}

FALLBACK_TEXTS = [
    "The committee met on Tuesday to review the budget for the new library.",
    "She said the train was late again, so the meeting started without her.",
    "Our team shipped the update after fixing two bugs in the login page.",
    "The museum is open every day except Monday, and entry is free for children.",
    "Please send the signed form back to the office before the end of the week.",
    "Heavy rain is expected in the north, with clearer skies later in the evening.",
    "The recipe needs flour, eggs, milk and a pinch of salt.",
    "He moved to the coast to open a small bakery with his brother.",
]


def has(feature: str, text: str) -> bool:
    return bool(FEATURES[feature][1].search(text))


def format_pack(i: int, base: str, rng: random.Random, n_decisions: int = 2) -> dict:
    """One state, several format decisions about it (different features, mixed answers)."""
    text = base
    for f in rng.sample(list(FEATURES), rng.randint(0, 2)):        # insert 0–2 features
        words = text.split()
        pos = rng.randint(0, len(words))
        text = " ".join(words[:pos] + [FEATURES[f][2](rng)] + words[pos:])
    q, opts = canonical("format_check")
    asked = rng.sample(list(FEATURES), min(n_decisions, len(FEATURES)))
    decs = []
    for f in asked:
        phr = rng.choice(FEATURES[f][0])
        decs.append({"name": "format_check", "kind": "noul", "scope": "global",
                     "question": q.format(feature=phr), "options": list(opts),
                     "label": 0 if has(f, text) else 1, "fields": {"feature": phr},
                     "feature": f})
    return {"id": f"format-{i}", "source": "format", "split": "train",
            "state": {"header": text, "segments": []}, "decisions": decs,
            "meta": {"family": "format"}}


def format_packs(n: int, seed: int = 0, texts: list[str] | None = None) -> list[dict]:
    rng = random.Random(f"format/{seed}")
    pool = [t for t in (texts or FALLBACK_TEXTS) if t and len(t) < 600]
    return [format_pack(i, rng.choice(pool), rng, rng.randint(1, 3)) for i in range(n)]
