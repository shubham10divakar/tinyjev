"""Where Tiny-Jev weights come from: a local folder, a version ("v0.1"), or a Hub repo id
("user/tiny-jev", optionally "@revision"). Hub repos are downloaded whole (adapter, head,
markers, calibration files); the Qwen3 base is fetched by transformers on first use.
"""

from pathlib import Path

DEFAULT_REPO = "sdmlai/tiny-jev"
VERSIONS = {"v0.1": "v0.1"}       # version -> Hub tag (nothing released yet)
DEFAULT_VERSION = "v0.1"


def resolve(target: str | None = None) -> Path:
    if target and Path(target).is_dir():
        return Path(target)
    target = target or DEFAULT_VERSION
    if target in VERSIONS:
        repo, rev = DEFAULT_REPO, VERSIONS[target]
    else:
        repo, _, rev = target.partition("@")
        rev = rev or None
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(repo_id=repo, revision=rev))
