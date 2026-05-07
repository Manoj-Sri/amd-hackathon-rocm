"""query_rocm_kb tool — semantic search over the curated ROCm rule YAML.

At import time this module:

  1. Loads ``kb/rocm_rules.yaml`` and validates each entry against the
     :class:`Rule` pydantic model. Invalid entries are skipped with a
     ``warnings.warn`` so a single typo never breaks the agent loop.
  2. Embeds every valid rule's ``symptom`` field with
     ``sentence-transformers/all-MiniLM-L6-v2``. Embeddings are cached to
     ``kb/.embeddings_cache_<sha256_of_yaml>.npy`` so subsequent imports
     skip the ~3-second model load + encode pass.
  3. Stashes a normalized embedding matrix for fast cosine similarity.

At query time:

  * Embed the query symptom.
  * Cosine similarity against every rule.
  * Return the top-k rules sorted by score (descending) inside the standard
    :class:`ToolResult` envelope.

Each rule is returned as ``rule.model_dump()`` so the agent loop sees plain
JSON-serializable dicts.
"""

from __future__ import annotations

import hashlib
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from pydantic import ValidationError

from agent.schemas import Rule, ToolResult
from agent.tools import Tool

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_KB_DIR = Path(__file__).resolve().parent.parent.parent / "kb"
_KB_YAML = _KB_DIR / "rocm_rules.yaml"
_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


# ---------------------------------------------------------------------------
# Loading + validation
# ---------------------------------------------------------------------------


def _load_rules(yaml_path: Path) -> tuple[list[Rule], bytes]:
    """Parse the YAML file, validate each entry, return (rules, raw_bytes).

    The raw bytes are returned alongside so the caller can hash them for the
    embeddings cache key without re-reading the file.
    """
    raw = yaml_path.read_bytes()
    data = yaml.safe_load(raw)
    if not isinstance(data, list):
        raise ValueError(
            f"{yaml_path}: top-level must be a list of rule dicts, "
            f"got {type(data).__name__}"
        )

    rules: list[Rule] = []
    for idx, entry in enumerate(data):
        if not isinstance(entry, dict):
            warnings.warn(
                f"{yaml_path.name}: skipping entry #{idx} — not a mapping",
                stacklevel=2,
            )
            continue
        try:
            rules.append(Rule(**entry))
        except ValidationError as exc:
            rule_id = entry.get("id", f"<entry #{idx}>")
            warnings.warn(
                f"{yaml_path.name}: skipping rule {rule_id!r} — validation failed: {exc}",
                stacklevel=2,
            )
    return rules, raw


# ---------------------------------------------------------------------------
# Embedding model — lazy singleton so importing this module is cheap when the
# embeddings cache is warm and we only need to embed at query time.
# ---------------------------------------------------------------------------

_MODEL: Any = None


def _get_model() -> Any:
    """Return the SentenceTransformer instance, loading it on first use."""
    global _MODEL
    if _MODEL is None:
        # Imported lazily so module import doesn't pay for torch/transformers
        # if the embeddings cache already covers the rules.
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(_EMBED_MODEL_NAME)
    return _MODEL


# ---------------------------------------------------------------------------
# Embeddings — load from cache if hash matches, otherwise compute and persist
# ---------------------------------------------------------------------------


def _cache_path(yaml_bytes: bytes) -> Path:
    digest = hashlib.sha256(yaml_bytes).hexdigest()
    return _KB_DIR / f".embeddings_cache_{digest}.npy"


def _embed_rules(rules: list[Rule], yaml_bytes: bytes) -> np.ndarray:
    """Return an (N, D) float32 matrix of L2-normalized rule embeddings.

    Cache layout: a single ``.npy`` file keyed by sha256 of the YAML bytes.
    If the YAML changes by a single byte the hash flips and we recompute;
    otherwise the cached embeddings are reused.
    """
    cache = _cache_path(yaml_bytes)
    if cache.exists():
        try:
            cached = np.load(cache)
            # Defend against a corrupt or shape-mismatched cache file. A row
            # count change implies the rule list changed, even if the YAML
            # hash collided unrealistically — recompute in that case.
            if cached.ndim == 2 and cached.shape[0] == len(rules):
                return cached.astype(np.float32, copy=False)
            warnings.warn(
                f"Embeddings cache shape {cached.shape} does not match "
                f"{len(rules)} rules; recomputing.",
                stacklevel=2,
            )
        except (OSError, ValueError) as exc:
            warnings.warn(
                f"Embeddings cache at {cache} unreadable ({exc}); recomputing.",
                stacklevel=2,
            )

    symptoms = [r.symptom for r in rules]
    try:
        model = _get_model()
    except ImportError as exc:
        # On Hugging Face Spaces we deliberately omit `sentence-transformers`
        # from `requirements.txt` to keep the cold-start fast — the YAML hash
        # of the shipped KB matches a committed cache file. If someone edits
        # `kb/rocm_rules.yaml` without re-running the cache build, we land
        # here. Return a zero-width embedding matrix and let the query path
        # surface a clean ToolResult(ok=False) so the agent loop can adapt
        # rather than crash.
        warnings.warn(
            f"sentence-transformers unavailable ({exc}); KB will return "
            "ok=False until the embeddings cache is rebuilt for the new YAML.",
            stacklevel=2,
        )
        return np.zeros((len(rules), 0), dtype=np.float32)
    embeddings = model.encode(
        symptoms,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32, copy=False)

    # Best-effort cache write. If the kb/ dir is read-only (e.g. in a frozen
    # container layer) we still return correct embeddings, just slower next
    # import.
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache, embeddings)
    except OSError as exc:
        warnings.warn(
            f"Could not persist embeddings cache to {cache}: {exc}",
            stacklevel=2,
        )

    return embeddings


# ---------------------------------------------------------------------------
# Module-level state — populated on import
# ---------------------------------------------------------------------------


def _load_module_state() -> tuple[list[Rule], np.ndarray]:
    """Load rules + embeddings. Tolerates a missing YAML by returning empty
    state so downstream tools can still import this module in test setups."""
    if not _KB_YAML.exists():
        warnings.warn(
            f"KB YAML not found at {_KB_YAML}; query_rocm_kb will return no rules.",
            stacklevel=2,
        )
        return [], np.zeros((0, 0), dtype=np.float32)
    rules, raw = _load_rules(_KB_YAML)
    if not rules:
        return [], np.zeros((0, 0), dtype=np.float32)
    embeddings = _embed_rules(rules, raw)
    return rules, embeddings


_RULES, _RULE_EMBEDDINGS = _load_module_state()


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------


def _keyword_score(symptom: str, rule_text: str) -> float:
    """Cheap fallback when sentence-transformers can't embed the query.

    Tokenise both sides on word characters, lowercase, drop very short tokens,
    return the size of the intersection. Not as good as cosine over MiniLM,
    but more useful than ToolResult(ok=False) when the agent is mid-audit.
    """
    import re

    def _toks(s: str) -> set[str]:
        return {t for t in re.findall(r"[a-zA-Z0-9_]{3,}", s.lower())}

    q_tokens = _toks(symptom)
    if not q_tokens:
        return 0.0
    return float(len(q_tokens & _toks(rule_text)))


def _rule_lite(rule: Rule) -> dict[str, Any]:
    """LLM-facing slim view of a Rule.

    Trims fields the model doesn't need on subsequent tool calls (it only
    forwards `id` to propose_patch, which looks the full Rule up against
    the loaded KB). This shrinks query_rocm_kb's output by ~50% so 5-10
    rules per query don't blow past Qwen2.5-7B's 8K context window.
    """
    return {
        "id": rule.id,
        "symptom": rule.symptom,
        "transform": rule.transform,
        "expected_impact": rule.expected_impact,
        "citation": rule.citation,
    }


def _query_one_keyword(symptom: str, top_k: int) -> list[dict[str, Any]]:
    """Last-resort keyword scoring over rule.symptom + rule.id + rule.category."""
    scored: list[tuple[float, int]] = []
    for i, rule in enumerate(_RULES):
        haystack = f"{rule.symptom} {rule.id} {rule.category} {rule.expected_impact}"
        s = _keyword_score(symptom, haystack)
        if s > 0:
            scored.append((s, i))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [_rule_lite(_RULES[i]) for _, i in scored[:top_k]]


def _query_one(symptom: str, top_k: int) -> tuple[bool, list[dict[str, Any]] | str]:
    """Run one query. Prefer cosine similarity over MiniLM embeddings; fall back
    to keyword scoring if sentence-transformers isn't installed (the deployed
    Hugging Face Space and the rocm/vllm container both omit it for size).
    Returns (ok, rules-or-error).
    """
    try:
        model = _get_model()
        query_vec = model.encode(
            [symptom],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype(np.float32, copy=False)
    except (ImportError, ModuleNotFoundError):
        # sentence-transformers missing → keyword-only fallback. Rule
        # embeddings cache may be loaded (we only need it for cosine,
        # which we're skipping). Still better than returning empty.
        return True, _query_one_keyword(symptom, top_k)
    except Exception as exc:  # pragma: no cover — depends on model state
        return False, f"embedding failed: {type(exc).__name__}: {exc}"

    scores = (_RULE_EMBEDDINGS @ query_vec.T).reshape(-1)
    k = min(top_k, len(_RULES))
    top_idx = np.argsort(scores)[-k:][::-1]
    return True, [_rule_lite(_RULES[i]) for i in top_idx]


def _query_rocm_kb(
    symptom: str | None = None,
    symptoms: list[str] | None = None,
    top_k: int = 5,
) -> ToolResult:
    """Semantic search the KB. Accepts a single ``symptom`` (string) or a
    batch of ``symptoms`` (list of strings).

    Live-AMD-GPU lesson: models naturally batch related queries. We honor
    that by accepting either form. With a list, we run one similarity pass
    per element and return the deduplicated union of top-k hits per query —
    deterministic ordering by best per-query score.
    """
    # Normalize input into a non-empty list of trimmed query strings.
    queries: list[str] = []
    if symptoms:
        if not isinstance(symptoms, list) or not all(isinstance(s, str) for s in symptoms):
            return ToolResult(ok=False, error="symptoms must be a list of strings")
        queries.extend(s.strip() for s in symptoms if s and s.strip())
    if symptom:
        if not isinstance(symptom, str):
            return ToolResult(ok=False, error="symptom must be a string")
        if symptom.strip():
            queries.append(symptom.strip())

    if not queries:
        return ToolResult(
            ok=False,
            error="provide either 'symptom' (string) or 'symptoms' (non-empty list of strings)",
        )
    if top_k < 1:
        return ToolResult(ok=False, error="top_k must be >= 1")

    if not _RULES:
        return ToolResult(
            ok=False,
            error="Rule index is empty — kb/rocm_rules.yaml missing or all entries invalid.",
        )

    if _RULE_EMBEDDINGS.ndim != 2 or _RULE_EMBEDDINGS.shape[1] == 0:
        # Resilient-degradation path — see `_embed_rules`. The shipped cache
        # didn't match the current YAML and `sentence-transformers` isn't
        # installed in this environment, so we have rules but no usable
        # embedding index.
        return ToolResult(
            ok=False,
            error=(
                "KB embeddings unavailable — the committed cache doesn't match "
                "the current kb/rocm_rules.yaml and sentence-transformers is "
                "not installed. Rebuild the cache locally and recommit, or "
                "install the dev extras."
            ),
        )

    seen_ids: set[str] = set()
    aggregated: list[dict[str, Any]] = []
    for q in queries:
        ok, payload = _query_one(q, top_k)
        if not ok:
            return ToolResult(ok=False, error=payload)  # type: ignore[arg-type]
        for rule in payload:  # type: ignore[union-attr]
            rid = rule["id"]
            if rid not in seen_ids:
                seen_ids.add(rid)
                aggregated.append(rule)

    return ToolResult(ok=True, result={"rules": aggregated})


QUERY_ROCM_KB = Tool(
    name="query_rocm_kb",
    description=(
        "Search the curated ROCm/MI300X optimization knowledge base by natural-"
        "language symptom. Pass a single ``symptom`` string OR batch related "
        "queries via ``symptoms`` (list of strings). Returns the deduplicated "
        "union of top_k Rules per query, with citations. Use this after "
        "profile_run to find rules matching the observed waste pattern."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "symptom": {
                "type": "string",
                "description": (
                    "Single natural-language description of the observed "
                    "problem. Provide either this OR `symptoms`."
                ),
            },
            "symptoms": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Multiple natural-language descriptions to query in one "
                    "call. Returns the deduplicated union of top_k hits per "
                    "query. Use when several distinct concerns came out of "
                    "profile_run (e.g. precision + attention + dataloader)."
                ),
            },
            "top_k": {
                "type": "integer",
                "default": 5,
                "minimum": 1,
                "maximum": 20,
                "description": "Maximum number of rules to return per query.",
            },
        },
    },
    fn=_query_rocm_kb,
)
