"""Deploy GPU Goblin to a Hugging Face Space under the AMD hackathon org.

Follows the lablab "Build and Deploy an AI App on AMD MI300X as a HuggingFace
Space" tutorial (https://lablab.ai/ai-tutorials/amd-huggingface-deployment-for-ai-hackathons),
but uses ``HfApi.upload_file`` instead of ``git push`` so we can hand-pick
which files go up — keeps the deployed Space lean (no ``bench_cache/``,
no ``.venv``, no ``__pycache__``, no ``brainstorming/`` design docs, etc.).

Prerequisites:
  * `pip install huggingface_hub` (already in the project's runtime deps).
  * `export HF_TOKEN=hf_yourtokenhere`  (write-scoped token).
  * The target Space must already exist under the
    ``lablab-ai-amd-developer-hackathon`` org. Create it once via the HF web
    UI: New Space → Streamlit → CPU basic → Public → name it ``gpu-goblin``.

Usage:
  python scripts/deploy_to_hf_space.py
  python scripts/deploy_to_hf_space.py --space-name gpu-goblin
  python scripts/deploy_to_hf_space.py --owner lablab-ai-amd-developer-hackathon \\
                                       --space-name gpu-goblin \\
                                       --dry-run

Files uploaded (matches the Streamlit Space's runtime needs):
  * README.md                                  — has the Space frontmatter
  * requirements.txt                           — Streamlit-only deps
  * ui/__init__.py + ui/app.py                 — entry point (app_file)
  * agent/                                     — schemas, tools, backends, etc.
  * kb/rocm_rules.yaml + .embeddings_cache_*   — KB + pre-built embeddings
  * runner/__init__.py + runner/protocol.py    — MockRunner for offline replay
  * workloads/synthetic/                       — cached metric scenarios
  * workloads/_runtime.py                      — argparse helper (referenced
                                                  by the train_qwen_lora.py
                                                  sample workload — kept here
                                                  so the Space's "Use sample
                                                  workload" button has a
                                                  real file to point at)
  * workloads/train_qwen_lora.py               — the canonical sample
  * tests/fixtures/cached_audit.json           — offline-replay trajectory

Files deliberately NOT uploaded:
  * bench_cache/, .venv/, .pytest_cache/, __pycache__/   — build artifacts
  * brainstorming/                                       — design docs
  * tests/                                               — except cached_audit.json
  * workloads/scenarios/                                 — alternate workloads
                                                            (they're great for
                                                            the AMD GPU path
                                                            but inflate the Space)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OWNER = "lablab-ai-amd-developer-hackathon"
DEFAULT_SPACE_NAME = "gpu-goblin"


# Files (relative to REPO_ROOT) that get uploaded as-is. Order doesn't matter
# for HF — files appear in repo order — but we group logically for readability.
FILES_TO_UPLOAD: list[str] = [
    # --- Space metadata + entry point ---
    "README.md",
    "Dockerfile",          # HF Space SDK = docker (Streamlit runs inside)
    "requirements.txt",
    "ui/__init__.py",
    "ui/app.py",
    # --- Agent package ---
    "agent/__init__.py",
    "agent/__main__.py",
    "agent/loop.py",
    "agent/prompts.py",
    "agent/schemas.py",
    "agent/server.py",  # not used by Streamlit Space but small + handy
    "agent/backends/__init__.py",
    "agent/backends/base.py",
    "agent/backends/qwen_hf.py",
    "agent/backends/qwen_vllm.py",
    "agent/tools/__init__.py",
    "agent/tools/benchmark.py",
    "agent/tools/compare_runs.py",
    "agent/tools/parse_config.py",
    "agent/tools/profile_run.py",
    "agent/tools/propose_patch.py",
    "agent/tools/query_rocm_kb.py",
    # --- KB ---
    "kb/__init__.py",
    "kb/rocm_rules.yaml",
    # --- Runner (MockRunner is what the Space actually uses) ---
    "runner/__init__.py",
    "runner/protocol.py",
    # --- Synthetic corpus + sample workload + offline-replay fixture ---
    "workloads/_runtime.py",
    "workloads/train_qwen_lora.py",
    "workloads/synthetic/01_baseline_bad/manifest.json",
    "workloads/synthetic/01_baseline_bad/cached_metrics.json",
    "workloads/synthetic/02_optimized/manifest.json",
    "workloads/synthetic/02_optimized/cached_metrics.json",
    "tests/__init__.py",
    "tests/fixtures/cached_audit.json",
]


def _embeddings_cache_files() -> list[str]:
    """Return any kb/.embeddings_cache_<sha>.npy paths to upload."""
    kb = REPO_ROOT / "kb"
    if not kb.exists():
        return []
    return [
        f"kb/{f.name}"
        for f in sorted(kb.iterdir())
        if f.name.startswith(".embeddings_cache_") and f.suffix == ".npy"
    ]


def _check_token() -> str:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
    if not token:
        sys.exit(
            "ERROR: set HF_TOKEN (or HUGGINGFACEHUB_API_TOKEN) to a Hugging Face\n"
            "       token with `write` scope before running this script.\n"
            "       Create one at https://huggingface.co/settings/tokens"
        )
    return token


def _check_files(files: list[str]) -> list[str]:
    missing = [f for f in files if not (REPO_ROOT / f).exists()]
    if missing:
        sys.exit(
            "ERROR: the following files don't exist under the repo root:\n"
            + "\n".join(f"  - {f}" for f in missing)
            + "\n\nDid you run this from outside the project root, or did some\n"
            + "files get deleted? Re-clone if in doubt."
        )
    return files


def main() -> None:
    parser = argparse.ArgumentParser(description="Deploy GPU Goblin to its HF Space.")
    parser.add_argument(
        "--owner",
        default=DEFAULT_OWNER,
        help="HF org/user that owns the Space (default: %(default)s).",
    )
    parser.add_argument(
        "--space-name",
        default=DEFAULT_SPACE_NAME,
        help="Space repo name (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files we'd upload and exit without making any HF API call.",
    )
    args = parser.parse_args()

    repo_id = f"{args.owner}/{args.space_name}"
    files = _check_files(FILES_TO_UPLOAD + _embeddings_cache_files())

    if args.dry_run:
        print(f"DRY RUN — would upload {len(files)} files to spaces/{repo_id}:")
        for f in files:
            size = (REPO_ROOT / f).stat().st_size
            print(f"  {size:>10}  {f}")
        return

    token = _check_token()
    api = HfApi(token=token)

    print(f"Uploading {len(files)} files to spaces/{repo_id} ...")
    for path in files:
        try:
            api.upload_file(
                path_or_fileobj=str(REPO_ROOT / path),
                path_in_repo=path,
                repo_id=repo_id,
                repo_type="space",
            )
            print(f"  ok  {path}")
        except HfHubHTTPError as exc:
            print(f"  FAIL {path}: {exc}", file=sys.stderr)
            sys.exit(1)

    print(
        f"\nDone. Build log at: "
        f"https://huggingface.co/spaces/{repo_id}\n"
        f"(Cold-start ~30-60s; once up, click 'Use sample workload' to see "
        f"the canonical 142 → 318 tok/s replay.)"
    )


if __name__ == "__main__":
    main()
