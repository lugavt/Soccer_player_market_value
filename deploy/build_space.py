"""Assemble the Hugging Face Space and, optionally, publish it as a PRIVATE Space.

    python deploy/build_space.py                         # build build/hf_space/ only
    python deploy/build_space.py --push                  # build + upload to <you>/transfer-edge
    python deploy/build_space.py --push --repo org/name  # e.g. an organisation you share with

Needs `hf auth login` first (a token with write access). The Space is created
private, and forced back to private if it already existed as public: the data
behind it is not licensed for redistribution.

Test the bundle locally before pushing:

    docker build -t transfer-edge-space build/hf_space
    docker run --rm -p 7860:8501 transfer-edge-space      # -> http://localhost:7860
"""

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
OUT = ROOT / "build" / "hf_space"

# exactly what the app reads at runtime (src/data/loader.py, src/models/*)
DATA_FILES = ["panel_final.parquet", "fbref_matched.parquet", "test_predictions.parquet"]
MODEL_FILES = ["xgb_model.pkl", "xgb_intervals.pkl"]
CODE_DIRS = ["app", "src"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", ".ipynb_checkpoints")


def build() -> Path:
    missing = [f"data/{f}" for f in DATA_FILES if not (ROOT / "data" / f).exists()]
    missing += [f"models/{f}" for f in MODEL_FILES if not (ROOT / "models" / f).exists()]
    if missing:
        raise SystemExit(f"missing artifacts — run the pipeline and train first: {missing}")

    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    for name in ["Dockerfile", "requirements.txt", "README.md"]:
        shutil.copy2(DEPLOY / name, OUT / name)
    for d in CODE_DIRS:
        shutil.copytree(ROOT / d, OUT / d, ignore=IGNORE)
    for sub, files in [("data", DATA_FILES), ("models", MODEL_FILES)]:
        (OUT / sub).mkdir()
        for f in files:
            shutil.copy2(ROOT / sub / f, OUT / sub / f)

    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"built {OUT.relative_to(ROOT)}  ({size / 1e6:.1f} MB)")
    return OUT


def push(folder: Path, repo: str | None) -> None:
    from huggingface_hub import HfApi

    api = HfApi()
    user = api.whoami()["name"]            # fails clearly if not logged in
    repo_id = repo or f"{user}/transfer-edge"

    api.create_repo(repo_id, repo_type="space", space_sdk="docker", private=True, exist_ok=True)
    api.update_repo_settings(repo_id, repo_type="space", private=True)   # never leave it public
    api.upload_folder(
        folder_path=str(folder), repo_id=repo_id, repo_type="space",
        commit_message="Deploy from Transfer Edge repository",
        delete_patterns=["app/**", "src/**", "data/**", "models/**"],   # drop files removed locally
    )
    print(f"pushed -> https://huggingface.co/spaces/{repo_id}  (private; first build takes a few minutes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--push", action="store_true", help="upload to Hugging Face as a private Space")
    parser.add_argument("--repo", help="Space id, default <your username>/transfer-edge")
    args = parser.parse_args()
    folder = build()
    if args.push:
        push(folder, args.repo)
