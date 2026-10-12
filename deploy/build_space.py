"""Assemble the app-only deploy bundle for one of two hosts, and publish it.

Streamlit Community Cloud (free; deployed from a PRIVATE GitHub repo — the app itself may be public):

    python deploy/build_space.py --target streamlit                      # build build/streamlit_app/
    python deploy/build_space.py --target streamlit \
        --git-remote git@github.com:<you>/transfer-edge-app.git          # build + commit + push

  build/streamlit_app/ is its own git repository (kept across rebuilds), so every
  redeploy is one more commit; Community Cloud redeploys on push. Never point
  --git-remote at a public repository: the data is not licensed for redistribution.

Hugging Face Space (Docker Spaces need a PRO subscription since 2026):

    python deploy/build_space.py                         # build build/hf_space/ only
    python deploy/build_space.py --push                  # build + upload to <you>/transfer-edge
    python deploy/build_space.py --push --repo org/name  # e.g. an organisation you share with

  Needs `hf auth login` first (a token with write access). The Space is created
  private, and forced back to private if it already existed as public.

Test the bundle locally before pushing:

    docker build -t transfer-edge-space build/hf_space
    docker run --rm -p 7860:8501 transfer-edge-space      # -> http://localhost:7860
"""

import argparse
import shutil
from pathlib import Path

import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "deploy"
TARGETS = {
    # target: (output folder, {deploy file: name in the bundle})
    "hf": (ROOT / "build" / "hf_space",
           {"Dockerfile": "Dockerfile", "requirements.txt": "requirements.txt", "README.md": "README.md"}),
    "streamlit": (ROOT / "build" / "streamlit_app",
                  # packages.txt: apt package names ONLY — Community Cloud passes every
                  # word to apt-get, so a comment line breaks the build. (libgomp1 is the
                  # OpenMP runtime xgboost links against.)
                  {"requirements.txt": "requirements.txt", "packages.txt": "packages.txt",
                   "README_streamlit.md": "README.md"}),
}

# exactly what the app reads at runtime (src/data/loader.py, src/models/*)
DATA_FILES = ["panel_final.parquet", "fbref_matched.parquet", "test_predictions.parquet"]
MODEL_FILES = ["xgb_model.pkl", "xgb_intervals.pkl"]
CODE_DIRS = ["app", "src"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store", ".ipynb_checkpoints")


def build(target: str = "hf") -> Path:
    out, deploy_files = TARGETS[target]
    missing = [f"data/{f}" for f in DATA_FILES if not (ROOT / "data" / f).exists()]
    missing += [f"models/{f}" for f in MODEL_FILES if not (ROOT / "models" / f).exists()]
    if missing:
        raise SystemExit(f"missing artifacts — run the pipeline and train first: {missing}")

    # clear the previous build but keep its .git: the streamlit bundle is a repository
    out.mkdir(parents=True, exist_ok=True)
    for old in out.iterdir():
        if old.name != ".git":
            shutil.rmtree(old) if old.is_dir() else old.unlink()

    for src, dst in deploy_files.items():
        shutil.copy2(DEPLOY / src, out / dst)
    for d in CODE_DIRS:
        shutil.copytree(ROOT / d, out / d, ignore=IGNORE)
    for sub, files in [("data", DATA_FILES), ("models", MODEL_FILES)]:
        (out / sub).mkdir()
        for f in files:
            shutil.copy2(ROOT / sub / f, out / sub / f)

    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and ".git" not in p.parts)
    print(f"built {out.relative_to(ROOT)}  ({size / 1e6:.1f} MB)")
    return out


def git_push(folder: Path, remote: str) -> None:
    """Commit the bundle in its own repository and push it (Community Cloud redeploys)."""
    def git(*args, check=True):
        return subprocess.run(["git", "-C", str(folder), *args], check=check, text=True,
                              capture_output=True)

    if not (folder / ".git").exists():
        git("init", "-b", "main")
    if git("remote", "get-url", "origin", check=False).returncode != 0:
        git("remote", "add", "origin", remote)
    git("add", "-A")
    if git("diff", "--cached", "--quiet", check=False).returncode == 0:
        print("bundle unchanged since the last deploy — nothing to push")
        return
    git("commit", "-m", "Deploy from Transfer Edge repository")
    pushed = git("push", "-u", "origin", "main", check=False)
    if pushed.returncode != 0:
        print(pushed.stderr.strip())
        print(f"\ncommitted but not pushed. Push from your terminal:\n  git -C {shlex.quote(str(folder))} push -u origin main")
    else:
        print(f"pushed -> {remote}")


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
    parser.add_argument("--target", choices=TARGETS, default="hf", help="hf (default) or streamlit")
    parser.add_argument("--push", action="store_true", help="hf: upload as a private Space")
    parser.add_argument("--repo", help="hf: Space id, default <your username>/transfer-edge")
    parser.add_argument("--git-remote", help="streamlit: PRIVATE GitHub repo to commit and push the bundle to")
    args = parser.parse_args()
    folder = build(args.target)
    if args.target == "hf" and args.push:
        push(folder, args.repo)
    if args.target == "streamlit" and args.git_remote:
        git_push(folder, args.git_remote)
