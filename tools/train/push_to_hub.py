"""Push a trained GLiNER2 checkpoint to the Hugging Face Hub.

Run::

    uv run python tools/train/push_to_hub.py \\
        --checkpoint ./out/mmbert-small/final \\
        --repo-id <username>/gliner2-mmbert-small \\
        --private

Authentication: log in once with ``uv run huggingface-cli login`` (or set the
``HF_TOKEN`` env var). The script uploads via ``HfApi.upload_folder``, so the
target repo layout matches what ``GLiNER2.from_pretrained`` expects.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from huggingface_hub import HfApi

from gliner2 import GLiNER2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Local path to a saved GLiNER2 checkpoint (e.g. ./out/mmbert-small/final).",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Target HuggingFace repo id, e.g. 'username/gliner2-mmbert-small'.",
    )
    visibility = parser.add_mutually_exclusive_group()
    visibility.add_argument(
        "--private", dest="private", action="store_true",
        help="Create the repo as private (default).",
    )
    visibility.add_argument(
        "--public", dest="private", action="store_false",
        help="Create the repo as public.",
    )
    parser.set_defaults(private=True)
    parser.add_argument(
        "--commit-message",
        default="Upload GLiNER2 checkpoint",
        help="Commit message for the upload.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    if not checkpoint.is_dir():
        raise SystemExit(f"checkpoint not found: {checkpoint}")

    print(f"Loading checkpoint from {checkpoint}")
    model = GLiNER2.from_pretrained(str(checkpoint), map_location="cpu")

    api = HfApi()
    print(f"Ensuring repo '{args.repo_id}' exists (private={args.private})")
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Serializing model to {tmp_dir}")
        model.save_pretrained(tmp_dir)
        print(f"Uploading to https://huggingface.co/{args.repo_id}")
        api.upload_folder(
            folder_path=tmp_dir,
            repo_id=args.repo_id,
            commit_message=args.commit_message,
        )

    print(f"Done. View at https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
