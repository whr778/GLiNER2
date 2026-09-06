"""Push a trained GLiNER2 checkpoint to the Hugging Face Hub.

Run::

    uv run python tools/train/push_to_hub.py \\
        --checkpoint ./out/mmbert-small/final \\
        --repo-id <username>/gliner2-mmbert-small \\
        --private

Authentication: log in once with ``uv run huggingface-cli login`` (or set the
``HF_TOKEN`` env var). The script uploads via ``HfApi.upload_folder``, so the
target repo layout matches what ``AutoExtractor.from_pretrained`` expects.
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

from huggingface_hub import HfApi

from gliner2 import AutoExtractor


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
    model = AutoExtractor.from_pretrained(str(checkpoint), map_location="cpu")

    api = HfApi()
    print(f"Ensuring repo '{args.repo_id}' exists (private={args.private})")
    api.create_repo(repo_id=args.repo_id, private=args.private, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"Serializing model to {tmp_dir}")
        model.save_pretrained(tmp_dir)

        # save_pretrained writes only weights/config/tokenizer, so without this the
        # card train.py generated stays on disk and the Hub page renders EMPTY. The
        # Hub reads the card from README.md specifically -- MODEL_CARD.md is ignored.
        # train.py writes the card to <output_dir>/best/, so pushing an epoch
        # checkpoint directly found nothing and shipped an empty Hub page (the EKF
        # front-end run: card written to best/, push pointed at checkpoint-epoch-6).
        card = next((c for c in (checkpoint / "MODEL_CARD.md",
                                 checkpoint.parent / "best" / "MODEL_CARD.md")
                     if c.is_file()), None)
        if card:
            shutil.copyfile(card, Path(tmp_dir) / "README.md")
            print(f"Including model card ({card.stat().st_size} bytes) from {card} as README.md")
        else:
            print(f"WARNING: no MODEL_CARD.md in {checkpoint} or {checkpoint.parent}/best; "
                  "Hub page will be empty")

        print(f"Uploading to https://huggingface.co/{args.repo_id}")
        # RETRY, because a single transient failure here destroys a whole run. The
        # realsynth re-run (2026-09-06) trained 3h34m, wrote its card and metrics, then
        # lost the model to one 400 from the commit endpoint -- and the box self-
        # terminated immediately after. Proven transient, not a card, token, or repo
        # problem: the same card pushes fine and a small file pushed to that same repo
        # minutes later.
        last = None
        for attempt in range(1, 4):
            try:
                api.upload_folder(
                    folder_path=tmp_dir,
                    repo_id=args.repo_id,
                    commit_message=args.commit_message,
                )
                last = None
                break
            except Exception as e:  # noqa: BLE001 - any upload failure is worth retrying
                last = e
                print(f"[push] attempt {attempt}/3 failed: {type(e).__name__}: {e}")
                if attempt < 3:
                    delay = 30 * attempt
                    print(f"[push] retrying in {delay}s")
                    time.sleep(delay)
        if last is not None:
            raise last

    print(f"Done. View at https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
