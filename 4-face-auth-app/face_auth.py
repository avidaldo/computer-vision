"""
Phase 2 — authentication: decides whether the person in an image may enter.

    uv run python face_auth.py                                   # the image set in IMAGE_PATH
    uv run python face_auth.py ../resources/images/face1.jpg    # any other image

Any failing step denies access: unreadable image, poor quality, nobody detected, no enrolled users,
or a similarity below the threshold. An access-control system must fail closed.
"""

import argparse
from pathlib import Path

import chromadb

from config import FaceAuthConfig
from quality_guard import extract_person
from verification import COSINE_CONFIGURATION, verify_identity


def main() -> None:
    parser = argparse.ArgumentParser(description="Face access control")
    parser.add_argument("image", nargs="?", type=Path, help="image to check (default: IMAGE_PATH from .env)")
    arguments = parser.parse_args()

    config = FaceAuthConfig()
    image_path = arguments.image.resolve() if arguments.image else config.image_path
    print(f"Checking {image_path.name}")

    crop, failure = extract_person(image_path, config)
    if crop is None:
        print(f"ACCESS DENIED: {failure}")
        return

    collection = chromadb.PersistentClient(path=str(config.db_path)).get_or_create_collection(
        config.collection_name, configuration=COSINE_CONFIGURATION
    )
    result = verify_identity(crop, collection, config)
    if result.similarity is not None:
        print(f"  closest enrolled user: {result.matched_name} (similarity {result.similarity:.3f})")

    if result.verified:
        print(f"ACCESS GRANTED: welcome, {result.matched_name}")
    else:
        print(f"ACCESS DENIED: {result.reason}")


if __name__ == "__main__":
    main()
