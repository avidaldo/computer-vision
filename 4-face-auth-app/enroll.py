"""
Phase 1 — enrolment: stores one embedding per known user in a persistent ChromaDB collection.

    uv run python enroll.py

Run it once, and again whenever ENROLMENTS changes (the collection is rebuilt from scratch).
"""

import datetime

import chromadb

from config import APP_DIR, FaceAuthConfig
from quality_guard import extract_person
from verification import COSINE_CONFIGURATION, embed_face

IMAGES_DIR = APP_DIR.parent / "resources" / "images"

# One picture per user. scene1.jpg shows several people: the most confident detection is enrolled.
ENROLMENTS = {
    "Alice": IMAGES_DIR / "scene1.jpg",
    "Bob": IMAGES_DIR / "scene2.jpg",
    "Charlie": IMAGES_DIR / "scene3.jpg",
}


def main() -> None:
    config = FaceAuthConfig()
    client = chromadb.PersistentClient(path=str(config.db_path))
    if config.collection_name in {collection.name for collection in client.list_collections()}:
        client.delete_collection(config.collection_name)
    collection = client.create_collection(name=config.collection_name, configuration=COSINE_CONFIGURATION)

    for name, image_path in ENROLMENTS.items():
        print(f"{name} ({image_path.name})")
        crop, failure = extract_person(image_path, config)
        if crop is None:
            print(f"  skipped: {failure}")
            continue
        collection.add(
            ids=[f"{name.lower()}_001"],
            embeddings=[embed_face(crop, config)],
            metadatas=[{"name": name, "source": image_path.name, "enrolled_at": datetime.datetime.now().isoformat()}],
        )
        print("  enrolled")

    print(f"\n{collection.count()} users enrolled in {config.db_path}")


if __name__ == "__main__":
    main()
