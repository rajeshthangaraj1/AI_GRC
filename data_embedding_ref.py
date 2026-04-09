"""
data_embedding_ref.py
Embeds ISO 42001 reference content (controls list, management clauses,
terms & definitions, AI objectives, risk sources) into the grc_docs collection.

Run AFTER data_embedding.py and data_embedding_ppt.py:
    source .venv/bin/activate
    python data_embedding_ref.py
"""

import re
import uuid
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

# =========================
# CONFIG
# =========================
REF_FILE    = "/home/rmrobot/Desktop/Rajesh/AI_GRC/GRC_Requirement/ISO42001_reference.txt"
MODEL_PATH  = "/home/rmrobot/Desktop/Rajesh/AI_GRC/models/bge-large"
COLLECTION  = "grc_docs"
QDRANT_PATH = "./data"
FRAMEWORK   = "ISO"
SOURCE      = "ISO42001_reference"

# Each section: (section title, marker string that starts it in the file)
# Sections are extracted as everything from one marker to the next
SECTION_MARKERS = [
    (
        "ISO 42001 Annex A Controls - Complete List of 39 Controls",
        "List the controls of ISO42001"
    ),
    (
        "ISO 42001 Management Clauses - 10 Management Clauses",
        "9. number of ISO management clauses"
    ),
    (
        "ISO 42001 Terms and Definitions - 25 Key Terms",
        "Provide terms"
    ),
    (
        "ISO 42001 AI Objectives for Organizations - 11 Objectives",
        "What are  organization Objectives for AI"
    ),
    (
        "ISO 42001 Risk Sources for AI - 7 Risk Sources",
        "What are Risk sources for AI"
    ),
]

# =========================
# HELPERS
# =========================
def clean_text(text):
    text = re.sub(r"[ \t]+", " ", text)   # collapse spaces/tabs
    text = re.sub(r"\n{3,}", "\n\n", text) # max 2 consecutive newlines
    return text.strip()

def extract_sections(filepath):
    """Split file into sections based on SECTION_MARKERS.
    Returns list of (title, content) tuples."""
    with open(filepath, encoding="utf-8") as f:
        content = f.read()

    # Find positions of each marker (case-insensitive)
    positions = []
    for title, marker in SECTION_MARKERS:
        idx = content.lower().find(marker.lower())
        if idx == -1:
            print(f"[WARN] Marker not found: {marker!r}")
            continue
        positions.append((idx, title))

    positions.sort(key=lambda x: x[0])

    sections = []
    for i, (start, title) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(content)
        section_text = clean_text(content[start:end])
        sections.append((title, section_text))

    return sections

# =========================
# LOAD MODEL & QDRANT
# =========================
print("Loading embedding model...")
model = SentenceTransformer(MODEL_PATH, device="cuda")

client = QdrantClient(path=QDRANT_PATH)

if not client.collection_exists(COLLECTION):
    raise RuntimeError(
        f"Collection '{COLLECTION}' not found. "
        "Run data_embedding.py first."
    )

# =========================
# EMBED & UPSERT
# =========================
sections = extract_sections(REF_FILE)
print(f"\nFound {len(sections)} sections to embed:")

points = []

for title, body in sections:
    print(f"  - {title} ({len(body)} chars)")

    # Full embedding text
    text = clean_text(
        f"This is ISO 42001 reference content. "
        f"Section: {title}. Content: {body}"
    )

    # BM25 text — lowercase, no source prefix
    bm25_text = clean_text(f"{title}. {body}").lower()

    embedding = model.encode(
        f"passage: {text}",
        normalize_embeddings=True
    )

    points.append(PointStruct(
        id=str(uuid.uuid4()),
        vector=embedding.tolist(),
        payload={
            "text":      text,
            "bm25_text": bm25_text,
            "framework": FRAMEWORK,
            "sheet":     SOURCE,
        },
    ))

client.upsert(collection_name=COLLECTION, points=points)

# =========================
# VERIFY
# =========================
scroll = client.scroll(collection_name=COLLECTION, limit=20000)
all_points = scroll[0]

ref_count = sum(1 for p in all_points if p.payload.get("sheet") == SOURCE)
print(f"\nDONE: {len(points)} reference sections upserted into '{COLLECTION}'")
print(f"Total collection size: {len(all_points)} points")
print(f"Reference points in collection: {ref_count}")

client.close()
