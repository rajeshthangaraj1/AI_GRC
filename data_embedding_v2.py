import pandas as pd
import numpy as np
import re
import uuid
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

# =========================
# CONFIG
# =========================
FILE_PATH = "/home/rmrobot/Desktop/Rajesh/AI_GRC/file/Combined_EU_NIST_ISO_AI_Compliance_2311.xlsx"
MODEL_PATH = "/home/rmrobot/Desktop/Rajesh/AI_GRC/models/bge-large"
COLLECTION_NAME = "grc_docs_v2"
QDRANT_PATH = "./data_v2"
BATCH_SIZE = 100

# Skip dashboard / utility sheets
SKIP_SHEETS = {
    "Dashboard-ISO42001",
    "Dashboard-EUAIACT",
    "Dashboard-NIST",
    "Control Dashboard",
    "Sheet3",
    "Sheet7",
}

# Values to treat as empty/placeholder — do NOT embed these
TBA_VALUES = {"tbd", "tba", "n/a", "na", "not assessed", "nan", "none", ""}

# Keywords: any column whose name contains one of these is a content column
CONTENT_KEYWORDS = [
    "clause", "subclause", "number", "title", "description", "topic",
    "control", "statement", "purpose", "question", "evidence",
    "section", "subsection", "article", "category", "function",
    "group", "roadmap", "objective", "activity", "criteria",
    "risk level", "priority", "requirement", "guidance",
]

# Cross-reference columns — include in embedding but EXCLUDE from BM25
# (avoids "4.1" in "ISO 42K References: 4.1" matching clause queries)
BM25_EXCLUDE_PATTERNS = [
    "iso 42k ref", "eu ai act ref", "42k ref", "references", "reference",
]

# Sheets where structural columns (clause number, title) need forward-filling
# because sub-bullet rows have blank values in those columns
FFILL_SHEETS = {"ISO-42K Management Clauses", "ISO-42K-Annex A"}

# Keywords identifying structural/navigation columns that need forward-filling
FFILL_KEYWORDS = ["clause", "subclause", "number", "title", "topic", "section"]

# Maximum empty/TBD ratio for a column to be included
MAX_TBD_RATIO = 0.5

# =========================
# LOAD MODEL
# =========================
print("Loading embedding model...")
model = SentenceTransformer(MODEL_PATH, device="cuda")

# =========================
# INIT QDRANT
# =========================
client = QdrantClient(path=QDRANT_PATH)

if client.collection_exists(COLLECTION_NAME):
    print("Deleting old v2 collection...")
    client.delete_collection(COLLECTION_NAME)

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
)

# =========================
# HELPERS
# =========================
def clean_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def is_placeholder(value):
    """Return True if value is TBD/TBA/empty/etc."""
    return str(value).strip().lower() in TBA_VALUES

def is_summary_row(row):
    """True for Excel summary rows like 'Clause 4 - Compliance Average'."""
    for val in row.values:
        if "compliance average" in str(val).lower():
            return True
    return False

def detect_framework(sheet_name, text_sample=""):
    s = (sheet_name + " " + text_sample).lower()
    if "eu" in s or "ai act" in s:
        return "EU AI Act"
    elif "nist" in s:
        return "NIST AI RMF"
    elif "iso" in s:
        return "ISO"
    return "General"

def needs_header_row1(df):
    """True only when column 0 is Unnamed — real headers are in row 1.
    Checking only column 0 avoids false-triggers on sheets that have a
    proper first column but unnamed columns further right."""
    return "Unnamed:" in str(df.columns[0])

def preprocess_df(df, sheet):
    """Clean up the dataframe before column selection:
    1. Remove 'Compliance Average' summary rows.
    2. Forward-fill structural columns (clause number, title) for ISO sheets
       so every sub-row inherits its parent clause context.
    """
    # Remove summary rows
    summary_mask = df.apply(is_summary_row, axis=1)
    if summary_mask.any():
        print(f"    Removed {summary_mask.sum()} summary rows")
    df = df[~summary_mask].reset_index(drop=True)

    # Forward-fill structural columns for ISO hierarchical sheets
    if sheet in FFILL_SHEETS:
        for i, col in enumerate(df.columns):
            col_lower = str(col).strip().lower()
            if any(kw in col_lower for kw in FFILL_KEYWORDS):
                # Replace empty strings with NaN, ffill, fill remaining NaN with ""
                series = df.iloc[:, i].replace("", np.nan)
                series = series.ffill().fillna("")
                df.iloc[:, i] = series

    return df

def select_content_columns(df):
    """Return a sub-DataFrame with only meaningful, non-TBD-heavy columns.

    Logic:
    1. Always include columns A-E (positions 0-4).
    2. Include any column whose name matches a CONTENT_KEYWORD.
    3. Drop columns where >{MAX_TBD_RATIO} of values are placeholders
       (after forward-fill, structural columns will pass this check).
    """
    n_cols = len(df.columns)
    selected = set(range(min(5, n_cols)))  # always include A-E

    for i, col in enumerate(df.columns):
        col_lower = str(col).strip().lower()
        if any(kw in col_lower for kw in CONTENT_KEYWORDS):
            selected.add(i)

    # Filter out high-placeholder columns
    final_positions = []
    for pos in sorted(selected):
        col_data = df.iloc[:, pos]
        total = len(col_data)
        if total == 0:
            continue
        tbd_count = sum(1 for v in col_data if is_placeholder(str(v).strip()))
        if (tbd_count / total) < MAX_TBD_RATIO:
            final_positions.append(pos)

    return df.iloc[:, final_positions]

def build_bm25_text(col_names, values):
    """BM25 text excludes cross-reference columns to prevent false positives.
    E.g. 'ISO 42K References: 4.1 A.2.2' would otherwise match clause queries."""
    parts = []
    for col_name, val in zip(col_names, values):
        col_lower = str(col_name).strip().lower()
        val_str = str(val).strip()
        if is_placeholder(val_str):
            continue
        # Skip cross-reference columns for BM25
        if any(excl in col_lower for excl in BM25_EXCLUDE_PATTERNS):
            continue
        parts.append(val_str)
    return clean_text(" ".join(parts)).lower()

# =========================
# LOAD EXCEL
# =========================
xls = pd.ExcelFile(FILE_PATH)

points = []
total = 0
skipped_rows = 0

for sheet in xls.sheet_names:
    if sheet in SKIP_SHEETS:
        print(f"Skipping: {sheet}")
        continue

    print(f"\nProcessing sheet: {sheet}")

    try:
        df = pd.read_excel(FILE_PATH, sheet_name=sheet)
    except Exception:
        df = pd.read_excel(FILE_PATH, sheet_name=sheet, header=None)

    # EU AI ACT / NIST AI RMF: column 0 is 'Unnamed: 0' → real headers in row 1
    if needs_header_row1(df):
        df = pd.read_excel(FILE_PATH, sheet_name=sheet, header=1)

    df = df.fillna("")
    print(f"  Rows: {len(df)} | Columns: {len(df.columns)}")

    # Step 1: preprocess (remove summary rows, forward-fill clause numbers)
    df = preprocess_df(df, sheet)
    print(f"  After preprocess: {len(df)} rows")

    # Step 2: select content columns only
    df_focused = select_content_columns(df)
    print(f"  Selected columns ({len(df_focused.columns)}): {list(df_focused.columns)}")

    col_names = list(df_focused.columns)

    for idx, row in df_focused.iterrows():

        # Build full embedding text from all non-placeholder cells
        parts = []
        for col_name, val in zip(col_names, row.values):
            val_str = str(val).strip()
            if not is_placeholder(val_str):
                parts.append(f"{col_name}: {val_str}")

        if not parts:
            skipped_rows += 1
            continue

        raw_text = clean_text(" | ".join(parts))
        if len(raw_text) < 20:
            skipped_rows += 1
            continue

        framework = detect_framework(sheet, raw_text[:200])

        text = clean_text(
            f"This is an AI compliance requirement from {framework}. "
            f"Source Sheet: {sheet}. Content: {raw_text}"
        )

        embedding = model.encode(
            f"passage: {text}",
            normalize_embeddings=True
        )

        # BM25 text: exclude cross-reference columns to avoid false positives
        bm25 = build_bm25_text(col_names, row.values)

        payload = {
            "text": text,
            "framework": framework,
            "sheet": sheet,
            "bm25_text": bm25,
        }

        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload=payload,
        ))
        total += 1

        if total <= 5:
            print(f"  SAMPLE text: {text[:300]}")
            print(f"  SAMPLE bm25: {bm25[:200]}")

        if len(points) >= BATCH_SIZE:
            client.upsert(collection_name=COLLECTION_NAME, points=points)
            points = []

# =========================
# FINAL INSERT
# =========================
if points:
    client.upsert(collection_name=COLLECTION_NAME, points=points)

print(f"\nDONE: {total} records stored | {skipped_rows} rows skipped (empty/TBD)")

# =========================
# VERIFY
# =========================
scroll = client.scroll(collection_name=COLLECTION_NAME, limit=20000)

frameworks = {}
for p in scroll[0]:
    fw = p.payload.get("framework", "Unknown")
    frameworks[fw] = frameworks.get(fw, 0) + 1

print("\nFramework Distribution:")
for k, v in frameworks.items():
    print(f"  {k}: {v}")

client.close()
