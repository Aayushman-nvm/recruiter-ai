"""
Script 02 (v2): Crash-safe, memory-efficient candidate embedding pipeline.

Designed for constrained hardware (8GB RAM, ~3-4GB free for script).

Architecture:
  Stream JSONL → embed batch → write to open_memmap (.npy format) → discard → repeat
  Texts streamed to a staging file (not kept in RAM) → consolidated to pkl at end
  Checkpoint tracks progress — crash-safe resume with no re-embedding

Memory profile:
  Model:         ~500MB (loaded once, stays resident)
  Per batch:     EMBED_BATCH × ~500 chars text ≈ negligible
                 EMBED_BATCH × 768 × 4B float32 ≈ 0.2MB at batch=64
  Texts in RAM:  Zero during embedding (written to staging file on disk)
  End of run:    All texts read from staging file to write pkl (~50MB peak, brief)
  Peak total:    ~600MB for the script

Crash safety:
  embed_checkpoint.json written after every CHECKPOINT_EVERY batches
  Re-run script to resume — already-embedded rows are skipped, not re-done
  Staging file for texts is also preserved across restarts

Format guarantee:
  candidate_embeddings.npy is written via numpy.lib.format.open_memmap
  This produces a valid .npy file (with numpy header) that np.load() reads correctly.
  Plain np.memmap() creates a raw binary file WITHOUT the header — np.load() would fail.

CRITICAL ordering invariant:
  Row N in candidate_embeddings.npy == line N in candidate_ids.txt == item N in candidate_texts.pkl
  Never shuffle these independently. Never sort the JSONL before processing.

Input:  data/candidates.jsonl.gz
Output:
  precomputed/candidate_embeddings.npy   shape (N, 768), float16 — valid .npy format
  precomputed/candidate_ids.txt          one CAND_XXXXXXX per line
  precomputed/candidate_texts.pkl        list of N text strings (BM25 + cross-encoder)
  [precomputed/embed_checkpoint.json]    deleted on clean finish
  [precomputed/_texts_staging.txt]       deleted on clean finish
"""

import gzip
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
from numpy.lib.format import open_memmap   # produces valid .npy files, not raw binary
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# ── Path setup ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from utils import build_candidate_text  # noqa: E402 (import after sys.path insert)

# ── Configuration ───────────────────────────────────────────────────────────────
MODEL_NAME  = "BAAI/bge-small-en-v1.5"   
EMBED_DIM   = 384

# Batch size for model.encode() — affects GPU/CPU throughput and peak RAM.
# At 64: ~0.2MB RAM for one batch of float32 embeddings. Safe for 8GB systems.
# At 128: ~0.4MB. Also fine, slightly faster. Increase if RAM allows.
EMBED_BATCH = 64

# Write memmap and update checkpoint every N batches.
# 50 batches × 64 rows = 3200 rows per checkpoint cycle.
# Higher = fewer disk flushes = faster run. Lower = finer-grained resume.
CHECKPOINT_EVERY = 50   # checkpoints every ~3200 rows (~30 per full 100K run)

# ── Paths ────────────────────────────────────────────────────────────────────────
DATA_PATH        = ROOT / "data" / "candidates.jsonl.gz"
OUT_DIR          = ROOT / "precomputed"
EMB_PATH         = OUT_DIR / "candidate_embeddings.npy"
IDS_PATH         = OUT_DIR / "candidate_ids.txt"
TEXTS_PATH       = OUT_DIR / "candidate_texts.pkl"
CHECKPOINT_PATH  = OUT_DIR / "embed_checkpoint.json"
TEXTS_STAGING    = OUT_DIR / "_texts_staging.txt"   # one JSON-escaped text per line


# ── Checkpoint ──────────────────────────────────────────────────────────────────

def load_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        with open(CHECKPOINT_PATH) as f:
            cp = json.load(f)
        print(f"Resuming: {cp['rows_done']} rows already embedded.")
        return cp
    return {"rows_done": 0, "total_n": None}


def save_checkpoint(rows_done: int, total_n: int | None) -> None:
    with open(CHECKPOINT_PATH, "w") as f:
        json.dump({"rows_done": rows_done, "total_n": total_n}, f)


def delete_checkpoint() -> None:
    for p in (CHECKPOINT_PATH, TEXTS_STAGING):
        if p.exists():
            p.unlink()


# ── Count candidates ─────────────────────────────────────────────────────────────

def count_candidates() -> int:
    """Single JSONL pass to get exact N for memmap pre-allocation (~15s for 100K)."""
    print("Counting candidates for memmap pre-allocation...")
    n = 0
    with gzip.open(DATA_PATH, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    print(f"Total candidates: {n}")
    return n


# ── Memmap (valid .npy format) ────────────────────────────────────────────────────

def get_memmap(total_n: int) -> np.memmap:
    """
    Open or create a .npy-format memmap for writing.

    open_memmap() from numpy.lib.format writes the standard numpy .npy header,
    so the resulting file is identical to one produced by np.save() — np.load()
    reads it correctly, shape and dtype are preserved.

    Plain np.memmap() creates a headerless raw binary file. np.load() raises:
      ValueError: magic bytes not found — this would break rank.py.
    """
    if EMB_PATH.exists():
        # Resume: open existing file for read+write (mode="r+")
        return open_memmap(str(EMB_PATH), dtype=np.float16, mode="r+",
                           shape=(total_n, EMBED_DIM))
    else:
        # First run: create new file (mode="w+"), pre-filled with zeros
        return open_memmap(str(EMB_PATH), dtype=np.float16, mode="w+",
                           shape=(total_n, EMBED_DIM))


def trim_memmap(old_n: int, actual_n: int) -> None:
    """Trim memmap to actual row count if JSONL had fewer rows than expected."""
    if old_n == actual_n:
        return
    print(f"Trimming memmap {old_n} → {actual_n} rows...")
    # Read the valid portion into RAM (actual_n × 768 × 2B, typically ≤154MB)
    src = open_memmap(str(EMB_PATH), dtype=np.float16, mode="r",
                      shape=(old_n, EMBED_DIM))
    data = np.array(src[:actual_n])
    del src
    dst = open_memmap(str(EMB_PATH), dtype=np.float16, mode="w+",
                      shape=(actual_n, EMBED_DIM))
    dst[:] = data
    dst.flush()
    del dst
    print("Trim complete.")


# ── Main ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)

    cp = load_checkpoint()
    rows_done: int = cp["rows_done"]
    total_n: int | None = cp["total_n"]

    # ── Count total candidates (once, cached in checkpoint) ──────────────────
    if total_n is None:
        total_n = count_candidates()
        save_checkpoint(rows_done, total_n)

    emb_size_mb = total_n * EMBED_DIM * 2 / 1e6
    print(f"\nPlan: {total_n} candidates, {EMBED_DIM}-dim float16")
    print(f"Memmap size on disk: {emb_size_mb:.1f} MB")
    print(f"Batch size: {EMBED_BATCH}, checkpoint every {CHECKPOINT_EVERY} batches "
          f"(~{CHECKPOINT_EVERY * EMBED_BATCH} rows)\n")

    # ── Load model ────────────────────────────────────────────────────────────
    t0 = time.time()
    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print(f"Device: {model.device}")
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # ── Open outputs ──────────────────────────────────────────────────────────
    mmap = get_memmap(total_n)

    # IDs file: append when resuming, overwrite on fresh start
    ids_mode = "a" if rows_done > 0 else "w"
    ids_file = open(IDS_PATH, ids_mode, encoding="utf-8")

    # Texts staging file: append when resuming, overwrite on fresh start.
    # Each line is a JSON-encoded string → safe for any text content.
    # Kept on disk, NOT in RAM, for the entire embedding run.
    texts_mode = "a" if rows_done > 0 else "w"
    texts_staging_file = open(TEXTS_STAGING, texts_mode, encoding="utf-8")

    # ── Stream, batch, embed, flush ───────────────────────────────────────────
    batch_ids:   list[str] = []
    batch_texts: list[str] = []
    row_index   = 0          # absolute position in JSONL
    batch_count = 0          # batches since last checkpoint

    print(f"Starting embedding. Skipping first {rows_done} rows...")
    t_start = time.time()

    with gzip.open(DATA_PATH, "rt", encoding="utf-8") as f:
        pbar = tqdm(f, desc="Streaming", unit="cand", total=total_n)
        for line in pbar:
            line = line.strip()
            if not line:
                continue

            if row_index < rows_done:
                row_index += 1
                continue   # already embedded in a prior run — skip

            c = json.loads(line)
            batch_ids.append(c["candidate_id"])
            batch_texts.append(build_candidate_text(c))
            row_index += 1

            is_last_row = (row_index == total_n)

            if len(batch_ids) == EMBED_BATCH or is_last_row:
                if not batch_ids:
                    continue

                # ── Embed this batch ──────────────────────────────────────
                embs = model.encode(
                    batch_texts,
                    batch_size=EMBED_BATCH,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                ).astype(np.float16)

                # ── Write to memmap slice ─────────────────────────────────
                start = rows_done
                end   = rows_done + len(batch_ids)
                mmap[start:end] = embs

                # ── Append IDs to file ────────────────────────────────────
                ids_file.write("\n".join(batch_ids) + "\n")

                # ── Append texts to staging file (not in RAM) ─────────────
                for text in batch_texts:
                    texts_staging_file.write(json.dumps(text) + "\n")

                rows_done  += len(batch_ids)
                batch_count += 1

                # ── Flush and checkpoint every CHECKPOINT_EVERY batches ───
                if batch_count >= CHECKPOINT_EVERY or is_last_row:
                    mmap.flush()
                    ids_file.flush()
                    texts_staging_file.flush()
                    save_checkpoint(rows_done, total_n)
                    batch_count = 0

                pbar.set_postfix({"done": rows_done})

                # ── Discard batch — keep RAM flat ─────────────────────────
                batch_ids.clear()
                batch_texts.clear()
                del embs   # explicit: free the float16 array

    ids_file.close()
    texts_staging_file.close()

    # ── Trim memmap if actual count differs from pre-allocated size ───────────
    if rows_done != total_n:
        del mmap
        trim_memmap(total_n, rows_done)
    else:
        mmap.flush()
        del mmap

    actual_n = rows_done

    # ── Fix candidate_ids.txt (trim trailing newline, verify count) ───────────
    with open(IDS_PATH, "r", encoding="utf-8") as f:
        all_ids = [ln.rstrip("\n") for ln in f if ln.strip()]
    assert len(all_ids) == actual_n, \
        f"ID count mismatch: {len(all_ids)} ids vs {actual_n} embeddings"
    with open(IDS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(all_ids))

    # ── Consolidate texts staging → candidate_texts.pkl ──────────────────────
    # Only now do we load all texts into RAM — brief peak of ~50MB for 100K texts.
    # This is unavoidable: pickle.dump() needs the full list in memory once.
    print(f"\nConsolidating {actual_n} texts → candidate_texts.pkl ...")
    all_texts: list[str] = []
    with open(TEXTS_STAGING, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading staging", total=actual_n):
            line = line.strip()
            if line:
                all_texts.append(json.loads(line))

    assert len(all_texts) == actual_n, \
        f"Text count mismatch: {len(all_texts)} texts vs {actual_n} embeddings"

    with open(TEXTS_PATH, "wb") as f:
        pickle.dump(all_texts, f)
    del all_texts   # free immediately

    # ── Final verification ────────────────────────────────────────────────────
    # Quick sanity: reload first row of memmap and check shape
    check = np.load(str(EMB_PATH), mmap_mode="r")
    assert check.shape == (actual_n, EMBED_DIM), \
        f"Embedding shape mismatch: {check.shape} expected ({actual_n}, {EMBED_DIM})"
    assert check.dtype == np.float16, f"Expected float16, got {check.dtype}"
    del check

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done. {actual_n} candidates embedded in {elapsed/3600:.2f}h.")
    print(f"  candidate_embeddings.npy : {EMB_PATH.stat().st_size / 1e6:.1f} MB  (valid .npy)")
    print(f"  candidate_ids.txt        : {actual_n} lines")
    print(f"  candidate_texts.pkl      : {TEXTS_PATH.stat().st_size / 1e6:.1f} MB")
    print(f"{'='*60}")

    delete_checkpoint()
    print("Staging and checkpoint files removed. All clean.")


if __name__ == "__main__":
    main()
