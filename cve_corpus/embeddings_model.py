import json
import os
import re
from pathlib import Path
from typing import List, Tuple

import nltk
import numpy as np
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.datasets import DenoisingAutoEncoderDataset
from torch.utils.data import DataLoader


def save_parquet(array, filename):
    import pyarrow as pa
    import pyarrow.parquet as pq

    # Convert the NumPy array columns into PyArrow arrays
    #    .T transposes the array so we can iterate over columns
    arrow_arrays = [pa.array(col) for col in array.T]

    # Create a PyArrow Table from the arrays
    #    You must name your columns for the Parquet format
    column_names = [f"col_{i}" for i in range(array.shape[1])]
    pa_table = pa.Table.from_arrays(arrow_arrays, names=column_names)

    # Write the PyArrow Table to a Parquet file with zlib compression
    #    The `compression` parameter is set to 'gzip' for zlib
    pq.write_table(pa_table, f"{filename}.parquet", compression="gzip")


# -----------------------------
# IO: train.jsonl (CVE docs)
# -----------------------------
def load_train_jsonl(path: str) -> List[str]:
    """
    Expects JSONL with rows like:
    {"text": ["CVE-2014-9374", "(no title)", "CVE-... | ... | n/a n/a"]}
    Returns list[str] documents.
    """
    docs: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            row = obj.get("text")
            if isinstance(row, list) and len(row) >= 3:
                cve_id = str(row[0]).strip()
                title = str(row[1]).strip()
                text = str(row[2]).strip()
                doc = f"{cve_id} {title}\n{text}".strip()
                if doc:
                    docs.append(doc)
            elif isinstance(row, str):
                docs.append(row.strip())

    return docs


# -----------------------------
# IO: cwe_glossary.txt
# -----------------------------
def load_cwe_glossary(path: str) -> List[str]:
    """
    Expects blocks like:
      TERM: AITM
      DEFINITION: Adversary-in-the-Middle. Formerly MITM.

    Returns list[str] docs like: "AITM: Adversary-in-the-Middle..."
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Split on TERM lines, keep content
    # This tolerates extra blank lines.
    term_blocks = re.split(r"\n(?=TERM:\s*)", raw.strip(), flags=re.MULTILINE)
    out: List[str] = []

    for block in term_blocks:
        m_term = re.search(r"TERM:\s*(.+)", block)
        m_def = re.search(r"DEFINITION:\s*(.+)", block, flags=re.DOTALL)
        if not m_term or not m_def:
            continue

        term = m_term.group(1).strip()
        definition = re.sub(r"\s+", " ", m_def.group(1)).strip()
        if term and definition:
            out.append(f"{term}: {definition}")

    return out


# -----------------------------
# Optional: sentence split
# -----------------------------
def to_sentences(docs: List[str], max_sentences_per_doc: int = 8) -> List[str]:
    """
    DAE works well on sentences. Keep a small cap per doc to avoid overweighting long CVEs.
    """
    nltk.download("punkt", quiet=True)
    sents: List[str] = []
    for doc in docs:
        for s in nltk.sent_tokenize(doc):
            s = s.strip()
            if s:
                sents.append(s)
        # cap per doc (simple)
        if max_sentences_per_doc is not None and max_sentences_per_doc > 0:
            # keep last cap window to preserve "impact / mitigation" tail
            sents = sents[:-max_sentences_per_doc] + sents[-max_sentences_per_doc:]
    return sents


# -----------------------------
# Fine-tune + save
# -----------------------------
def finetune_dae(
    corpus_sentences: List[str],
    base_model: str = "all-MiniLM-L6-v2",
    out_dir: str = "./domain_adapted_model",
    batch_size: int = 32,
    epochs: int = 1,
    lr: float = 3e-5,
):
    model = SentenceTransformer(base_model)
    # keep consistent with your tokenizer config expectations (small context)
    model.max_seq_length = 256

    train_dataset = DenoisingAutoEncoderDataset(corpus_sentences)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    train_loss = losses.DenoisingAutoEncoderLoss(model, tie_encoder_decoder=True)

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=epochs,
        weight_decay=0.0,
        optimizer_params={"lr": lr},
        show_progress_bar=True,
    )

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.save(str(out_path))
    return str(out_path)


# -----------------------------
# Embeddings build (your logic)
# -----------------------------
def build_embeddings(
    texts: List[str],
    model_path: str = "./domain_adapted_model",
    cache_file: str = "cve_embeddings_cache.npy",
):
    """
    Same caching & scaling behavior as your existing ArrowSpace CVE script. [file:22]
    """
    if os.path.exists(cache_file):
        try:
            X = np.load(cache_file)
            if len(X) == len(texts):
                return X
        except Exception:
            pass

    model = SentenceTransformer(model_path)
    X = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)

    # Preserve your original scaling
    X_scaled = X.astype(np.float64) * 1.2e1  # [file:22]
    np.save(cache_file, X_scaled)
    save_parquet("cve_embeddings_cache.parquet", X_scaled)
    return X_scaled


def main():
    dataset_root = "train.jsonl"
    vocabulary_root = "cwe_glossary.txt"

    cve_docs = load_train_jsonl(dataset_root)
    cwe_docs = load_cwe_glossary(vocabulary_root)

    # Merge and sentence-split
    merged_docs = cve_docs + cwe_docs
    corpus = to_sentences(merged_docs, max_sentences_per_doc=8)

    print(f"Loaded CVE docs: {len(cve_docs)}")
    print(f"Loaded CWE glossary entries: {len(cwe_docs)}")
    print(f"Training sentences: {len(corpus)}")

    out_model_dir = finetune_dae(
        corpus_sentences=corpus,
        base_model="all-MiniLM-L6-v2",
        out_dir="./domain_adapted_model",
        batch_size=32,
        epochs=1,
        lr=3e-5,
    )
    print(f"Model saved to: {out_model_dir}")

    # Optional: build embeddings for ArrowSpace
    X = build_embeddings(
        merged_docs, model_path=out_model_dir, cache_file="cve_embeddings_cache.npy"
    )
    print(f"Embeddings shape: {X.shape}")


if __name__ == "__main__":
    main()
