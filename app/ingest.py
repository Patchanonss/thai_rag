# PDF -> chunk -> embed (dense + sparse) -> store
import fitz  # PyMuPDF
from fastembed import TextEmbedding
from fastembed.sparse.bm25 import Bm25
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams,
    PointStruct, SparseVector
)
from app.config import (
    QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME,
    EMBEDDING_MODEL, CHUNK_SIZE, CHUNK_OVERLAP
)
import uuid

dense_embedder = TextEmbedding(EMBEDDING_MODEL)
sparse_embedder = Bm25("Qdrant/bm25")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def extract_text(pdf_path: str) -> list[dict]:
    """Extract text จาก PDF แบบ page by page"""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages


def chunk_text(pages: list[dict]) -> list[dict]:
    """ตัด text เป็น chunks พร้อม metadata"""
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        while start < len(text):
            end = start + CHUNK_SIZE
            chunk = text[start:end]
            if chunk.strip():
                chunks.append({
                    "chunk_id": str(uuid.uuid4()),
                    "page": page["page"],
                    "text": chunk
                })
            start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def ensure_collection():
    """สร้าง Qdrant collection ที่รองรับ dense + sparse"""
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                "dense": VectorParams(
                    size=384,
                    distance=Distance.COSINE
                )
            },
            sparse_vectors_config={
                "sparse": SparseVectorParams()
            }
        )


UPSERT_BATCH_SIZE = 64


def ingest_pdf(pdf_path: str, filename: str) -> int:
    """Main function — รับ path ของ PDF แล้ว ingest เข้า Qdrant"""
    ensure_collection()

    pages = extract_text(pdf_path)
    chunks = chunk_text(pages)
    total = len(chunks)
    print(f"📄 {filename}: {total} chunks, processing in batches of {UPSERT_BATCH_SIZE}")

    for batch_start in range(0, total, UPSERT_BATCH_SIZE):
        batch = chunks[batch_start: batch_start + UPSERT_BATCH_SIZE]
        texts = [c["text"] for c in batch]

        print(f"  🔢 dense embedding batch {batch_start}...", flush=True)
        dense_vectors = [v.tolist() for v in dense_embedder.embed(texts, parallel=0)]
        print(f"  🔢 sparse embedding batch {batch_start}...", flush=True)
        sparse_vectors = list(sparse_embedder.embed(texts, parallel=0))
        print(f"  📤 upserting batch {batch_start}...", flush=True)

        points = [
            PointStruct(
                id=batch[i]["chunk_id"],
                vector={
                    "dense": dense_vectors[i],
                    "sparse": SparseVector(
                        indices=sparse_vectors[i].indices.tolist(),
                        values=sparse_vectors[i].values.tolist()
                    )
                },
                payload={
                    "text": batch[i]["text"],
                    "page": batch[i]["page"],
                    "source": filename
                }
            )
            for i in range(len(batch))
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        print(f"  ✅ upserted {batch_start + len(batch)}/{total}")

    return total