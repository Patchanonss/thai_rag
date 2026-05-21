# question -> retrieve -> generate
from groq import Groq
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from fastembed.sparse.bm25 import Bm25
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
from flashrank import Ranker, RerankRequest
from app.config import (
    GROQ_API_KEY, QDRANT_HOST, QDRANT_PORT,
    COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL, TOP_K
)

reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
embedder = TextEmbedding(EMBEDDING_MODEL)
sparse_embedder = Bm25("Qdrant/bm25")
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
llm = Groq(api_key=GROQ_API_KEY)


def query_rag(question: str) -> dict:
    # 1. embed คำถามทั้ง dense และ sparse
    dense_vector = list(embedder.embed([question], parallel=0))[0].tolist()
    sparse_vector = list(sparse_embedder.embed([question], parallel=0))[0]

    # 2. hybrid search — Qdrant จัดการ RRF merge ให้เลย
    results = client.query_points(
    collection_name=COLLECTION_NAME,
    prefetch=[
        Prefetch(query=dense_vector, using="dense", limit=20),
        Prefetch(
            query=SparseVector(
                indices=sparse_vector.indices.tolist(),
                values=sparse_vector.values.tolist()
            ),
            using="sparse",
            limit=20
        )
    ],
    query=FusionQuery(fusion=Fusion.RRF),
    limit=TOP_K
    ).points

    # 3. rerank แล้ว build context
    chunks = [r.payload["text"] for r in results]
    request = RerankRequest(query=question, passages=[{"text": c} for c in chunks])
    reranked = reranker.rerank(request)
    if reranked[0]["score"] < 0.1:
        return {
            "answer": "ไม่พบข้อมูลในเอกสารที่ให้มา",
            "sources": []
        }

    context_parts = []
    sources = []
    top_chunks = reranked[:5]          # list of dicts พร้อม score
    payload_map = {r.payload["text"]: r.payload for r in results}  # lookup table

    for chunk in top_chunks:           # loop จาก reranked order ✅
        text = chunk["text"]
        payload = payload_map.get(text, {})  # ดึง metadata (source, page) จาก Qdrant
        context_parts.append(text)
        sources.append({
            "source": payload.get("source"),
            "page": payload.get("page"),
            "rerank_score": round(float(chunk["score"]), 4),
            "text": text  # ← เพิ่ม
        })

    context = "\n\n---\n\n".join(context_parts)

    # 4. สร้าง prompt แล้วส่ง LLM
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "คุณคือผู้ช่วยตอบคำถามจากเอกสาร ตอบเป็นภาษาไทย"
                    "ตอบจากเอกสารที่ให้มาเท่านั้น"
                    "สรุปและตอบตรงประเด็นจากเอกสารที่ให้มา ห้ามเพิ่มเติมข้อมูลที่ไม่มีในเอกสาร"  # ← เพิ่ม
                    "ถ้าไม่มีข้อมูลในเอกสาร ให้ตอบว่า 'ไม่พบข้อมูลในเอกสารที่ให้มา'"
                    )                   
            },
            {
                "role": "user",
                "content": f"เอกสาร:\n{context}\n\nคำถาม: {question}"
            }
        ]
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": sources
    }