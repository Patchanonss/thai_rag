# question -> retrieve -> generate
from groq import Groq
from qdrant_client import QdrantClient
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client.models import Prefetch, FusionQuery, Fusion, SparseVector
from flashrank import Ranker, RerankRequest
from app.config import (
    GROQ_API_KEY, QDRANT_HOST, QDRANT_PORT,
    COLLECTION_NAME, EMBEDDING_MODEL, LLM_MODEL, TOP_K
)

reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
embedder = TextEmbedding(EMBEDDING_MODEL)
sparse_embedder = SparseTextEmbedding("prithivida/Splade_PP_en_v1")  # ← เพิ่ม
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
llm = Groq(api_key=GROQ_API_KEY)


def query_rag(question: str) -> dict:
    # 1. embed คำถามทั้ง dense และ sparse
    dense_vector = list(embedder.embed([question]))[0].tolist()
    sparse_vector = list(sparse_embedder.embed([question]))[0]

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
    top_chunks = [r["text"] for r in reranked[:5]]

    context_parts = []
    sources = []
    for r in results:
        if r.payload["text"] in top_chunks:
            context_parts.append(r.payload["text"])
            sources.append({
                "source": r.payload["source"],
                "page": r.payload["page"],
                "score": round(r.score, 4)
            })

    context = "\n\n---\n\n".join(context_parts)

    # 4. สร้าง prompt แล้วส่ง LLM
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "คุณคือผู้ช่วยตอบคำถามจากเอกสาร ตอบเป็นภาษาไทย "
                    "ตอบจากเอกสารที่ให้มาเท่านั้น "
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