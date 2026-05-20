from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.query import query_rag
from pydantic import BaseModel
import tempfile, os
from app.ingest import ingest_pdf

app = FastAPI(title="Thai RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    # save ไฟล์ชั่วคราวก่อน
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        count = ingest_pdf(tmp_path, file.filename)
        return {"message": f"ingested {count} chunks", "filename": file.filename}
    finally:
        os.unlink(tmp_path)  # ลบไฟล์ temp ทิ้ง

class QuestionRequest(BaseModel):
    question: str

@app.post("/query")
def query(req: QuestionRequest):
    return query_rag(req.question)