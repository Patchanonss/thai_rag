from fastapi import FastAPI, UploadFile, File, BackgroundTasks
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
async def ingest(file: UploadFile = File(...), background_tasks: BackgroundTasks = None):
    # save ไฟล์ชั่วคราวก่อน
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    def run_and_cleanup():
        try:
            count = ingest_pdf(tmp_path, file.filename)
            print(f"✅ ingest เสร็จ: {count} chunks")
        except Exception as e:
            print(f"❌ ingest error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            os.unlink(tmp_path)

    background_tasks.add_task(run_and_cleanup)
    return {"message": "กำลัง ingest อยู่ ดู progress ที่ docker logs", "filename": file.filename}

class QuestionRequest(BaseModel):
    question: str

@app.post("/query")
def query(req: QuestionRequest):
    return query_rag(req.question)