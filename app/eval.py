from ragas.metrics import Faithfulness, ResponseRelevancy
from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings.base import BaseRagasEmbedding
from ragas.run_config import RunConfig
from langchain_groq import ChatGroq
from fastembed import TextEmbedding
from app.config import GROQ_API_KEY, EMBEDDING_MODEL, EVAL_LLM_MODEL
from app.query import query_rag
import warnings
warnings.filterwarnings("ignore")

class FastEmbedRagas(BaseRagasEmbedding):
    def __init__(self):
        super().__init__()
        self._model = TextEmbedding(EMBEDDING_MODEL)

    def embed_text(self, text: str, **kwargs) -> list[float]:
        return list(self._model.embed([text]))[0].tolist()

    async def aembed_text(self, text: str, **kwargs) -> list[float]:
        return self.embed_text(text)

    # legacy interface ที่ ResponseRelevancy ยังต้องการ
    def embed_query(self, text: str) -> list[float]:
        return self.embed_text(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_text(t) for t in texts]

TEST_QUESTIONS = [
    "โครงงานที่อยู่ในเอกสารนี้เกี่ยวกับอะไร",
    "โครงงานนี้มีวัตถุประสงค์อะไรบ้าง",
]

def run_eval():
    llm = LangchainLLMWrapper(ChatGroq(api_key=GROQ_API_KEY, model=EVAL_LLM_MODEL), bypass_n=True)
    embeddings = FastEmbedRagas()

    samples = []
    for q in TEST_QUESTIONS:
        result = query_rag(q)
        samples.append(SingleTurnSample(
            user_input=q,
            response=result["answer"],
            retrieved_contexts=[s["text"] for s in result["sources"]][:3],
        ))

    dataset = EvaluationDataset(samples=samples)

    scores = evaluate(
        dataset=dataset,
        metrics=[Faithfulness(llm=llm), ResponseRelevancy(llm=llm, embeddings=embeddings)],
        run_config=RunConfig(max_workers=2, max_retries=5, max_wait=60),
    )

    print("\n===== RAGAS EVAL =====")
    print(scores)

if __name__ == "__main__":
    run_eval()