"""Person 4 FastAPI stub. Loads Person 3 artifacts."""

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.config import FEATURE_COLS

app = FastAPI(title="Behaviour-Aware Fraud Risk API")


class ScoreRequest(BaseModel):
    customer_id: str
    amount: float = Field(gt=0)
    merchant: str
    beneficiary: str
    timestamp: str


class ScoreResponse(BaseModel):
    risk_score: float
    risk_band: str
    reason_codes: list[str]


@app.get("/health")
def health():
    return {"status": "ok", "expected_features": FEATURE_COLS}


@app.post("/score", response_model=ScoreResponse)
def score(_payload: ScoreRequest) -> ScoreResponse:
    # Person 4: look up customer baseline, build FEATURE_COLS, load RF, call explain.score_one
    return ScoreResponse(
        risk_score=0.0,
        risk_band="LOW",
        reason_codes=["API stub — wire model after python -m src.train"],
    )
