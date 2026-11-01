"""FastAPI service.  Run:  uvicorn fraud_xai.api.main:app --reload
Env var FRAUD_BUNDLE overrides the bundle path (default artifacts/latest_bundle.joblib)."""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from fraud_xai.config import DEFAULT_BUNDLE
from fraud_xai.services.inference import Scorer
from fraud_xai.services.persistence import load_bundle


class PredictRequest(BaseModel):
    records: list[dict] = Field(..., min_length=1, description="Raw transactions (same columns as the training file)")


class ExplainRequest(BaseModel):
    record: dict
    top_n: int = Field(4, ge=1, le=8)


def create_app(bundle_path: str | Path | None = None) -> FastAPI:
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        path = Path(bundle_path or os.getenv("FRAUD_BUNDLE", DEFAULT_BUNDLE))
        if path.exists():
            state["scorer"] = Scorer(load_bundle(path))
        yield

    app = FastAPI(title="Explainable Fraud Detection API", version="0.1.0", lifespan=lifespan)

    def scorer() -> Scorer:
        if "scorer" not in state:
            raise HTTPException(503, "No model bundle loaded. Export one from the Streamlit app (page 'Export').")
        return state["scorer"]

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": "scorer" in state}

    @app.get("/model")
    def model_info():
        b = scorer().b
        return {k: b[k] for k in ("model_label", "params", "threshold", "created_at", "metrics",
                                  "raw_columns", "client_supplied_columns")}

    @app.post("/predict")
    def predict(req: PredictRequest):
        sc = scorer()
        try:
            return sc.predict(req.records).to_dict("records")
        except Exception as e:  # malformed payload
            raise HTTPException(422, f"Could not score records: {e}")

    @app.post("/explain")
    def explain(req: ExplainRequest):
        sc = scorer()
        try:
            return sc.explain(req.record, req.top_n)
        except Exception as e:
            raise HTTPException(422, f"Could not explain record: {e}")

    return app


app = create_app()
