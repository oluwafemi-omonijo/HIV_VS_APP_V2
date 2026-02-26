# api/main.py — FIXED for robust multi-horizon (Y1–Y5)

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# -----------------------------
# ENV + SECURITY
# -----------------------------
load_dotenv()
API_KEY = (os.getenv("API_KEY") or "").strip()

def require_api_key(x_api_key: Optional[str] = Header(default=None, alias="X-API-Key")):
    if not API_KEY:
        return True
    if not x_api_key or x_api_key.strip() != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid API key.")
    return True

# -----------------------------
# PATHS
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# -----------------------------
# FASTAPI APP
# -----------------------------
app = FastAPI(title="HIV Viral Suppression Predictor API", version="1.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# SCHEMAS
# -----------------------------
Horizon = Literal["Y1", "Y2", "Y3", "Y4", "Y5"]
ThresholdStrategy = Literal["f1", "youden", "roc_top_left", "custom"]

class PredictRequest(BaseModel):
    horizon: Horizon
    threshold_strategy: ThresholdStrategy = "youden"
    custom_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    features: Dict[str, Any]

class PredictResponse(BaseModel):
    model_tag: str
    horizon: str
    threshold_strategy: str
    threshold_used: float
    probability: float
    predicted_class: int
    risk_category: str
    missing_features_filled: str = ""

class BatchPredictRequest(BaseModel):
    horizon: Horizon
    threshold_strategy: ThresholdStrategy = "youden"
    custom_threshold: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    rows: List[Dict[str, Any]]

class BatchPredictResponse(BaseModel):
    model_tag: str
    horizon: str
    threshold_strategy: str
    threshold_used: float
    missing_features_filled: str = ""
    results: List[Dict[str, Any]]

# -----------------------------
# REGISTRY
# -----------------------------
REGISTRY: Dict[str, Dict[str, Any]] = {}

HORIZON_TO_TAG = {
    "Y1": "DeepANN_T0_to_Y1",
    "Y2": "DeepANN_T0Y1_to_Y2",
    "Y3": "DeepANN_T0Y1Y2_to_Y3",
    "Y4": "DeepANN_T0Y1Y2Y3_to_Y4",
    "Y5": "DeepANN_T0Y1Y2Y3Y4_to_Y5",
}

STRATEGY_KEY_MAP = {
    "f1": "F1",
    "youden": "Youden",
    "roc_top_left": "ROC_top_left",
}

def load_registry() -> None:
    if not MODELS_DIR.exists():
        raise RuntimeError(f"models/ folder not found at: {MODELS_DIR}")

    meta_files = list(MODELS_DIR.glob("*_metadata.json"))
    if not meta_files:
        raise RuntimeError(f"No *_metadata.json found in {MODELS_DIR}")

    skipped: List[str] = []
    for meta_path in meta_files:
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            model_tag = meta.get("model_tag") or meta_path.name.replace("_metadata.json", "")
            model_path = MODELS_DIR / f"{model_tag}.keras"

            if not model_path.exists():
                skipped.append(f"{meta_path.name} (missing model: {model_path.name})")
                continue

            feature_cols = meta.get("feature_cols")
            cat_cols = meta.get("cat_cols") or []
            num_cols = meta.get("num_cols") or []
            best_thresholds = meta.get("best_thresholds")

            if not isinstance(feature_cols, list) or not feature_cols:
                skipped.append(f"{meta_path.name} (missing/invalid feature_cols)")
                continue
            if not isinstance(cat_cols, list):
                cat_cols = []
            if not isinstance(num_cols, list):
                num_cols = []
            if not isinstance(best_thresholds, dict):
                skipped.append(f"{meta_path.name} (missing/invalid best_thresholds)")
                continue

            model = tf.keras.models.load_model(model_path)

            REGISTRY[model_tag] = {
                "model": model,
                "meta": meta,
                "model_path": str(model_path),
                "meta_path": str(meta_path),
            }

        except Exception as e:
            skipped.append(f"{meta_path.name} (error: {e})")

    if not REGISTRY:
        raise RuntimeError("No models loaded. Check .keras filenames match metadata model_tag.")

    if skipped:
        print("Skipped metadata/models:")
        for s in skipped[:50]:
            print(" -", s)

@app.on_event("startup")
def _startup():
    load_registry()

# -----------------------------
# HELPERS
# -----------------------------
def get_model_bundle(horizon: Horizon) -> Dict[str, Any]:
    model_tag = HORIZON_TO_TAG.get(horizon)
    if not model_tag:
        raise HTTPException(status_code=400, detail=f"Invalid horizon: {horizon}")
    if model_tag not in REGISTRY:
        raise HTTPException(status_code=500, detail=f"Model '{model_tag}' not loaded.")
    return {"model_tag": model_tag, **REGISTRY[model_tag]}

def pick_threshold(meta: Dict[str, Any], strategy: str, custom_threshold: Optional[float]) -> float:
    if strategy == "custom":
        if custom_threshold is None:
            raise HTTPException(status_code=400, detail="custom_threshold required when strategy='custom'")
        return float(custom_threshold)

    best_thresholds = meta.get("best_thresholds", {})
    json_key = STRATEGY_KEY_MAP.get(strategy)
    if not json_key:
        raise HTTPException(status_code=400, detail=f"Unknown threshold_strategy: {strategy}")

    if json_key not in best_thresholds:
        raise HTTPException(status_code=400, detail=f"Strategy '{strategy}' not available. Available: {list(best_thresholds.keys())}")

    thr = best_thresholds[json_key].get("threshold")
    if thr is None:
        raise RuntimeError(f"Threshold missing for '{json_key}'")

    return float(thr)

def risk_bucket(prob_unsuppressed: float, threshold: float) -> str:
    # prob is "risk of being UNSUPPRESSED"
    if prob_unsuppressed >= max(0.80, threshold):
        return "High"
    if prob_unsuppressed >= threshold:
        return "Moderate"
    return "Low"

def align_to_schema(
    df_in: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
    num_cols: List[str],
    imputation: Optional[Dict[str, Any]] = None,
) -> tuple[pd.DataFrame, List[str]]:
    X = df_in.copy()

    # drop extra cols early (prevents accidental leakage/garbage)
    X = X[[c for c in X.columns if c in set(feature_cols)]].copy()

    missing = [c for c in feature_cols if c not in X.columns]

    cat_fill = "Unknown"
    num_medians = {}
    if isinstance(imputation, dict):
        cat_fill = imputation.get("cat_fill") or "Unknown"
        num_medians = imputation.get("num_medians_from_train") or {}

    for c in missing:
        if c in cat_cols:
            X[c] = cat_fill
        elif c in num_cols:
            X[c] = float(num_medians.get(c, 0.0))
        else:
            X[c] = cat_fill

    X = X[feature_cols].copy()

    for c in cat_cols:
        X[c] = X[c].fillna(cat_fill).astype(str).replace({"": cat_fill})

    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
        X[c] = X[c].fillna(float(num_medians.get(c, 0.0))).astype(np.float32)

    return X, missing

def to_model_input(X: pd.DataFrame, cat_cols: List[str], num_cols: List[str]) -> Dict[str, tf.Tensor]:
    d: Dict[str, tf.Tensor] = {}
    for c in cat_cols:
        d[c] = tf.convert_to_tensor(X[c].astype(str).to_numpy().reshape(-1, 1), dtype=tf.string)
    for c in num_cols:
        d[c] = tf.convert_to_tensor(X[c].to_numpy().astype(np.float32).reshape(-1, 1), dtype=tf.float32)
    return d

# -----------------------------
# ENDPOINTS
# -----------------------------
@app.get("/health")
def health(_=Depends(require_api_key)):
    return {"status": "ok", "models_loaded": len(REGISTRY)}

@app.get("/models")
def models(_=Depends(require_api_key)):
    return {
        k: {
            "n_features": len(v["meta"].get("feature_cols", [])),
            "n_cat": len(v["meta"].get("cat_cols", [])),
            "n_num": len(v["meta"].get("num_cols", [])),
            "best_threshold_keys": list(v["meta"].get("best_thresholds", {}).keys()),
        }
        for k, v in REGISTRY.items()
    }

@app.get("/schema/{horizon}")
def schema(horizon: Horizon, _=Depends(require_api_key)):
    bundle = get_model_bundle(horizon)
    meta = bundle["meta"]
    return {
        "model_tag": bundle["model_tag"],
        "feature_cols": meta.get("feature_cols", []),
        "cat_cols": meta.get("cat_cols", []),
        "num_cols": meta.get("num_cols", []),
        "imputation": meta.get("imputation", {}),
    }

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest, _=Depends(require_api_key)):
    bundle = get_model_bundle(req.horizon)
    model = bundle["model"]
    meta = bundle["meta"]
    model_tag = bundle["model_tag"]

    feature_cols = meta["feature_cols"]
    cat_cols = meta.get("cat_cols", [])
    num_cols = meta.get("num_cols", [])
    imputation = meta.get("imputation", {})

    thr = pick_threshold(meta, req.threshold_strategy, req.custom_threshold)

    df = pd.DataFrame([req.features])
    X, missing = align_to_schema(df, feature_cols, cat_cols, num_cols, imputation=imputation)
    X_input = to_model_input(X, cat_cols, num_cols)

    prob = float(model.predict(X_input, verbose=0).ravel()[0])
    pred = int(prob >= thr)

    return PredictResponse(
        model_tag=model_tag,
        horizon=req.horizon,
        threshold_strategy=req.threshold_strategy,
        threshold_used=float(thr),
        probability=prob,
        predicted_class=pred,
        risk_category=risk_bucket(prob, thr),
        missing_features_filled=", ".join(missing) if missing else "",
    )

@app.post("/predict_batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest, _=Depends(require_api_key)):
    bundle = get_model_bundle(req.horizon)
    model = bundle["model"]
    meta = bundle["meta"]
    model_tag = bundle["model_tag"]

    feature_cols = meta["feature_cols"]
    cat_cols = meta.get("cat_cols", [])
    num_cols = meta.get("num_cols", [])
    imputation = meta.get("imputation", {})

    thr = pick_threshold(meta, req.threshold_strategy, req.custom_threshold)

    df = pd.DataFrame(req.rows)
    X, missing = align_to_schema(df, feature_cols, cat_cols, num_cols, imputation=imputation)
    X_input = to_model_input(X, cat_cols, num_cols)

    probs = model.predict(X_input, verbose=0).ravel()
    preds = (probs >= thr).astype(int)

    out = df.copy()
    out["pred_prob_unsuppressed"] = probs.astype(float)
    out["pred_class"] = preds.astype(int)
    out["used_threshold"] = float(thr)

    return BatchPredictResponse(
        model_tag=model_tag,
        horizon=req.horizon,
        threshold_strategy=req.threshold_strategy,
        threshold_used=float(thr),
        missing_features_filled=", ".join(missing) if missing else "",
        results=out.to_dict(orient="records"),
    )
