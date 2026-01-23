# ===========================
# BATCH 1/5
# ===========================
# ============================================================
# app.py — HIV Viral Suppression Risk (Chain Models)
# UPGRADED (Jan 2026):
# 1) Fixes T0 naming mismatch (_Y0 vs _T0) so baseline features are captured correctly
# 2) Download entries (retraining CSV) = ADMIN ONLY
# 3) Agentic AI tools = PREMIUM ONLY
# 4) Hide internal model names; show friendly aliases only
# 5) Remove "DeepANN" from title and remove "(executive-friendly)"
# 6) Disclaimer reframed as research + decision-support, not standalone clinical
# 7) UI polish for desktop + mobile friendliness (Flutter WebView-ready)
# 8) FIXED: Single, consistent Premium/Admin gating (no duplicate access blocks)
# 9) NEW: Results dashboard for better appeal
# ============================================================

import json
import math
import io
import hashlib
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import requests


# ============================================================
# Access control (Premium + Admin) — SINGLE SOURCE OF TRUTH
# Secrets expected:
#   PREMIUM_ACCESS_KEY = "...."
#   ADMIN_ACCESS_KEY   = "...."
# ============================================================

def _secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""


def is_premium() -> bool:
    required = _secret("PREMIUM_ACCESS_KEY")
    if not required:
        return True  # if not configured, don't block (dev-friendly)
    return bool(st.session_state.get("premium_unlocked", False))


def is_admin() -> bool:
    required = _secret("ADMIN_ACCESS_KEY")
    if not required:
        return True
    return bool(st.session_state.get("admin_unlocked", False))


def render_access_sidebar():
    with st.sidebar:
        st.markdown("### Access")
        st.caption("Admin controls downloads. Premium unlocks Agentic AI tools.")

        # --- Premium unlock ---
        required_premium = _secret("PREMIUM_ACCESS_KEY")
        if required_premium:
            premium_input = st.text_input(
                "Premium access code",
                type="password",
                key="premium_input",
                placeholder="Enter premium code",
            )
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("Unlock Premium", key="btn_unlock_premium"):
                    if premium_input.strip() == required_premium:
                        st.session_state["premium_unlocked"] = True
                        st.success("Premium unlocked")
                    else:
                        st.session_state["premium_unlocked"] = False
                        st.error("Invalid premium code.")
            with c2:
                if st.button("Lock Premium", key="btn_lock_premium"):
                    st.session_state["premium_unlocked"] = False
                    st.info("Premium locked.")

        else:
            st.caption("Premium key not configured in secrets (premium is open).")
            st.session_state["premium_unlocked"] = True

        st.divider()

        # --- Admin unlock ---
        required_admin = _secret("ADMIN_ACCESS_KEY")
        if required_admin:
            admin_input = st.text_input(
                "Admin access code",
                type="password",
                key="admin_input",
                placeholder="Enter admin code",
            )
            c3, c4 = st.columns([1, 1])
            with c3:
                if st.button("Unlock Admin", key="btn_unlock_admin"):
                    if admin_input.strip() == required_admin:
                        st.session_state["admin_unlocked"] = True
                        st.success("Admin unlocked")
                    else:
                        st.session_state["admin_unlocked"] = False
                        st.error("Invalid admin code.")
            with c4:
                if st.button("Lock Admin", key="btn_lock_admin"):
                    st.session_state["admin_unlocked"] = False
                    st.info("Admin locked.")
        else:
            st.caption("Admin key not configured in secrets (admin is open).")
            st.session_state["admin_unlocked"] = True

        st.divider()
        st.markdown("### Status")
        st.write({"premium": is_premium(), "admin": is_admin()})


# -------------------------
# PATHS
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"


# -------------------------
# DROPDOWN OPTIONS
# -------------------------
OPTIONS = {
    "who_stage": [1, 2, 3, 4],
    "suppressed_lt1000": [0, 1],  # 0=No, 1=Yes
    "functional_status": ["Ambulatory", "Bedridden", "Working"],
    "regimen_line": ["1st-line", "2nd-line"],
    "regimen_type": ["AZT/3TC/NVP", "TDF/3TC/DTG", "TDF/3TC/EFV", "AZT/3TC/LPV/r"],
    "tb_status": ["History of TB", "Active TB", "No TB"],
    "gender": ["Male", "Female"],
    # NOTE: stateProvince and facilityName are OPEN TEXT
}


# -------------------------
# MODEL ALIASES (hide internal keys)
# -------------------------
MODEL_ALIAS = {
    "DeepANN_T0_to_Y1": "Baseline → Year 1 Risk Model (v1)",
    "DeepANN_T0Y1_to_Y2": "Baseline+Y1 → Year 2 Risk Model (v1)",
    "DeepANN_T0Y1Y2_to_Y3": "Baseline+Y1+Y2 → Year 3 Risk Model (v1)",
    "DeepANN_T0Y1Y2Y3_to_Y4": "Baseline+Y1+Y2+Y3 → Year 4 Risk Model (v1)",
    "DeepANN_T0Y1Y2Y3Y4_to_Y5": "Baseline+Y1–Y4 → Year 5 Risk Model (v1)",
}

# ===========================
# BATCH 2/5
# ===========================
# -------------------------
# VARIABLE DEFINITIONS (tooltips)
# -------------------------
VAR_HELP = {
    "age": "Client age in completed years (0–120).",
    "age_baseline": "Baseline age in completed years recorded at Year 1 (if available in the model schema).",
    "cd4": "CD4 cell count (cells/mm³).",
    "viral_load": "Viral load in copies/mL (must be ≥ 0). log10(VL) is computed automatically when VL > 0.",
    "log10_vl": "Auto-calculated as log10(viral_load). Not entered manually.",
    "weight": "Weight in kilograms (kg).",
    "missed_appointments": "Number of missed appointments within the assessment period.",
    "days_late": "Total number of days late for visits/pickups within the assessment period.",
    "days_in_period": "Assessment window length used for refill adherence (e.g., 30/60/90 days).",
    "days_covered": "Number of days the client had ART available within the assessment period (from refill records).",
    "pharmacy_refill_adherence_pct": "Auto-calculated as (Days covered ÷ Days in period) × 100, capped to 0–100%.",
    "adherence_prop": "Auto-calculated as pharmacy_refill_adherence_pct ÷ 100 (range 0–1).",
    "gender": "Sex recorded at enrollment.",
    "functional_status": "Client functional status at last assessment.",
    "regimen_line": "ART regimen line (1st-line or 2nd-line).",
    "regimen_type": "ART regimen category/type.",
    "tb_status": "TB status category.",
    "who_stage": "WHO clinical stage (1–4).",
    "stateProvince": "State/Province where the client receives care (open entry).",
    "facilityName": "Facility where the client receives care (open entry).",
    "suppressed_lt1000": "Whether viral load is suppressed below 1000 copies/mL (0=No, 1=Yes).",
}


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="HIV Viral Suppression Predictor", layout="wide")

st.markdown(
    """
    <style>
      .stApp { background: #0b0f14; color: #e5e7eb; }
      .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

      .title-card {
        background: #0f172a;
        padding: 18px 22px;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.35);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 14px;
      }
      .small-note { color: #cbd5e1; font-size: 0.92rem; }

      .metric-card {
        background: #0f172a;
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
      }

      div[data-baseweb="tab-list"]{
        gap: 10px;
        background: transparent;
        border-bottom: 1px solid rgba(255,255,255,0.10);
        padding-bottom: 10px;
      }

      button[role="tab"]{
        border-radius: 12px !important;
        padding: 10px 14px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
        color: #e5e7eb !important;
        font-weight: 700 !important;
        transition: transform .12s ease, background-color .18s ease, border-color .18s ease, filter .18s ease;
      }

      button[role="tab"]:nth-child(1){ background: #111827 !important; }
      button[role="tab"]:nth-child(2){ background: #0b1220 !important; }

      button[role="tab"]:hover{
        transform: translateY(-1px);
        border-color: rgba(255,255,255,0.25) !important;
        filter: brightness(1.15);
      }

      button[role="tab"][aria-selected="true"]{
        background: #1f2937 !important;
        border-color: rgba(59,130,246,0.85) !important;
        box-shadow: 0 0 0 2px rgba(59,130,246,0.25);
      }

      .stTextInput input, .stNumberInput input {
        background: #0b1220 !important;
        color: #e5e7eb !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
      }

      div[data-baseweb="select"] > div {
        background: #0b1220 !important;
        color: #e5e7eb !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
      }

      details summary { color: #e5e7eb !important; }
      .stCaption { color: #9ca3af !important; }
      code { color: #93c5fd !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="title-card">
      <h2 style="margin:0;">HIV Viral Suppression Risk</h2>
      <div class="small-note">
        Single patient form + Batch CSV scoring • Auto-selects the correct model based on available timepoints • Uses model threshold from metadata
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="metric-card" style="margin-bottom:14px;">
      <b>Disclaimer</b><br/>
      <span class="small-note">
        This tool is designed for <b>research</b> and <b>program decision-support</b>. It can help summarize patterns and estimate risk using the provided inputs.
        It is <b>not</b> a standalone clinical decision system and must <b>not</b> replace clinical judgement, laboratory confirmation, or local guidelines.
        Always interpret outputs within routine program workflows and professional oversight.
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# One access sidebar only (fixed)
render_access_sidebar()


# ============================================================
# GOOGLE SHEETS LOGGING (Webhook via Apps Script)
# ============================================================

def _now_iso():
    return dt.datetime.now().isoformat(timespec="seconds")

def flatten_for_sheet(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, (int, float, str, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out

def make_entry_id(payload: dict) -> str:
    raw = json.dumps(flatten_for_sheet(payload), sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]

def sheets_enabled() -> bool:
    return bool(_secret("GSHEETS_WEBHOOK_URL"))

def append_to_gsheet(payload: dict) -> (bool, str):
    url = _secret("GSHEETS_WEBHOOK_URL")
    if not url:
        return False, "GSHEETS_WEBHOOK_URL not configured in secrets."

    try:
        r = requests.post(url, json=payload, timeout=12)
        r.raise_for_status()
        return True, "Saved to Google Sheet."
    except Exception as e:
        return False, f"Sheet save failed: {e}"

# ===========================
# BATCH 3/5
# ===========================
# ============================================================
# Load available models
# ============================================================

def list_available_models(models_dir: Path):
    out = {}
    if not models_dir.exists():
        return out

    for meta_file in models_dir.glob("*_metadata.json"):
        key = meta_file.name.replace("_metadata.json", "")
        keras_file = models_dir / f"{key}.keras"
        if keras_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                out[key] = {"model_path": keras_file, "meta": meta}
            except Exception:
                pass
    return out


available = list_available_models(MODELS_DIR)
available_keys = set(available.keys())

if not available:
    st.error("No models found in models/. Put your .keras and *_metadata.json files inside the models/ folder.")
    st.stop()


# ============================================================
# Helpers (model selection, schema alignment, prediction)
# ============================================================

def extract_bestf1_threshold(meta: dict, fallback: float = 0.5) -> float:
    try:
        return float(meta.get("best_thresholds", {}).get("F1", {}).get("threshold", fallback))
    except Exception:
        return fallback


@st.cache_resource
def load_model_cached(path: str):
    return tf.keras.models.load_model(path)


def detect_max_year_from_cols(cols) -> int:
    max_year = -1
    if any(str(c).endswith("_T0") for c in cols):
        max_year = max(max_year, 0)
    for y in [0, 1, 2, 3, 4]:
        if any(str(c).endswith(f"_Y{y}") for c in cols):
            max_year = max(max_year, y)
    return max(0, max_year)


def choose_model_key_by_year(max_year: int, keys: set):
    if max_year >= 4 and "DeepANN_T0Y1Y2Y3Y4_to_Y5" in keys:
        return "DeepANN_T0Y1Y2Y3Y4_to_Y5"
    if max_year == 3 and "DeepANN_T0Y1Y2Y3_to_Y4" in keys:
        return "DeepANN_T0Y1Y2Y3_to_Y4"
    if max_year == 2 and "DeepANN_T0Y1Y2_to_Y3" in keys:
        return "DeepANN_T0Y1Y2_to_Y3"
    if max_year == 1 and "DeepANN_T0Y1_to_Y2" in keys:
        return "DeepANN_T0Y1_to_Y2"
    if max_year == 0 and "DeepANN_T0_to_Y1" in keys:
        return "DeepANN_T0_to_Y1"
    return None


def choose_model_key_from_df(df: pd.DataFrame, keys: set):
    max_year = detect_max_year_from_cols(df.columns)
    return choose_model_key_by_year(max_year, keys)


def model_alias(key: str) -> str:
    return MODEL_ALIAS.get(key, "HIV Risk Model")


def align_to_schema(df_in: pd.DataFrame, feature_cols: list, cat_cols: list, num_cols: list):
    X = df_in.copy()
    missing = [c for c in feature_cols if c not in X.columns]

    for c in missing:
        if c in cat_cols:
            X[c] = "Unknown"
        elif c in num_cols:
            X[c] = 0.0
        else:
            X[c] = "Unknown"

    X = X[feature_cols].copy()

    for c in cat_cols:
        X[c] = X[c].fillna("Unknown").astype(str)

    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0).astype(np.float32)

    return X, missing


def to_model_input(X: pd.DataFrame, cat_cols: list, num_cols: list):
    d = {}
    for c in cat_cols:
        arr = X[c].fillna("Unknown").astype(str).to_numpy().reshape(-1, 1)
        d[c] = tf.convert_to_tensor(arr, dtype=tf.string)

    for c in num_cols:
        arr = pd.to_numeric(X[c], errors="coerce").fillna(0.0).to_numpy().astype(np.float32).reshape(-1, 1)
        d[c] = tf.convert_to_tensor(arr, dtype=tf.float32)

    return d


def predict_df(df_in: pd.DataFrame, model_info: dict):
    meta = model_info["meta"]
    model = load_model_cached(str(model_info["model_path"]))

    feature_cols = meta["feature_cols"]
    cat_cols = meta["cat_cols"]
    num_cols = meta["num_cols"]
    thr = extract_bestf1_threshold(meta, 0.5)

    X, missing = align_to_schema(df_in, feature_cols, cat_cols, num_cols)
    X_input = to_model_input(X, cat_cols, num_cols)

    prob = model.predict(X_input, verbose=0).ravel()
    pred = (prob >= thr).astype(int)

    out = df_in.copy()
    out["pred_prob_unsuppressed"] = prob
    out["pred_class"] = pred
    out["used_threshold"] = thr
    out["missing_features_filled"] = ", ".join(missing) if missing else ""
    return out


def add_if_in_schema(row: dict, colname: str, value, schema_cols: set):
    if colname in schema_cols:
        row[colname] = value


# ============================================================
# Baseline timepoint handling (T0 vs Y0)
# ============================================================

def time_suffix_for_schema(schema_cols: set, timepoint: int) -> str:
    if timepoint == 0:
        has_t0 = any(str(c).endswith("_T0") for c in schema_cols)
        return "T0" if has_t0 else "Y0"
    return f"Y{timepoint}"

# ===========================
# BATCH 4/5
# ===========================
def col(schema_cols: set, base: str, timepoint: int) -> str:
    suf = time_suffix_for_schema(schema_cols, timepoint)
    return f"{base}_{suf}"


# ============================================================
# Agentic AI Layer (Validation → Explanation → Audit)
# PREMIUM ONLY (gated in UI)
# ============================================================

def agent_validate_row(row: dict, schema_cols: set, max_year: int) -> dict:
    errors, warnings, fixes = [], [], []

    for y in range(0, max_year + 1):
        vl_key = col(schema_cols, "viral_load", y)
        log_key = col(schema_cols, "log10_vl", y)
        pct_key = col(schema_cols, "pharmacy_refill_adherence_pct", y)
        prop_key = col(schema_cols, "adherence_prop", y)

        if vl_key in row:
            vl = float(row.get(vl_key, 0.0))
            if vl < 0:
                errors.append(f"{vl_key}: cannot be negative.")
            if vl == 0:
                warnings.append(f"{vl_key}: VL is 0 → log10_vl will be 0.000 (check if VL is missing).")

        if (vl_key in row) and (log_key in row):
            vl = float(row.get(vl_key, 0.0))
            logvl = float(row.get(log_key, 0.0))
            expected = round(math.log10(vl), 3) if vl > 0 else 0.0
            if abs(logvl - expected) > 0.02:
                warnings.append(f"{log_key}: mismatch with log10({vl_key}). Using computed value.")
                row[log_key] = expected
                fixes.append(f"Set {log_key}={expected} from {vl_key}.")

        if pct_key in row:
            pct = float(row.get(pct_key, 0.0))
            if pct < 0 or pct > 100:
                warnings.append(f"{pct_key}: out of range; capped to 0–100.")
                pct = max(0.0, min(100.0, pct))
                row[pct_key] = round(pct, 2)
                fixes.append(f"Capped {pct_key} to {row[pct_key]}.")

        if (pct_key in row) and (prop_key in row):
            pct = float(row.get(pct_key, 0.0))
            prop = float(row.get(prop_key, 0.0))
            expected_prop = round(pct / 100.0, 4)
            if abs(prop - expected_prop) > 0.01:
                warnings.append(f"{prop_key}: mismatch with {pct_key}/100. Using computed value.")
                row[prop_key] = expected_prop
                fixes.append(f"Set {prop_key}={expected_prop} from {pct_key}.")

        if prop_key in row:
            prop = float(row.get(prop_key, 0.0))
            if prop < 0 or prop > 1:
                warnings.append(f"{prop_key}: out of range; capped to 0–1.")
                prop = max(0.0, min(1.0, prop))
                row[prop_key] = round(prop, 4)
                fixes.append(f"Capped {prop_key} to {row[prop_key]}.")

    ok = len(errors) == 0
    return {"ok": ok, "errors": errors, "warnings": warnings, "fixes": fixes, "row": row}


def agent_template_explanation(row: dict, schema_cols: set, prob_unsupp: float, pred_class: int, thr: float, max_year: int) -> str:
    y = max_year
    vl = row.get(col(schema_cols, "viral_load", y), None)
    logvl = row.get(col(schema_cols, "log10_vl", y), None)
    pct = row.get(col(schema_cols, "pharmacy_refill_adherence_pct", y), None)
    prop = row.get(col(schema_cols, "adherence_prop", y), None)
    missed = row.get(col(schema_cols, "missed_appointments", y), None)
    late = row.get(col(schema_cols, "days_late", y), None)
    cd4 = row.get(col(schema_cols, "cd4", y), None)

    risk_label = "Higher risk of non-suppression" if pred_class == 1 else "Lower risk of non-suppression"

    lines = []
    lines.append(f"**Summary:** {risk_label}.")
    lines.append(f"Model probability (unsuppressed risk) = **{prob_unsupp:.3f}** (threshold **{thr:.3f}**).")

    lines.append("\n**Key signals from the latest timepoint (program interpretation):**")
    if vl is not None:
        lines.append(f"- Viral load: **{vl}** (log10: **{logvl}**). Higher values generally increase risk.")
    if pct is not None:
        lines.append(f"- Refill adherence: **{pct}%** (prop: **{prop}**). Lower adherence generally increases risk.")
    if missed is not None:
        lines.append(f"- Missed appointments: **{missed}**. More missed visits may increase risk.")
    if late is not None:
        lines.append(f"- Days late: **{late}**. Frequent delays can signal gaps in continuity.")
    if cd4 is not None:
        lines.append(f"- CD4: **{cd4}**. Interpretation depends on clinical context and timing.")

    lines.append("\n**Suggested next program actions (non-clinical):**")
    if pred_class == 1:
        lines.append("- Prioritize adherence support and routine follow-up in the program workflow.")
        lines.append("- Verify data completeness (VL, refill period definition, missed visit tracking).")
        lines.append("- Consider a targeted review per your SOP (no treatment recommendations here).")
    else:
        lines.append("- Continue routine follow-up per SOP.")
        lines.append("- Maintain refill continuity and timely visit monitoring.")

    lines.append("\n*Decision-support only; not a diagnosis or treatment recommendation.*")
    return "\n".join(lines)

# ===========================
# BATCH 5/5
# ===========================
def llm_enabled() -> bool:
    return bool(_secret("LLM_API_KEY")) and bool(_secret("LLM_BASE_URL")) and bool(_secret("LLM_MODEL"))


def call_llm_narrative(prompt: str) -> str:
    api_key = _secret("LLM_API_KEY")
    base_url = _secret("LLM_BASE_URL").rstrip("/")
    model = _secret("LLM_MODEL")

    if not api_key or not base_url or not model:
        return ""

    if base_url.endswith("/v1"):
        url = f"{base_url}/chat/completions"
    else:
        url = f"{base_url}/v1/chat/completions"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an assistant embedded in a health program decision-support tool. "
                    "Do NOT provide diagnosis, treatment, dosing, or regimen advice. "
                    "Keep outputs programmatic (data quality checks, follow-up prioritization, interpretation). "
                    "Assume all data is de-identified. Keep it concise and clear."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=10)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


def agent_explain(row: dict, schema_cols: set, prob_unsupp: float, pred_class: int, thr: float, max_year: int, use_llm: bool) -> str:
    base = agent_template_explanation(row, schema_cols, prob_unsupp, pred_class, thr, max_year)
    if not use_llm:
        return base
    if not llm_enabled():
        return base + "\n\n*(LLM not configured; showing standard explanation.)*"

    prompt = (
        "Rewrite the following explanation to be easier for a program manager to read. "
        "Keep it non-clinical, avoid diagnosis/treatment, and preserve the key numbers.\n\n"
        f"{base}"
    )
    improved = call_llm_narrative(prompt)
    return improved if improved else base


def agent_build_audit_record(row: dict, schema_cols: set, chosen_key: str, prob: float, pred: int, thr: float, max_year: int) -> dict:
    y = max_year
    return {
        "timestamp": _now_iso(),
        "model_alias": model_alias(chosen_key),
        "max_timepoint": int(max_year),
        "pred_prob_unsuppressed": float(prob),
        "pred_class": int(pred),
        "threshold": float(thr),
        "viral_load_latest": row.get(col(schema_cols, "viral_load", y), None),
        "log10_vl_latest": row.get(col(schema_cols, "log10_vl", y), None),
        "refill_pct_latest": row.get(col(schema_cols, "pharmacy_refill_adherence_pct", y), None),
        "adherence_prop_latest": row.get(col(schema_cols, "adherence_prop", y), None),
    }


# ============================================================
# Tabs
# ============================================================
tab1, tab2 = st.tabs(["🧍 Single patient form", "📤 Upload CSV (batch)"])


def render_results_dashboard(prob: float, pred: int, thr: float, model_name: str):
    """Pretty dashboard view for results."""
    risk_label = "At-risk (likely NOT suppressed)" if pred == 1 else "Likely suppressed"
    risk_badge = "HIGH RISK" if pred == 1 else "LOW RISK"

    st.markdown("## Results dashboard")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Unsuppressed risk (prob.)", f"{prob:.3f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Threshold used", f"{thr:.3f}")
        st.markdown("</div>", unsafe_allow_html=True)

    with c3:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Prediction", risk_badge)
        st.markdown("</div>", unsafe_allow_html=True)

    with c4:
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        st.metric("Model", model_name)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div class='metric-card' style='margin-top:10px;'>", unsafe_allow_html=True)
    st.markdown(f"**Interpretation:** {risk_label}")
    st.caption("Risk bar shows the predicted probability (0 → 1).")
    st.progress(min(max(prob, 0.0), 1.0))
    st.markdown("</div>", unsafe_allow_html=True)


# ============================================================
# Single patient form (UNCHANGED core logic; UPDATED result UI + gated tools)
# ============================================================
with tab1:
    st.subheader("Single patient form")
    st.markdown(
        "<div class='small-note'>Select the latest timepoint you have. The app will auto-pick the correct model. State + facility are open entry.</div>",
        unsafe_allow_html=True,
    )

    max_year = st.selectbox(
        "Data available up to which timepoint?",
        options=[0, 1, 2, 3, 4],
        index=1,
        key="sp_max_year",
        help="Choose how many timepoints of inputs you want to provide (Baseline=0, Y1..Y4). The app selects the matching model.",
    )

    chosen_key = choose_model_key_by_year(max_year, available_keys)
    if not chosen_key:
        st.error("No matching model found for the selected timepoints. Confirm your models exist in the models/ folder.")
        st.stop()

    meta = available[chosen_key]["meta"]
    schema_cols = set(meta["feature_cols"])

    st.success(f"Model selected: {model_alias(chosen_key)}")

    # -------------------------
    # Build input row
    # -------------------------
    with st.container():
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
        row = {}

        for y in range(0, max_year + 1):
            suf = time_suffix_for_schema(schema_cols, y)
            title = "Baseline (T0) inputs" if y == 0 else f"Year {y} inputs"

            with st.expander(title, expanded=(y in [0, 1])):
                st.markdown("#### Clinical / adherence metrics")

                n1, n2, n3 = st.columns(3)
                with n1:
                    age = st.number_input(f"age_{suf}", 0, 120, 35, 1, key=f"sp_age_{suf}", help=VAR_HELP["age"])
                    add_if_in_schema(row, f"age_{suf}", float(age), schema_cols)

                with n2:
                    cd4 = st.number_input(f"cd4_{suf}", 0, 5000, 350, 10, key=f"sp_cd4_{suf}", help=VAR_HELP["cd4"])
                    add_if_in_schema(row, f"cd4_{suf}", float(cd4), schema_cols)

                with n3:
                    vl = st.number_input(f"viral_load_{suf}", min_value=0.0, value=1000.0, step=50.0, key=f"sp_vl_{suf}", help=VAR_HELP["viral_load"])
                    add_if_in_schema(row, f"viral_load_{suf}", float(vl), schema_cols)

                n4, n5, n6 = st.columns(3)
                with n4:
                    logvl = round(math.log10(vl), 3) if vl > 0 else 0.0
                    st.metric(label=f"log10_vl_{suf} (auto)", value=f"{logvl:.3f}", help=VAR_HELP["log10_vl"])
                    add_if_in_schema(row, f"log10_vl_{suf}", float(logvl), schema_cols)

                with n5:
                    wt = st.number_input(f"weight_{suf}", min_value=0.0, value=60.0, step=0.5, key=f"sp_wt_{suf}", help=VAR_HELP["weight"])
                    add_if_in_schema(row, f"weight_{suf}", float(wt), schema_cols)

                with n6:
                    missed = st.number_input(f"missed_appointments_{suf}", min_value=0, value=0, step=1, key=f"sp_missed_{suf}", help=VAR_HELP["missed_appointments"])
                    add_if_in_schema(row, f"missed_appointments_{suf}", float(missed), schema_cols)

                st.markdown("#### Pharmacy refill adherence calculator")
                cA, cB, cC = st.columns(3)
                with cA:
                    days_in_period = st.number_input(f"Days in assessment period ({suf})", 1, 365, 30, 1, key=f"sp_days_in_period_{suf}", help=VAR_HELP["days_in_period"])
                with cB:
                    days_covered = st.number_input(f"Days covered by refills ({suf})", 0, 365, 0, 1, key=f"sp_days_covered_{suf}", help=VAR_HELP["days_covered"])

                pharmacy_refill_adherence_pct = (days_covered / days_in_period) * 100.0 if days_in_period > 0 else 0.0
                pharmacy_refill_adherence_pct = round(max(0.0, min(100.0, pharmacy_refill_adherence_pct)), 2)
                adherence_prop = round(pharmacy_refill_adherence_pct / 100.0, 4)

                with cC:
                    st.metric(label=f"pharmacy_refill_adherence_pct_{suf} (auto)", value=f"{pharmacy_refill_adherence_pct:.2f}%", help=VAR_HELP["pharmacy_refill_adherence_pct"])
                    st.metric(label=f"adherence_prop_{suf} (auto)", value=f"{adherence_prop:.4f}", help=VAR_HELP["adherence_prop"])

                add_if_in_schema(row, f"pharmacy_refill_adherence_pct_{suf}", float(pharmacy_refill_adherence_pct), schema_cols)
                add_if_in_schema(row, f"adherence_prop_{suf}", float(adherence_prop), schema_cols)

                late = st.number_input(f"days_late_{suf}", 0, value=0, step=1, key=f"sp_late_{suf}", help=VAR_HELP["days_late"])
                add_if_in_schema(row, f"days_late_{suf}", float(late), schema_cols)

                if "age_baseline_Y1" in schema_cols and y == 0:
                    base_age = st.number_input("age_baseline_Y1", 0, 120, 35, 1, key="sp_base_age_baseline", help=VAR_HELP["age_baseline"])
                    add_if_in_schema(row, "age_baseline_Y1", float(base_age), schema_cols)

                st.markdown("#### Program / clinical context")

                c1, c2, c3 = st.columns(3)
                with c1:
                    gender = st.selectbox(f"gender_{suf}", OPTIONS["gender"], key=f"sp_gender_{suf}", help=VAR_HELP["gender"])
                    add_if_in_schema(row, f"gender_{suf}", gender, schema_cols)
                with c2:
                    func = st.selectbox(f"functional_status_{suf}", OPTIONS["functional_status"], key=f"sp_func_{suf}", help=VAR_HELP["functional_status"])
                    add_if_in_schema(row, f"functional_status_{suf}", func, schema_cols)
                with c3:
                    regimen_line = st.selectbox(f"regimen_line_{suf}", OPTIONS["regimen_line"], key=f"sp_line_{suf}", help=VAR_HELP["regimen_line"])
                    add_if_in_schema(row, f"regimen_line_{suf}", regimen_line, schema_cols)

                c4, c5, c6 = st.columns(3)
                with c4:
                    regimen_type = st.selectbox(f"regimen_type_{suf}", OPTIONS["regimen_type"], key=f"sp_type_{suf}", help=VAR_HELP["regimen_type"])
                    add_if_in_schema(row, f"regimen_type_{suf}", regimen_type, schema_cols)
                with c5:
                    tb = st.selectbox(f"tb_status_{suf}", OPTIONS["tb_status"], key=f"sp_tb_{suf}", help=VAR_HELP["tb_status"])
                    add_if_in_schema(row, f"tb_status_{suf}", tb, schema_cols)
                with c6:
                    who = st.selectbox(f"who_stage_{suf}", OPTIONS["who_stage"], key=f"sp_who_{suf}", help=VAR_HELP["who_stage"])
                    add_if_in_schema(row, f"who_stage_{suf}", who, schema_cols)

                c7, c8, c9 = st.columns(3)
                with c7:
                    state = st.text_input(f"stateProvince_{suf}", value="", key=f"sp_state_{suf}", help=VAR_HELP["stateProvince"], placeholder="e.g., Lagos, Rivers, FCT-Abuja...")
                    add_if_in_schema(row, f"stateProvince_{suf}", state.strip(), schema_cols)
                with c8:
                    fac = st.text_input(f"facilityName_{suf}", value="", key=f"sp_fac_{suf}", help=VAR_HELP["facilityName"], placeholder="Enter facility name (free text)")
                    add_if_in_schema(row, f"facilityName_{suf}", fac.strip(), schema_cols)
                with c9:
                    sup = st.selectbox(f"suppressed_lt1000_{suf}", OPTIONS["suppressed_lt1000"], key=f"sp_sup_{suf}", help=VAR_HELP["suppressed_lt1000"])
                    st.caption("Note: 0 = No, 1 = Yes")
                    add_if_in_schema(row, f"suppressed_lt1000_{suf}", int(sup), schema_cols)

        st.markdown("</div>", unsafe_allow_html=True)

    # Predict button
    if st.button("Predict (Single Patient)", type="primary", key="sp_predict_btn"):
        df1 = pd.DataFrame([row])
        res = predict_df(df1, available[chosen_key])

        prob = float(res.loc[0, "pred_prob_unsuppressed"])
        pred = int(res.loc[0, "pred_class"])
        thr = float(res.loc[0, "used_threshold"])

        st.session_state["last_row"] = row
        st.session_state["last_res"] = res
        st.session_state["last_prob"] = prob
        st.session_state["last_pred"] = pred
        st.session_state["last_thr"] = thr
        st.session_state["last_model_key"] = chosen_key
        st.session_state["last_max_year"] = int(max_year)
        st.session_state["last_schema_cols"] = schema_cols

        st.session_state.pop("agent_validation", None)
        st.session_state.pop("agent_explanation", None)
        st.session_state.pop("last_audit", None)

        record = flatten_for_sheet({
            **row,
            "entry_id": "",
            "timestamp": _now_iso(),
            "model_alias": model_alias(chosen_key),
            "max_timepoint": int(max_year),
            "pred_prob_unsuppressed": prob,
            "pred_class": pred,
            "threshold_used": thr,
            "missing_features_filled": str(res.loc[0, "missing_features_filled"] or ""),
            "agent_validation_ok": "",
            "agent_validation_errors": "",
            "agent_validation_warnings": "",
            "agent_validation_fixes": "",
            "agent_explanation_text": "",
            "agent_audit_json": "",
        })
        record["entry_id"] = make_entry_id(record)

        st.session_state.setdefault("retrain_log", [])
        st.session_state["retrain_log"].append(record)
        st.session_state["last_record_index"] = len(st.session_state["retrain_log"]) - 1

        if sheets_enabled():
            ok, msg = append_to_gsheet(record)
            if ok:
                st.success(msg)
            else:
                st.warning(msg + " (You can still download CSV as Admin.)")
        else:
            st.info("Google Sheet logging not configured yet.")

        missing_txt = str(res.loc[0, "missing_features_filled"] or "").strip()
        if missing_txt:
            st.warning("Some model features were not provided in the form and were auto-filled with defaults.")
            st.write("Auto-filled features:", missing_txt)

    # Results UI
    if "last_res" in st.session_state:
        prob = st.session_state["last_prob"]
        pred = st.session_state["last_pred"]
        thr = st.session_state["last_thr"]

        # New dashboard
        render_results_dashboard(prob, pred, thr, model_alias(st.session_state["last_model_key"]))

        # Technical details toggle
        with st.expander("Show technical details (dataframe)"):
            st.dataframe(st.session_state["last_res"])

        # Premium gating
        st.markdown("## Agent tools")

        if not is_premium():
            st.info("Agent tools are available to Premium users only. Use the Premium access code in the sidebar to unlock.")
        else:
            use_llm = st.toggle(
                "Use AI narrative (LLM)",
                value=st.session_state.get("use_llm", False),
                key="use_llm",
                help="Optional. Requires LLM secrets configured. If not configured, the app shows standard explanation.",
            )

            a1, a2, a3 = st.columns(3)

            def update_last_record_and_sheet(patch: dict):
                if "retrain_log" not in st.session_state or "last_record_index" not in st.session_state:
                    return
                idx = st.session_state["last_record_index"]
                st.session_state["retrain_log"][idx].update(flatten_for_sheet(patch))
                if sheets_enabled():
                    append_to_gsheet(st.session_state["retrain_log"][idx])

            with a1:
                if st.button("1) Validation", key="agent_validate_btn"):
                    schema_cols = st.session_state["last_schema_cols"]
                    v = agent_validate_row(st.session_state["last_row"], schema_cols, st.session_state["last_max_year"])
                    st.session_state["agent_validation"] = v

                    patch = {
                        "agent_validation_ok": bool(v["ok"]),
                        "agent_validation_errors": " | ".join(v["errors"]) if v["errors"] else "",
                        "agent_validation_warnings": " | ".join(v["warnings"]) if v["warnings"] else "",
                        "agent_validation_fixes": " | ".join(v["fixes"]) if v["fixes"] else "",
                    }
                    update_last_record_and_sheet(patch)

                    if v["errors"]:
                        st.error("Validation errors found.")
                    elif v["warnings"]:
                        st.warning("Validation completed with warnings.")
                    else:
                        st.success("Validation passed (no errors).")

            with a2:
                if st.button("2) Explanation", key="agent_explain_btn"):
                    schema_cols = st.session_state["last_schema_cols"]
                    v = st.session_state.get("agent_validation")
                    row_for_explain = v["row"] if v else st.session_state["last_row"]

                    explanation = agent_explain(
                        row=row_for_explain,
                        schema_cols=schema_cols,
                        prob_unsupp=st.session_state["last_prob"],
                        pred_class=st.session_state["last_pred"],
                        thr=st.session_state["last_thr"],
                        max_year=st.session_state["last_max_year"],
                        use_llm=use_llm,
                    )
                    st.session_state["agent_explanation"] = explanation
                    update_last_record_and_sheet({"agent_explanation_text": explanation})
                    st.success("Explanation generated.")

            with a3:
                if st.button("3) Audit", key="agent_audit_btn"):
                    schema_cols = st.session_state["last_schema_cols"]
                    v = st.session_state.get("agent_validation")
                    row_for_audit = v["row"] if v else st.session_state["last_row"]

                    audit = agent_build_audit_record(
                        row=row_for_audit,
                        schema_cols=schema_cols,
                        chosen_key=st.session_state["last_model_key"],
                        prob=st.session_state["last_prob"],
                        pred=st.session_state["last_pred"],
                        thr=st.session_state["last_thr"],
                        max_year=st.session_state["last_max_year"],
                    )
                    st.session_state["last_audit"] = audit
                    update_last_record_and_sheet({"agent_audit_json": json.dumps(audit)})
                    st.success("Audit record added.")

            v = st.session_state.get("agent_validation")
            if v:
                if v["errors"]:
                    st.error("**Errors:**\n" + "\n".join([f"- {e}" for e in v["errors"]]))
                if v["warnings"]:
                    st.warning("**Warnings:**\n" + "\n".join([f"- {w}" for w in v["warnings"]]))
                if v["fixes"]:
                    st.info("**Auto-fixes applied:**\n" + "\n".join([f"- {f}" for f in v["fixes"]]))

            explanation = st.session_state.get("agent_explanation")
            if explanation:
                st.markdown("### Explanation")
                st.markdown(explanation)

            if st.session_state.get("last_audit"):
                st.markdown("### Audit record (latest)")
                st.json(st.session_state["last_audit"])

        # ADMIN ONLY downloads
        st.markdown("## Download entries (for retraining)")
        if not is_admin():
            st.caption("Downloads are restricted to Admin only.")
        else:
            if "retrain_log" in st.session_state and len(st.session_state["retrain_log"]) > 0:
                last_entry_df = pd.DataFrame([st.session_state["retrain_log"][-1]])
                st.download_button(
                    "Download last prediction row (CSV)",
                    data=last_entry_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"hiv_last_prediction_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="dl_last_entry_csv",
                )

                all_entries_df = pd.DataFrame(st.session_state["retrain_log"])
                st.download_button(
                    "Download ALL session prediction rows (CSV)",
                    data=all_entries_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"hiv_session_predictions_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key="dl_all_entries_csv",
                )
            else:
                st.caption("No saved entries yet. Run a prediction first.")
    else:
        st.info("Run a prediction first to unlock results and (Premium) agent tools.")


# ============================================================
# Batch scoring (Upload CSV)
# ============================================================
with tab2:
    st.subheader("Batch scoring (Upload CSV)")
    st.markdown(
        "<div class='small-note'>Upload a CSV. The app auto-selects the best model based on timepoint columns present (_T0 or _Y0.._Y4).</div>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch")

    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        st.write("Preview:", df_up.head())

        chosen_key = choose_model_key_from_df(df_up, available_keys)
        if not chosen_key:
            st.error("Could not auto-select a model for this CSV. Ensure it contains *_T0 or *_Y0..*_Y4 columns.")
            st.stop()

        st.success(f"Model selected: {model_alias(chosen_key)}")

        if st.button("Run batch predictions", type="primary", key="batch_predict_btn"):
            res = predict_df(df_up, available[chosen_key])
            st.dataframe(res.head(50))

            out_csv = res.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions CSV",
                data=out_csv,
                file_name=f"predictions_{model_alias(chosen_key).replace(' ', '_')}.csv",
                mime="text/csv",
                key="batch_download_btn",
            )

