import json
import math
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import requests

# -------------------------
# PATHS (always correct)
# -------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# -------------------------
# DROPDOWN OPTIONS (fixed)
# -------------------------
OPTIONS = {
    "who_stage": [1, 2, 3, 4],
    "suppressed_lt1000": [0, 1],  # Note shown in UI: 0=No, 1=Yes
    "functional_status": ["Ambulatory", "Bedridden", "Working"],
    "regimen_line": ["1st-line", "2nd-line"],
    "regimen_type": ["AZT/3TC/NVP", "TDF/3TC/DTG", "TDF/3TC/EFV", "AZT/3TC/LPV/r"],
    "tb_status": ["History of TB", "Active TB", "No TB"],
    "gender": ["Male", "Female"],
    "stateProvince": ["Abuja-FCT", "Kaduna", "Kano", "Lagos", "Oyo", "Rivers"],
    "facilityName": [f"Facility_{i}" for i in range(1, 21)],
}

# -------------------------
# VARIABLE DEFINITIONS (tooltips)
# -------------------------
VAR_HELP = {
    "age": "Client age in completed years (0–120).",
    "age_baseline": "Baseline age in completed years recorded at Year 1.",
    "cd4": "CD4 cell count (cells/mm³).",
    "viral_load": "Most recent viral load result in copies/mL. Must be > 0 to compute log10(VL).",
    "log10_vl": "Automatically calculated as log10(viral_load). Not entered manually.",
    "weight": "Weight in kilograms (kg).",
    "missed_appointments": "Number of missed appointments within the assessment period.",
    "days_late": "Total number of days late for visits/pickups within the assessment period.",
    "pharmacy_refill_adherence_pct": "Calculated as (Days covered ÷ Days in period) × 100, capped to 0–100%.",
    "adherence_prop": "Automatically calculated as pharmacy_refill_adherence_pct ÷ 100 (range 0–1).",
    "days_in_period": "Assessment window length used for refill adherence (commonly 30/60/90 days).",
    "days_covered": "Number of days the client had ART available within the assessment period (from refill records).",
    "gender": "Sex recorded at enrollment.",
    "functional_status": "Client functional status at last assessment.",
    "regimen_line": "ART regimen line (1st-line or 2nd-line).",
    "regimen_type": "ART regimen category/type.",
    "tb_status": "TB status category.",
    "who_stage": "WHO clinical stage (1–4).",
    "stateProvince": "State/Province where the client receives care.",
    "facilityName": "Facility where the client receives care.",
    "suppressed_lt1000": "Whether viral load is suppressed below 1000 copies/mL (0=No, 1=Yes).",
}

# -------------------------
# UI (Black theme + tab hover + varying tab depth)
# -------------------------
st.set_page_config(page_title="HIV Viral Suppression Risk", layout="wide")

st.markdown(
    """
    <style>
      /* ===== App background ===== */
      .stApp { background: #0b0f14; color: #e5e7eb; }
      .block-container { padding-top: 1.2rem; }

      /* ===== Title card ===== */
      .title-card {
        background: #0f172a;
        padding: 18px 22px;
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.35);
        border: 1px solid rgba(255,255,255,0.08);
        margin-bottom: 14px;
      }
      .small-note { color: #cbd5e1; font-size: 0.92rem; }

      /* ===== Cards ===== */
      .metric-card {
        background: #0f172a;
        padding: 14px 16px;
        border-radius: 12px;
        border: 1px solid rgba(255,255,255,0.08);
        box-shadow: 0 6px 18px rgba(0,0,0,0.25);
      }

      /* ===== Tabs styling (Streamlit/BaseWeb) ===== */
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

      /* Varying depth per tab */
      button[role="tab"]:nth-child(1){ background: #111827 !important; } /* deeper */
      button[role="tab"]:nth-child(2){ background: #0b1220 !important; } /* slightly lighter */

      /* Hover effect */
      button[role="tab"]:hover{
        transform: translateY(-1px);
        border-color: rgba(255,255,255,0.25) !important;
        filter: brightness(1.15);
      }

      /* Active tab */
      button[role="tab"][aria-selected="true"]{
        background: #1f2937 !important;
        border-color: rgba(59,130,246,0.85) !important;
        box-shadow: 0 0 0 2px rgba(59,130,246,0.25);
      }

      /* Inputs / selectors dark look */
      .stTextInput input, .stNumberInput input {
        background: #0b1220 !important;
        color: #e5e7eb !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
      }

      /* Selectbox baseweb */
      div[data-baseweb="select"] > div {
        background: #0b1220 !important;
        color: #e5e7eb !important;
        border-radius: 10px !important;
        border: 1px solid rgba(255,255,255,0.10) !important;
      }

      /* Expander */
      details summary { color: #e5e7eb !important; }
      .stCaption { color: #9ca3af !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="title-card">
      <h2 style="margin:0;">HIV Viral Suppression Risk (DeepANN Model)</h2>
      <div class="small-note">
        Single patient form + Batch CSV scoring • Auto-selects the right model based on available year columns • Uses BestF1 threshold from metadata
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Disclaimer
# -------------------------
st.markdown(
    """
    <div class="metric-card" style="margin-bottom:14px;">
      <b>Disclaimer</b><br/>
      <span class="small-note">
        This is a <b>viral suppression risk prediction</b> demo developed using a <b>simulated dataset</b>.
        Outputs are for demonstration, research, and decision-support exploration only and must <b>not</b> be used as a standalone basis for clinical decisions.
        Interpret results alongside clinical guidelines, laboratory results, and professional judgement.
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Load available models
# -------------------------
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
    st.error(
        "No models found in models/. Put your .keras and *_metadata.json files inside the models/ folder (same level as app.py)."
    )
    st.stop()

# -------------------------
# Helpers
# -------------------------
def extract_bestf1_threshold(meta: dict, fallback: float = 0.5) -> float:
    try:
        return float(meta.get("best_thresholds", {}).get("F1", {}).get("threshold", fallback))
    except Exception:
        return fallback


@st.cache_resource
def load_model_cached(path: str):
    return tf.keras.models.load_model(path)


def detect_max_year_from_cols(cols) -> int:
    max_year = 0
    for y in [1, 2, 3, 4]:
        if any(str(c).endswith(f"_Y{y}") for c in cols):
            max_year = y
    return max_year


def choose_model_key_by_year(max_year: int, keys: set):
    if max_year >= 4 and "DeepANN_Y1Y2Y3Y4_to_Y5" in keys:
        return "DeepANN_Y1Y2Y3Y4_to_Y5"
    if max_year == 3 and "DeepANN_Y1Y2Y3_to_Y4" in keys:
        return "DeepANN_Y1Y2Y3_to_Y4"
    if max_year == 2 and "DeepANN_Y1Y2_to_Y3" in keys:
        return "DeepANN_Y1Y2_to_Y3"
    if max_year == 1 and "DeepANN_Y1_to_Y2" in keys:
        return "DeepANN_Y1_to_Y2"
    return None


def choose_model_key_from_df(df: pd.DataFrame, keys: set):
    max_year = detect_max_year_from_cols(df.columns)
    return choose_model_key_by_year(max_year, keys)


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


# =========================
# AGENTIC AI LAYER
# =========================
def agent_validate_row(row: dict, max_year: int) -> dict:
    errors, warnings, fixes = [], [], []

    for y in range(1, max_year + 1):
        vl_key = f"viral_load_Y{y}"
        log_key = f"log10_vl_Y{y}"
        pct_key = f"pharmacy_refill_adherence_pct_Y{y}"
        prop_key = f"adherence_prop_Y{y}"

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


def agent_template_explanation(row: dict, prob_unsupp: float, pred_class: int, thr: float, max_year: int) -> str:
    y = max_year
    vl = row.get(f"viral_load_Y{y}", None)
    logvl = row.get(f"log10_vl_Y{y}", None)
    pct = row.get(f"pharmacy_refill_adherence_pct_Y{y}", None)
    prop = row.get(f"adherence_prop_Y{y}", None)
    missed = row.get(f"missed_appointments_Y{y}", None)
    late = row.get(f"days_late_Y{y}", None)
    cd4 = row.get(f"cd4_Y{y}", None)

    risk_label = "Higher risk of non-suppression" if pred_class == 1 else "Lower risk of non-suppression"
    lines = []
    lines.append(f"**Summary:** {risk_label}.")
    lines.append(f"Model output probability (unsuppressed risk) = **{prob_unsupp:.3f}** using threshold **{thr:.3f}**.")

    lines.append("\n**Key factors from latest year inputs (program interpretation):**")
    if vl is not None:
        lines.append(f"- Viral load (copies/mL): **{vl}** (log10: **{logvl}**) — higher values generally increase risk.")
    if pct is not None:
        lines.append(f"- Pharmacy refill adherence (%): **{pct}%** (prop: **{prop}**) — lower refill adherence generally increases risk.")
    if missed is not None:
        lines.append(f"- Missed appointments: **{missed}** — more missed visits can increase risk.")
    if late is not None:
        lines.append(f"- Days late: **{late}** — frequent delays can signal gaps in continuity.")
    if cd4 is not None:
        lines.append(f"- CD4: **{cd4}** — may correlate with risk depending on context.")

    lines.append("\n**Suggested next program actions (non-clinical):**")
    if pred_class == 1:
        lines.append("- Prioritize adherence support / follow-up for this client in routine program workflow.")
        lines.append("- Verify data completeness (VL value, refill period definition, missed visit counts).")
        lines.append("- Schedule routine follow-up review based on your program SOP.")
    else:
        lines.append("- Continue routine follow-up per program SOP.")
        lines.append("- Maintain refill continuity and timely visit tracking.")

    lines.append("\n*Note: This is decision-support for program workflows; not a diagnosis or treatment recommendation.*")
    return "\n".join(lines)


def llm_enabled() -> bool:
    return bool(st.secrets.get("LLM_API_KEY", "")) and bool(st.secrets.get("LLM_BASE_URL", "")) and bool(st.secrets.get("LLM_MODEL", ""))


def call_llm_narrative(prompt: str) -> str:
    api_key = st.secrets.get("LLM_API_KEY")
    base_url = st.secrets.get("LLM_BASE_URL", "").rstrip("/")
    model = st.secrets.get("LLM_MODEL", "")

    if not api_key or not base_url or not model:
        return ""

    url = f"{base_url}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    system = (
        "You are an assistant embedded in a health program decision-support app. "
        "Do NOT provide diagnosis, treatment, dosing, or regimen advice. "
        "Keep outputs programmatic (data quality, follow-up prioritization, interpretation). "
        "Assume all data is de-identified. Keep it concise and clear."
    )

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=30)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""


def agent_explain(row: dict, prob_unsupp: float, pred_class: int, thr: float, max_year: int, use_llm: bool) -> str:
    base = agent_template_explanation(row, prob_unsupp, pred_class, thr, max_year)
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


def agent_build_audit_record(row: dict, chosen_key: str, prob: float, pred: int, thr: float, max_year: int) -> dict:
    y = max_year
    return {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "model_key": chosen_key,
        "max_year": int(max_year),
        "pred_prob_unsuppressed": float(prob),
        "pred_class": int(pred),
        "threshold": float(thr),
        "viral_load_latest": row.get(f"viral_load_Y{y}", None),
        "log10_vl_latest": row.get(f"log10_vl_Y{y}", None),
        "refill_pct_latest": row.get(f"pharmacy_refill_adherence_pct_Y{y}", None),
        "adherence_prop_latest": row.get(f"adherence_prop_Y{y}", None),
    }


# -------------------------
# Tabs
# -------------------------
tab1, tab2 = st.tabs(["🧍 Single patient form", "📤Upload CSV (batch)"])

with tab1:
    st.subheader("Single patient form (executive-friendly)")
    st.markdown(
        "<div class='small-note'>Select how many years of data you have. The app will pick the right model and use the BestF1 threshold from its metadata.</div>",
        unsafe_allow_html=True,
    )

    max_year = st.selectbox(
        "Data available up to which year?",
        options=[1, 2, 3, 4],
        index=0,
        key="sp_max_year",
        help="Choose how many years of inputs you want to provide (Y1..Y4). The app selects the matching model.",
    )
    chosen_key = choose_model_key_by_year(max_year, available_keys)

    if not chosen_key:
        st.error("No matching model found for the selected years. Confirm your models exist in the models/ folder.")
        st.stop()

    meta = available[chosen_key]["meta"]
    schema_cols = set(meta["feature_cols"])  # what model expects

    st.success(f"Auto-selected model: {chosen_key}")

    with st.container():
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)

        row = {}

        for y in range(1, max_year + 1):
            with st.expander(f"Year {y} inputs", expanded=(y == 1)):
                st.markdown("#### Clinical / adherence metrics")

                n1, n2, n3 = st.columns(3)
                with n1:
                    age = st.number_input(
                        f"age_Y{y}",
                        min_value=0, max_value=120, value=35, step=1,
                        key=f"sp_age_{y}",
                        help=VAR_HELP["age"]
                    )
                    add_if_in_schema(row, f"age_Y{y}", float(age), schema_cols)

                with n2:
                    cd4 = st.number_input(
                        f"cd4_Y{y}",
                        min_value=0, max_value=5000, value=350, step=10,
                        key=f"sp_cd4_{y}",
                        help=VAR_HELP["cd4"]
                    )
                    add_if_in_schema(row, f"cd4_Y{y}", float(cd4), schema_cols)

                with n3:
                    vl = st.number_input(
                        f"viral_load_Y{y}",
                        min_value=0.0, value=1000.0, step=50.0,
                        key=f"sp_vl_{y}",
                        help=VAR_HELP["viral_load"]
                    )
                    add_if_in_schema(row, f"viral_load_Y{y}", float(vl), schema_cols)

                # IMPORTANT FIX:
                # Use st.metric for calculated fields (NOT disabled text_input),
                # because disabled inputs can "freeze" display on Streamlit Cloud.

                n4, n5, n6 = st.columns(3)

                with n4:
                    if vl > 0:
                        logvl = round(math.log10(vl), 3)
                    else:
                        logvl = 0.0
                        st.caption("VL is 0 → log10(VL) set to 0.000 (cannot take log10 of 0).")

                    st.metric(
                        label=f"log10_vl_Y{y} (auto)",
                        value=f"{logvl:.3f}",
                        help=VAR_HELP["log10_vl"]
                    )
                    add_if_in_schema(row, f"log10_vl_Y{y}", float(logvl), schema_cols)

                with n5:
                    wt = st.number_input(
                        f"weight_Y{y}",
                        min_value=0.0, value=60.0, step=0.5,
                        key=f"sp_wt_{y}",
                        help=VAR_HELP["weight"]
                    )
                    add_if_in_schema(row, f"weight_Y{y}", float(wt), schema_cols)

                with n6:
                    missed = st.number_input(
                        f"missed_appointments_Y{y}",
                        min_value=0, value=0, step=1,
                        key=f"sp_missed_{y}",
                        help=VAR_HELP["missed_appointments"]
                    )
                    add_if_in_schema(row, f"missed_appointments_Y{y}", float(missed), schema_cols)

                st.markdown("#### Pharmacy refill adherence calculator")

                cA, cB, cC = st.columns(3)
                with cA:
                    days_in_period = st.number_input(
                        f"Days in assessment period (Y{y})",
                        min_value=1, max_value=365, value=30, step=1,
                        key=f"sp_days_in_period_{y}",
                        help=VAR_HELP["days_in_period"]
                    )

                with cB:
                    days_covered = st.number_input(
                        f"Days covered by refills (Y{y})",
                        min_value=0, max_value=365, value=0, step=1,
                        key=f"sp_days_covered_{y}",
                        help=VAR_HELP["days_covered"]
                    )

                # hard guard
                if days_in_period <= 0:
                    pharmacy_refill_adherence_pct = 0.0
                else:
                    pharmacy_refill_adherence_pct = (days_covered / days_in_period) * 100.0

                pharmacy_refill_adherence_pct = max(0.0, min(100.0, pharmacy_refill_adherence_pct))
                pharmacy_refill_adherence_pct = round(pharmacy_refill_adherence_pct, 2)
                adherence_prop = round(pharmacy_refill_adherence_pct / 100.0, 4)

                with cC:
                    st.metric(
                        label=f"pharmacy_refill_adherence_pct_Y{y} (auto)",
                        value=f"{pharmacy_refill_adherence_pct:.2f}%",
                        help=VAR_HELP["pharmacy_refill_adherence_pct"]
                    )
                    st.metric(
                        label=f"adherence_prop_Y{y} (auto)",
                        value=f"{adherence_prop:.4f}",
                        help=VAR_HELP["adherence_prop"]
                    )

                add_if_in_schema(row, f"pharmacy_refill_adherence_pct_Y{y}", float(pharmacy_refill_adherence_pct), schema_cols)
                add_if_in_schema(row, f"adherence_prop_Y{y}", float(adherence_prop), schema_cols)

                late = st.number_input(
                    f"days_late_Y{y}",
                    min_value=0, value=0, step=1,
                    key=f"sp_late_{y}",
                    help=VAR_HELP["days_late"]
                )
                add_if_in_schema(row, f"days_late_Y{y}", float(late), schema_cols)

                if y == 1:
                    base_age = st.number_input(
                        "age_baseline_Y1",
                        min_value=0, max_value=120, value=35, step=1,
                        key="sp_base_age",
                        help=VAR_HELP["age_baseline"]
                    )
                    add_if_in_schema(row, "age_baseline_Y1", float(base_age), schema_cols)

                st.markdown("#### Program / clinical context")

                c1, c2, c3 = st.columns(3)
                with c1:
                    gender = st.selectbox(
                        f"gender_Y{y}",
                        OPTIONS["gender"],
                        key=f"sp_gender_{y}",
                        help=VAR_HELP["gender"]
                    )
                    add_if_in_schema(row, f"gender_Y{y}", gender, schema_cols)

                with c2:
                    func = st.selectbox(
                        f"functional_status_Y{y}",
                        OPTIONS["functional_status"],
                        key=f"sp_func_{y}",
                        help=VAR_HELP["functional_status"]
                    )
                    add_if_in_schema(row, f"functional_status_Y{y}", func, schema_cols)

                with c3:
                    regimen_line = st.selectbox(
                        f"regimen_line_Y{y}",
                        OPTIONS["regimen_line"],
                        key=f"sp_line_{y}",
                        help=VAR_HELP["regimen_line"]
                    )
                    add_if_in_schema(row, f"regimen_line_Y{y}", regimen_line, schema_cols)

                c4, c5, c6 = st.columns(3)
                with c4:
                    regimen_type = st.selectbox(
                        f"regimen_type_Y{y}",
                        OPTIONS["regimen_type"],
                        key=f"sp_type_{y}",
                        help=VAR_HELP["regimen_type"]
                    )
                    add_if_in_schema(row, f"regimen_type_Y{y}", regimen_type, schema_cols)

                with c5:
                    tb = st.selectbox(
                        f"tb_status_Y{y}",
                        OPTIONS["tb_status"],
                        key=f"sp_tb_{y}",
                        help=VAR_HELP["tb_status"]
                    )
                    add_if_in_schema(row, f"tb_status_Y{y}", tb, schema_cols)

                with c6:
                    who = st.selectbox(
                        f"who_stage_Y{y}",
                        OPTIONS["who_stage"],
                        key=f"sp_who_{y}",
                        help=VAR_HELP["who_stage"]
                    )
                    add_if_in_schema(row, f"who_stage_Y{y}", who, schema_cols)

                c7, c8, c9 = st.columns(3)
                with c7:
                    state = st.selectbox(
                        f"stateProvince_Y{y}",
                        OPTIONS["stateProvince"],
                        key=f"sp_state_{y}",
                        help=VAR_HELP["stateProvince"]
                    )
                    add_if_in_schema(row, f"stateProvince_Y{y}", state, schema_cols)

                with c8:
                    fac = st.selectbox(
                        f"facilityName_Y{y}",
                        OPTIONS["facilityName"],
                        key=f"sp_fac_{y}",
                        help=VAR_HELP["facilityName"]
                    )
                    add_if_in_schema(row, f"facilityName_Y{y}", fac, schema_cols)

                with c9:
                    sup = st.selectbox(
                        f"suppressed_lt1000_Y{y}",
                        OPTIONS["suppressed_lt1000"],
                        key=f"sp_sup_{y}",
                        help=VAR_HELP["suppressed_lt1000"]
                    )
                    st.caption("Note: 0 = No, 1 = Yes")
                    add_if_in_schema(row, f"suppressed_lt1000_Y{y}", int(sup), schema_cols)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    if st.button("Predict (Single Patient)", type="primary", key="sp_predict_btn"):
        df1 = pd.DataFrame([row])
        res = predict_df(df1, available[chosen_key])

        missing_txt = str(res.loc[0, "missing_features_filled"] or "").strip()
        if missing_txt:
            st.error("Please complete the missing inputs before predicting.")
            st.write("Missing features detected:", missing_txt)
            st.stop()

        prob = float(res.loc[0, "pred_prob_unsuppressed"])
        pred = int(res.loc[0, "pred_class"])
        thr = float(res.loc[0, "used_threshold"])

        left, right = st.columns([1, 1])

        with left:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("### Result")
            st.write(
                {
                    "Predicted probability (unsuppressed risk)": prob,
                    "BestF1 threshold used": thr,
                    "Predicted class (1 = at-risk/not suppressed, 0 = likely suppressed)": pred,
                }
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("### Technical details")
            st.write("Missing features auto-filled: None")
            st.write("Row preview:")
            st.dataframe(res)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("## 🤖 Agentic AI (Validation → Explanation → Audit)")

        st.session_state["agent_row"] = row
        st.session_state["agent_prob"] = prob
        st.session_state["agent_pred"] = pred
        st.session_state["agent_thr"] = thr
        st.session_state["agent_model_key"] = chosen_key

        a1, a2, a3 = st.columns(3)

        with a1:
            if st.button("1) Run Validation Agent", key="agent_validate_btn"):
                v = agent_validate_row(st.session_state["agent_row"], st.session_state["sp_max_year"])
                st.session_state["agent_validation"] = v
                if v["errors"]:
                    st.error("Validation errors found.")
                elif v["warnings"]:
                    st.warning("Validation completed with warnings.")
                else:
                    st.success("Validation passed (no errors).")

        with a2:
            use_llm = st.toggle(
                "Use AI narrative (LLM)",
                value=False,
                help="Optional. Requires LLM secrets configured. If not configured, the app will show the standard explanation.",
            )
            if st.button("2) Generate Explanation Agent", key="agent_explain_btn"):
                v = st.session_state.get("agent_validation")
                row_for_explain = v["row"] if v else st.session_state["agent_row"]

                explanation = agent_explain(
                    row=row_for_explain,
                    prob_unsupp=st.session_state["agent_prob"],
                    pred_class=st.session_state["agent_pred"],
                    thr=st.session_state["agent_thr"],
                    max_year=st.session_state["sp_max_year"],
                    use_llm=use_llm,
                )
                st.session_state["agent_explanation"] = explanation
                st.success("Explanation generated.")

        with a3:
            if st.button("3) Create Audit Record", key="agent_audit_btn"):
                v = st.session_state.get("agent_validation")
                row_for_audit = v["row"] if v else st.session_state["agent_row"]

                audit = agent_build_audit_record(
                    row=row_for_audit,
                    chosen_key=st.session_state["agent_model_key"],
                    prob=st.session_state["agent_prob"],
                    pred=st.session_state["agent_pred"],
                    thr=st.session_state["agent_thr"],
                    max_year=st.session_state["sp_max_year"],
                )
                st.session_state.setdefault("audit_log", [])
                st.session_state["audit_log"].append(audit)
                st.success("Audit record added (session).")

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

        if st.session_state.get("audit_log"):
            st.markdown("### Audit log (this session)")
            st.dataframe(pd.DataFrame(st.session_state["audit_log"]))

with tab2:
    st.subheader("Batch scoring (Upload CSV)")
    st.markdown(
        "<div class='small-note'>Upload a CSV. The app will auto-select the best model based on year columns present (_Y1.._Y4).</div>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch")

    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        st.write("Preview:", df_up.head())

        chosen_key = choose_model_key_from_df(df_up, available_keys)
        if not chosen_key:
            st.error("Could not auto-select a model for this CSV. Ensure it contains *_Y1..*_Y4 columns.")
            st.stop()

        st.success(f"Auto-selected model: {chosen_key}")

        if st.button("Run batch predictions", type="primary", key="batch_predict_btn"):
            res = predict_df(df_up, available[chosen_key])
            st.dataframe(res.head(50))

            out_csv = res.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions CSV",
                data=out_csv,
                file_name=f"predictions_{chosen_key}.csv",
                mime="text/csv",
                key="batch_download_btn",
            )

