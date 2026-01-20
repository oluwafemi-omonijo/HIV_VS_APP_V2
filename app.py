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
}

# -------------------------
# VARIABLE HELP
# -------------------------
VAR_HELP = {
    "age": "Client age in completed years (0–120).",
    "age_baseline": "Baseline age recorded at T0.",
    "cd4": "CD4 cell count (cells/mm³).",
    "viral_load": "Viral load in copies/mL. Must be > 0 to compute log10(VL).",
    "log10_vl": "Auto-calculated as log10(viral_load).",
    "weight": "Weight in kilograms (kg).",
    "missed_appointments": "Number of missed appointments within the period.",
    "days_late": "Total number of days late for visits/pickups within the period.",
    "days_in_period": "Assessment window length used for refill adherence (e.g., 30/60/90 days).",
    "days_covered": "Days client had ART available within the assessment period.",
    "pharmacy_refill_adherence_pct": "Auto: (days_covered ÷ days_in_period) × 100, capped 0–100%.",
    "adherence_prop": "Auto: pharmacy_refill_adherence_pct ÷ 100 (0–1).",
    "gender": "Sex recorded at enrollment.",
    "functional_status": "Client functional status at assessment.",
    "regimen_line": "ART regimen line (1st-line or 2nd-line).",
    "regimen_type": "ART regimen type/category.",
    "tb_status": "TB status category.",
    "who_stage": "WHO clinical stage (1–4).",
    "stateProvince": "State/Province where the client receives care (open entry).",
    "facilityName": "Facility where the client receives care (open entry).",
    "suppressed_lt1000": "Suppressed below 1000 copies/mL (0=No, 1=Yes).",
}

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="HIV Viral Suppression Risk", layout="wide")

st.markdown(
    """
    <style>
      .stApp { background: #0b0f14; color: #e5e7eb; }
      .block-container { padding-top: 1.2rem; }

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
      }

      .stTextInput input, .stNumberInput input, .stTextArea textarea {
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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="title-card">
      <h2 style="margin:0;">HIV Viral Suppression Risk (Chain Models)</h2>
      <div class="small-note">
        T0 baseline + follow-up years • Auto-selects correct chain model • Uses BestF1 threshold from metadata • Logs one row per prediction to Google Sheets
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
        Decision-support demo. Not for standalone clinical decisions.
      </span>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Admin gate
# -------------------------
def admin_enabled() -> bool:
    return bool(st.secrets.get("ADMIN_PASSWORD", ""))

def admin_ok() -> bool:
    return bool(st.session_state.get("admin_ok", False))

with st.sidebar:
    st.markdown("### 🔒 Admin")
    if not admin_enabled():
        st.caption("ADMIN_PASSWORD not set in secrets.")
    else:
        pw = st.text_input("Admin password", type="password")
        if st.button("Unlock admin"):
            if pw == st.secrets.get("ADMIN_PASSWORD"):
                st.session_state["admin_ok"] = True
                st.success("Admin unlocked.")
            else:
                st.session_state["admin_ok"] = False
                st.error("Wrong password.")

    st.markdown("---")
    st.markdown("### ⚙️ Logging")
    st.caption("Google Sheets logging is enabled when secrets are configured.")


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
    st.error("No models found in models/. Put your .keras and *_metadata.json in models/.")
    st.stop()

# -------------------------
# Hide real model names from end-users (aliasing)
# -------------------------
MODEL_ALIAS_MAP = {
    "DeepANN_T0_to_Y1": "Model A",
    "DeepANN_T0Y1_to_Y2": "Model B",
    "DeepANN_T0Y1Y2_to_Y3": "Model C",
    "DeepANN_T0Y1Y2Y3_to_Y4": "Model D",
    "DeepANN_T0Y1Y2Y3Y4_to_Y5": "Model E",
}

def model_alias(model_key: str) -> str:
    return MODEL_ALIAS_MAP.get(model_key, "Model")


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

def detect_max_followup_year_from_cols(cols) -> int:
    max_year = 0
    for y in [1, 2, 3, 4]:
        if any(str(c).endswith(f"_Y{y}") for c in cols):
            max_year = y
    return max_year

def choose_model_key_by_year(max_followup_year: int, keys: set):
    if max_followup_year >= 4 and "DeepANN_T0Y1Y2Y3Y4_to_Y5" in keys:
        return "DeepANN_T0Y1Y2Y3Y4_to_Y5"
    if max_followup_year == 3 and "DeepANN_T0Y1Y2Y3_to_Y4" in keys:
        return "DeepANN_T0Y1Y2Y3_to_Y4"
    if max_followup_year == 2 and "DeepANN_T0Y1Y2_to_Y3" in keys:
        return "DeepANN_T0Y1Y2_to_Y3"
    if max_followup_year == 1 and "DeepANN_T0Y1_to_Y2" in keys:
        return "DeepANN_T0Y1_to_Y2"
    if max_followup_year == 0 and "DeepANN_T0_to_Y1" in keys:
        return "DeepANN_T0_to_Y1"
    return None

def choose_model_key_from_df(df: pd.DataFrame, keys: set):
    max_year = detect_max_followup_year_from_cols(df.columns)
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

def timepoints(max_followup_year: int):
    suffixes = ["_T0"]
    for y in range(1, max_followup_year + 1):
        suffixes.append(f"_Y{y}")
    return suffixes


# =========================
# AGENTIC AI LAYER
# =========================
def agent_validate_row(row: dict, max_followup_year: int) -> dict:
    errors, warnings, fixes = [], [], []
    suffixes = timepoints(max_followup_year)

    for suf in suffixes:
        vl_key = f"viral_load{suf}"
        log_key = f"log10_vl{suf}"
        pct_key = f"pharmacy_refill_adherence_pct{suf}"
        prop_key = f"adherence_prop{suf}"

        if vl_key in row:
            vl = float(row.get(vl_key, 0.0) or 0.0)
            if vl < 0:
                errors.append(f"{vl_key}: cannot be negative.")
            if vl == 0:
                warnings.append(f"{vl_key}: VL is 0 → log10_vl will be 0.000 (check missing VL).")

        if (vl_key in row) and (log_key in row):
            vl = float(row.get(vl_key, 0.0) or 0.0)
            logvl = float(row.get(log_key, 0.0) or 0.0)
            expected = round(math.log10(vl), 3) if vl > 0 else 0.0
            if abs(logvl - expected) > 0.02:
                warnings.append(f"{log_key}: mismatch with log10({vl_key}). Using computed value.")
                row[log_key] = expected
                fixes.append(f"Set {log_key}={expected} from {vl_key}.")

        if pct_key in row:
            pct = float(row.get(pct_key, 0.0) or 0.0)
            if pct < 0 or pct > 100:
                warnings.append(f"{pct_key}: out of range; capped to 0–100.")
                pct = max(0.0, min(100.0, pct))
                row[pct_key] = round(pct, 2)
                fixes.append(f"Capped {pct_key} to {row[pct_key]}.")

        if (pct_key in row) and (prop_key in row):
            pct = float(row.get(pct_key, 0.0) or 0.0)
            prop = float(row.get(prop_key, 0.0) or 0.0)
            expected_prop = round(pct / 100.0, 4)
            if abs(prop - expected_prop) > 0.01:
                warnings.append(f"{prop_key}: mismatch with {pct_key}/100. Using computed value.")
                row[prop_key] = expected_prop
                fixes.append(f"Set {prop_key}={expected_prop} from {pct_key}.")

        if prop_key in row:
            prop = float(row.get(prop_key, 0.0) or 0.0)
            if prop < 0 or prop > 1:
                warnings.append(f"{prop_key}: out of range; capped to 0–1.")
                prop = max(0.0, min(1.0, prop))
                row[prop_key] = round(prop, 4)
                fixes.append(f"Capped {prop_key} to {row[prop_key]}.")

    ok = len(errors) == 0
    return {"ok": ok, "errors": errors, "warnings": warnings, "fixes": fixes, "row": row}

def agent_template_explanation(row: dict, prob_unsupp: float, pred_class: int, thr: float, max_followup_year: int) -> str:
    # Use latest available timepoint for summary
    suf = "_T0" if max_followup_year == 0 else f"_Y{max_followup_year}"

    vl = row.get(f"viral_load{suf}", None)
    logvl = row.get(f"log10_vl{suf}", None)
    pct = row.get(f"pharmacy_refill_adherence_pct{suf}", None)
    prop = row.get(f"adherence_prop{suf}", None)
    missed = row.get(f"missed_appointments{suf}", None)
    late = row.get(f"days_late{suf}", None)
    cd4 = row.get(f"cd4{suf}", None)

    risk_label = "Higher risk of non-suppression" if pred_class == 1 else "Lower risk of non-suppression"
    lines = []
    lines.append(f"**Summary:** {risk_label}.")
    lines.append(f"Probability (unsuppressed risk) = **{prob_unsupp:.3f}** using threshold **{thr:.3f}**.")

    lines.append("\n**Latest inputs used (program interpretation):**")
    if vl is not None:
        lines.append(f"- Viral load: **{vl}** (log10: **{logvl}**) — higher values generally increase risk.")
    if pct is not None:
        lines.append(f"- Refill adherence: **{pct}%** (prop: **{prop}**) — lower adherence generally increases risk.")
    if missed is not None:
        lines.append(f"- Missed appointments: **{missed}** — more missed visits can increase risk.")
    if late is not None:
        lines.append(f"- Days late: **{late}** — frequent delays can signal gaps in continuity.")
    if cd4 is not None:
        lines.append(f"- CD4: **{cd4}** — may correlate with risk depending on context.")

    lines.append("\n**Suggested next program actions (non-clinical):**")
    if pred_class == 1:
        lines.append("- Prioritize adherence support / follow-up in routine workflow.")
        lines.append("- Verify data completeness (VL, refill period, missed visits).")
        lines.append("- Schedule routine review based on your SOP.")
    else:
        lines.append("- Continue routine follow-up per SOP.")
        lines.append("- Maintain refill continuity and timely tracking.")

    lines.append("\n*Note: Decision-support for program workflows; not diagnosis/treatment advice.*")
    return "\n".join(lines)

def llm_enabled() -> bool:
    return bool(st.secrets.get("LLM_API_KEY", "")) and bool(st.secrets.get("LLM_BASE_URL", "")) and bool(st.secrets.get("LLM_MODEL", ""))

def call_llm_narrative(prompt: str) -> str:
    api_key = st.secrets.get("LLM_API_KEY")
    base_url = st.secrets.get("LLM_BASE_URL", "").rstrip("/")
    model = st.secrets.get("LLM_MODEL", "")

    if not api_key or not base_url or not model:
        return ""

    url = f"{base_url}/chat/completions" if base_url.endswith("/v1") else f"{base_url}/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an assistant embedded in a health program decision-support app. "
                    "Do NOT provide diagnosis, treatment, dosing, or regimen advice. "
                    "Keep outputs programmatic (data quality, follow-up prioritization, interpretation). "
                    "Assume all data is de-identified. Keep it concise and clear."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }

    try:
        r = requests.post(url, headers=headers, json=payload, timeout=12)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return ""

def agent_explain(row: dict, prob_unsupp: float, pred_class: int, thr: float, max_followup_year: int, use_llm: bool) -> str:
    base = agent_template_explanation(row, prob_unsupp, pred_class, thr, max_followup_year)
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

def agent_build_audit_record(row: dict, chosen_key: str, prob: float, pred: int, thr: float, max_followup_year: int) -> dict:
    # latest suffix used
    suf = "_T0" if max_followup_year == 0 else f"_Y{max_followup_year}"
    return {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "model_key": chosen_key,
        "model_alias": model_alias(chosen_key),
        "max_followup_year": int(max_followup_year),
        "pred_prob_unsuppressed": float(prob),
        "pred_class": int(pred),
        "threshold": float(thr),
        "viral_load_latest": row.get(f"viral_load{suf}", None),
        "log10_vl_latest": row.get(f"log10_vl{suf}", None),
        "refill_pct_latest": row.get(f"pharmacy_refill_adherence_pct{suf}", None),
        "adherence_prop_latest": row.get(f"adherence_prop{suf}", None),
    }


# -------------------------
# Google Sheets logging (webhook)
# -------------------------
def gsheet_enabled() -> bool:
    return bool(st.secrets.get("GSHEETS_WEBHOOK_URL", "")) and bool(st.secrets.get("GSHEETS_TOKEN", ""))

def send_to_gsheet(payload: dict) -> tuple[bool, str]:
    if not gsheet_enabled():
        return False, "Google Sheets not configured (missing secrets)."
    try:
        url = st.secrets["GSHEETS_WEBHOOK_URL"]
        token = st.secrets["GSHEETS_TOKEN"]

        payload = dict(payload)
        payload["token"] = token

        r = requests.post(url, json=payload, timeout=15)

        # Some Apps Script deployments return 302/303 redirects; treat those as failure with hint.
        if r.status_code != 200:
            detail = r.text[:250] if r.text else ""
            return False, f"Webhook HTTP {r.status_code}. {detail}"

        # Apps Script usually returns JSON
        try:
            data = r.json()
        except Exception:
            # if content-type is text/html etc.
            snippet = (r.text or "")[:250]
            return False, f"Webhook returned non-JSON response: {snippet}"

        if data.get("status") != "success":
            return False, f"Webhook response: {data}"

        return True, "Saved to Google Sheet."
    except Exception as e:
        return False, f"Sheets error: {e}"


# -------------------------
# Tabs
# -------------------------
tab1, tab2 = st.tabs(["🧍 Single patient form", "📤 Upload CSV (batch)"])

with tab1:
    st.subheader("Single patient form (T0 baseline → follow-up years)")
    st.markdown(
        "<div class='small-note'>Enter T0 baseline first. Add follow-up years if available. The app auto-selects the correct chain model.</div>",
        unsafe_allow_html=True,
    )

    max_followup_year = st.selectbox(
        "What is the latest FOLLOW-UP year you have available?",
        options=[0, 1, 2, 3, 4],
        format_func=lambda x: "T0 only (predict Y1)" if x == 0 else f"T0 + Y1..Y{x} (predict Y{x+1})",
        index=0,
        key="sp_max_followup",
    )

    chosen_key = choose_model_key_by_year(max_followup_year, available_keys)
    if not chosen_key:
        st.error("No matching model found for the selected time window. Confirm model filenames in models/.")
        st.stop()

    meta = available[chosen_key]["meta"]
    schema_cols = set(meta["feature_cols"])

    # Show only alias publicly
    st.success(f"Auto-selected model: {model_alias(chosen_key)}")

    # LLM toggle affects explanation generation
    use_llm = st.toggle(
        "Use AI narrative (LLM) for explanation",
        value=st.session_state.get("use_llm", False),
        key="use_llm",
        help="Optional. Requires LLM secrets configured. If not configured, standard explanation is used.",
    )

    row = {}
    with st.container():
        st.markdown("<div class='metric-card'>", unsafe_allow_html=True)

        for suf in timepoints(max_followup_year):
            label = "Baseline (T0)" if suf == "_T0" else f"Year {suf.replace('_Y','')} (Follow-up)"
            with st.expander(f"{label} inputs", expanded=(suf == "_T0")):
                st.markdown("#### Clinical / adherence metrics")

                n1, n2, n3 = st.columns(3)
                with n1:
                    age = st.number_input(
                        f"age{suf}",
                        min_value=0,
                        max_value=120,
                        value=35,
                        step=1,
                        key=f"sp_age{suf}",
                        help=VAR_HELP["age"],
                    )
                    add_if_in_schema(row, f"age{suf}", float(age), schema_cols)

                with n2:
                    cd4 = st.number_input(
                        f"cd4{suf}",
                        min_value=0,
                        max_value=5000,
                        value=350,
                        step=10,
                        key=f"sp_cd4{suf}",
                        help=VAR_HELP["cd4"],
                    )
                    add_if_in_schema(row, f"cd4{suf}", float(cd4), schema_cols)

                with n3:
                    vl = st.number_input(
                        f"viral_load{suf}",
                        min_value=0.0,
                        value=1000.0,
                        step=50.0,
                        key=f"sp_vl{suf}",
                        help=VAR_HELP["viral_load"],
                    )
                    add_if_in_schema(row, f"viral_load{suf}", float(vl), schema_cols)

                n4, n5, n6 = st.columns(3)
                with n4:
                    logvl = round(math.log10(vl), 3) if vl > 0 else 0.0
                    st.metric(label=f"log10_vl{suf} (auto)", value=f"{logvl:.3f}", help=VAR_HELP["log10_vl"])
                    add_if_in_schema(row, f"log10_vl{suf}", float(logvl), schema_cols)

                with n5:
                    wt = st.number_input(
                        f"weight{suf}",
                        min_value=0.0,
                        value=60.0,
                        step=0.5,
                        key=f"sp_wt{suf}",
                        help=VAR_HELP["weight"],
                    )
                    add_if_in_schema(row, f"weight{suf}", float(wt), schema_cols)

                with n6:
                    missed = st.number_input(
                        f"missed_appointments{suf}",
                        min_value=0,
                        value=0,
                        step=1,
                        key=f"sp_missed{suf}",
                        help=VAR_HELP["missed_appointments"],
                    )
                    add_if_in_schema(row, f"missed_appointments{suf}", float(missed), schema_cols)

                st.markdown("#### Pharmacy refill adherence calculator")

                cA, cB, cC = st.columns(3)
                with cA:
                    days_in_period = st.number_input(
                        f"Days in assessment period ({label})",
                        min_value=1,
                        max_value=365,
                        value=30,
                        step=1,
                        key=f"sp_days_in_period{suf}",
                        help=VAR_HELP["days_in_period"],
                    )
                with cB:
                    days_covered = st.number_input(
                        f"Days covered by refills ({label})",
                        min_value=0,
                        max_value=365,
                        value=0,
                        step=1,
                        key=f"sp_days_covered{suf}",
                        help=VAR_HELP["days_covered"],
                    )

                pct = (days_covered / days_in_period) * 100.0 if days_in_period > 0 else 0.0
                pct = max(0.0, min(100.0, pct))
                pct = round(pct, 2)
                prop = round(pct / 100.0, 4)

                with cC:
                    st.metric(label=f"pharmacy_refill_adherence_pct{suf} (auto)", value=f"{pct:.2f}%")
                    st.metric(label=f"adherence_prop{suf} (auto)", value=f"{prop:.4f}")

                add_if_in_schema(row, f"pharmacy_refill_adherence_pct{suf}", float(pct), schema_cols)
                add_if_in_schema(row, f"adherence_prop{suf}", float(prop), schema_cols)

                late = st.number_input(
                    f"days_late{suf}",
                    min_value=0,
                    value=0,
                    step=1,
                    key=f"sp_late{suf}",
                    help=VAR_HELP["days_late"],
                )
                add_if_in_schema(row, f"days_late{suf}", float(late), schema_cols)

                if suf == "_T0":
                    base_age = st.number_input(
                        "age_baseline_T0",
                        min_value=0,
                        max_value=120,
                        value=35,
                        step=1,
                        key="sp_base_age_T0",
                        help=VAR_HELP["age_baseline"],
                    )
                    add_if_in_schema(row, "age_baseline_T0", float(base_age), schema_cols)

                st.markdown("#### Program / clinical context")

                c1, c2, c3 = st.columns(3)
                with c1:
                    gender = st.selectbox(
                        f"gender{suf}",
                        OPTIONS["gender"],
                        key=f"sp_gender{suf}",
                        help=VAR_HELP["gender"],
                    )
                    add_if_in_schema(row, f"gender{suf}", gender, schema_cols)

                with c2:
                    func = st.selectbox(
                        f"functional_status{suf}",
                        OPTIONS["functional_status"],
                        key=f"sp_func{suf}",
                        help=VAR_HELP["functional_status"],
                    )
                    add_if_in_schema(row, f"functional_status{suf}", func, schema_cols)

                with c3:
                    regimen_line = st.selectbox(
                        f"regimen_line{suf}",
                        OPTIONS["regimen_line"],
                        key=f"sp_line{suf}",
                        help=VAR_HELP["regimen_line"],
                    )
                    add_if_in_schema(row, f"regimen_line{suf}", regimen_line, schema_cols)

                c4, c5, c6 = st.columns(3)
                with c4:
                    regimen_type = st.selectbox(
                        f"regimen_type{suf}",
                        OPTIONS["regimen_type"],
                        key=f"sp_type{suf}",
                        help=VAR_HELP["regimen_type"],
                    )
                    add_if_in_schema(row, f"regimen_type{suf}", regimen_type, schema_cols)

                with c5:
                    tb = st.selectbox(
                        f"tb_status{suf}",
                        OPTIONS["tb_status"],
                        key=f"sp_tb{suf}",
                        help=VAR_HELP["tb_status"],
                    )
                    add_if_in_schema(row, f"tb_status{suf}", tb, schema_cols)

                with c6:
                    who = st.selectbox(
                        f"who_stage{suf}",
                        OPTIONS["who_stage"],
                        key=f"sp_who{suf}",
                        help=VAR_HELP["who_stage"],
                    )
                    add_if_in_schema(row, f"who_stage{suf}", who, schema_cols)

                # ✅ OPEN ENTRY (not dropdown) for retraining data capture
                c7, c8, c9 = st.columns(3)
                with c7:
                    state = st.text_input(
                        f"stateProvince{suf}",
                        value="",
                        key=f"sp_state{suf}",
                        help=VAR_HELP["stateProvince"],
                    )
                    add_if_in_schema(row, f"stateProvince{suf}", state.strip() if state else "Unknown", schema_cols)

                with c8:
                    fac = st.text_input(
                        f"facilityName{suf}",
                        value="",
                        key=f"sp_fac{suf}",
                        help=VAR_HELP["facilityName"],
                    )
                    add_if_in_schema(row, f"facilityName{suf}", fac.strip() if fac else "Unknown", schema_cols)

                with c9:
                    sup = st.selectbox(
                        f"suppressed_lt1000{suf}",
                        OPTIONS["suppressed_lt1000"],
                        key=f"sp_sup{suf}",
                        help=VAR_HELP["suppressed_lt1000"],
                    )
                    st.caption("0 = No, 1 = Yes")
                    add_if_in_schema(row, f"suppressed_lt1000{suf}", int(sup), schema_cols)

        st.markdown("</div>", unsafe_allow_html=True)

    # -------------------------
    # Predict button
    # -------------------------
    if st.button("Predict (Single Patient)", type="primary", key="sp_predict_btn"):
        df1 = pd.DataFrame([row])
        res = predict_df(df1, available[chosen_key])

        # Save prediction to session
        st.session_state["last_row"] = row
        st.session_state["last_res"] = res
        st.session_state["last_prob"] = float(res.loc[0, "pred_prob_unsuppressed"])
        st.session_state["last_pred"] = int(res.loc[0, "pred_class"])
        st.session_state["last_thr"] = float(res.loc[0, "used_threshold"])
        st.session_state["last_model_key"] = chosen_key
        st.session_state["last_max_followup"] = int(max_followup_year)

        # Run agent validation + explanation immediately (so it can be logged in same row)
        v = agent_validate_row(dict(row), int(max_followup_year))
        st.session_state["agent_validation"] = v

        explanation = agent_explain(
            row=v["row"],
            prob_unsupp=st.session_state["last_prob"],
            pred_class=st.session_state["last_pred"],
            thr=st.session_state["last_thr"],
            max_followup_year=int(max_followup_year),
            use_llm=use_llm,
        )
        st.session_state["agent_explanation"] = explanation

        audit = agent_build_audit_record(
            row=v["row"],
            chosen_key=chosen_key,
            prob=st.session_state["last_prob"],
            pred=st.session_state["last_pred"],
            thr=st.session_state["last_thr"],
            max_followup_year=int(max_followup_year),
        )
        st.session_state.setdefault("audit_log", [])
        st.session_state["audit_log"].append(audit)

        # Missing features warning
        missing_txt = str(res.loc[0, "missing_features_filled"] or "").strip()
        if missing_txt:
            st.warning("Some model features were not provided and were auto-filled with defaults.")
            st.write("Auto-filled features:", missing_txt)

        # ✅ Google Sheets logging (ONE ROW PER PREDICTION)
        payload = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),

            # log real model key for YOU (not displayed to end-user)
            "model_key": chosen_key,
            "model_alias": model_alias(chosen_key),

            "prediction": int(st.session_state["last_pred"]),
            "probability": float(st.session_state["last_prob"]),
            "threshold": float(st.session_state["last_thr"]),
            "max_followup_year": int(max_followup_year),

            # store main identifiers at T0 (good for grouping)
            "state": row.get("stateProvince_T0", "Unknown"),
            "facility": row.get("facilityName_T0", "Unknown"),

            # agent outputs
            "agent_ok": bool(v["ok"]),
            "agent_errors": "\n".join(v["errors"]) if v["errors"] else "",
            "agent_warnings": "\n".join(v["warnings"]) if v["warnings"] else "",
            "agent_fixes": "\n".join(v["fixes"]) if v["fixes"] else "",
            "agent_note": explanation,

            # full raw inputs (json)
            "inputs_json": json.dumps(v["row"], ensure_ascii=False),
        }

        ok, msg = send_to_gsheet(payload)
        if ok:
            st.success(msg)
        else:
            st.warning(msg)
            if admin_ok():
                st.caption("Admin hint: check Apps Script deployment access = 'Anyone', and token match in secrets vs Script Properties.")

    # -------------------------
    # Results + Agent UI
    # -------------------------
    if "last_res" in st.session_state:
        prob = st.session_state["last_prob"]
        pred = st.session_state["last_pred"]
        thr = st.session_state["last_thr"]

        left, right = st.columns([1, 1])
        with left:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("### Result")
            st.write({
                "Predicted probability (unsuppressed risk)": prob,
                "Threshold used": thr,
                "Predicted class (1 = at-risk/not suppressed, 0 = likely suppressed)": pred,
                "Model used": model_alias(st.session_state["last_model_key"]),  # alias only
            })
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("### Technical details")
            st.dataframe(st.session_state["last_res"])
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("## 🤖 Agentic AI (Validation → Explanation → Audit)")

        a1, a2, a3 = st.columns(3)

        with a1:
            if st.button("1) Re-run Validation Agent", key="agent_validate_btn"):
                v = agent_validate_row(dict(st.session_state["last_row"]), st.session_state["last_max_followup"])
                st.session_state["agent_validation"] = v
                if v["errors"]:
                    st.error("Validation errors found.")
                elif v["warnings"]:
                    st.warning("Validation completed with warnings.")
                else:
                    st.success("Validation passed (no errors).")

        with a2:
            if st.button("2) Re-generate Explanation Agent", key="agent_explain_btn"):
                v = st.session_state.get("agent_validation")
                row_for_explain = v["row"] if v else st.session_state["last_row"]

                explanation = agent_explain(
                    row=row_for_explain,
                    prob_unsupp=st.session_state["last_prob"],
                    pred_class=st.session_state["last_pred"],
                    thr=st.session_state["last_thr"],
                    max_followup_year=st.session_state["last_max_followup"],
                    use_llm=st.session_state.get("use_llm", False),
                )
                st.session_state["agent_explanation"] = explanation
                st.success("Explanation generated (note: Google Sheet row is not updated; it remains one row per prediction).")

        with a3:
            if st.button("3) Create Audit Record (session)", key="agent_audit_btn"):
                v = st.session_state.get("agent_validation")
                row_for_audit = v["row"] if v else st.session_state["last_row"]

                audit = agent_build_audit_record(
                    row=row_for_audit,
                    chosen_key=st.session_state["last_model_key"],
                    prob=st.session_state["last_prob"],
                    pred=st.session_state["last_pred"],
                    thr=st.session_state["last_thr"],
                    max_followup_year=st.session_state["last_max_followup"],
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

        # ✅ Admin-only dataset export (restricted)
        st.markdown("## 🔐 Admin-only: download latest inputs (for debugging only)")
        if not admin_ok():
            st.info("Admin locked. Use sidebar to unlock.")
        else:
            st.download_button(
                "Download latest prediction inputs JSON",
                data=json.dumps(st.session_state.get("last_row", {}), indent=2).encode("utf-8"),
                file_name="latest_prediction_inputs.json",
                mime="application/json",
            )
    else:
        st.info("Run a prediction first to see results and agentic AI outputs.")


with tab2:
    st.subheader("Batch scoring (Upload CSV)")
    st.markdown(
        "<div class='small-note'>Upload a CSV containing T0 and optionally Y1..Y4 columns. The app auto-selects the correct chain model.</div>",
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch")

    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        st.write("Preview:", df_up.head())

        chosen_key = choose_model_key_from_df(df_up, available_keys)
        if not chosen_key:
            st.error("Could not auto-select a model. Ensure CSV has T0 and optional Y1..Y4 columns.")
            st.stop()

        st.success(f"Auto-selected model: {model_alias(chosen_key)}")

        if st.button("Run batch predictions", type="primary", key="batch_predict_btn"):
            res = predict_df(df_up, available[chosen_key])
            st.dataframe(res.head(50))

            out_csv = res.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions CSV",
                data=out_csv,
                file_name="predictions.csv",  # generic name
                mime="text/csv",
                key="batch_download_btn",
            )

