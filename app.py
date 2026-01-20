import json
import math
import datetime as dt
from pathlib import Path
from typing import Dict, Tuple, List, Any

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
import requests

# =========================
# PATHS
# =========================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# =========================
# DROPDOWN OPTIONS
# =========================
OPTIONS = {
    "who_stage": [1, 2, 3, 4],
    "suppressed_lt1000": [0, 1],  # 0=No, 1=Yes
    "functional_status": ["Ambulatory", "Bedridden", "Working"],
    "regimen_line": ["1st-line", "2nd-line"],
    "regimen_type": ["AZT/3TC/NVP", "TDF/3TC/DTG", "TDF/3TC/EFV", "AZT/3TC/LPV/r"],
    "tb_status": ["History of TB", "Active TB", "No TB"],
    "gender": ["Male", "Female"],
}

# =========================
# VARIABLE HELP
# =========================
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

# =========================
# UI CONFIG
# =========================
st.set_page_config(page_title="HIV Viral Suppression Risk Tool", layout="wide")

# ---------- Improved CSS (product look)
st.markdown(
    """
    <style>
      .stApp { background: #0b0f14; color: #e5e7eb; }
      .block-container { padding-top: 1.0rem; padding-bottom: 2.5rem; max-width: 1200px; }

      .card {
        background: rgba(15, 23, 42, 0.85);
        padding: 16px 18px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.10);
        box-shadow: 0 10px 24px rgba(0,0,0,0.35);
        backdrop-filter: blur(8px);
        margin-bottom: 12px;
      }
      .hero-title { font-size: 1.55rem; font-weight: 850; margin: 0; }
      .hero-sub { color: #cbd5e1; font-size: 0.92rem; margin-top: 6px; line-height: 1.35; }
      .small-note { color: #cbd5e1; font-size: 0.90rem; line-height: 1.35; }

      .stTextInput input, .stNumberInput input {
        background: rgba(11, 18, 32, 0.95) !important;
        color: #e5e7eb !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
      }
      div[data-baseweb="select"] > div {
        background: rgba(11, 18, 32, 0.95) !important;
        color: #e5e7eb !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255,255,255,0.12) !important;
      }

      div[data-baseweb="tab-list"]{
        gap: 10px;
        border-bottom: 1px solid rgba(255,255,255,0.12);
        padding-bottom: 10px;
      }
      button[role="tab"]{
        border-radius: 14px !important;
        padding: 10px 14px !important;
        border: 1px solid rgba(255,255,255,0.14) !important;
        color: #e5e7eb !important;
        font-weight: 800 !important;
        background: rgba(15,23,42,0.35) !important;
      }
      button[role="tab"][aria-selected="true"]{
        background: rgba(34, 197, 94, 0.15) !important;
        border: 1px solid rgba(34,197,94,0.45) !important;
      }

      details {
        background: rgba(15,23,42,0.40);
        border: 1px solid rgba(255,255,255,0.10);
        border-radius: 14px;
        padding: 8px 10px;
      }

      footer { visibility: hidden; }
      #MainMenu { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Product header (no model type exposure)
st.markdown(
    """
    <div class="card">
      <div class="hero-title">HIV Viral Suppression Risk Tool</div>
      <div class="hero-sub">
        Program decision-support • Baseline + follow-up (if available) • Saves one row per prediction to Google Sheets
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="card">
      <b>Disclaimer</b><br/>
      <span class="small-note">Decision-support demo. Not for standalone clinical decisions.</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# Admin gate (restricted items)
# =========================
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

# =========================
# Load available models
# =========================
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

# =========================
# Model aliasing (hide real names)
# =========================
def build_model_alias_map(keys: List[str]) -> Dict[str, str]:
    keys_sorted = sorted(keys)
    alias = {}
    for i, k in enumerate(keys_sorted):
        alias[k] = f"Model {chr(ord('A') + i)}" if i < 26 else f"Model {i+1}"
    return alias

MODEL_ALIAS = build_model_alias_map(list(available_keys))

# =========================
# Helpers
# =========================
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
# Google Sheets logging (webhook)
# =========================
def gsheet_enabled() -> bool:
    return bool(st.secrets.get("GSHEETS_WEBHOOK_URL", "")) and bool(st.secrets.get("GSHEETS_TOKEN", ""))

def send_to_gsheet(payload: dict) -> tuple[bool, str]:
    if not gsheet_enabled():
        return False, "Google Sheets not configured."
    try:
        url = st.secrets["GSHEETS_WEBHOOK_URL"]
        token = st.secrets["GSHEETS_TOKEN"]
        payload = dict(payload)
        payload["token"] = token

        r = requests.post(url, json=payload, timeout=12)
        if r.status_code != 200:
            return False, f"Webhook HTTP {r.status_code}"
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        if data.get("status") != "success":
            return False, "Webhook did not return success."
        return True, "Saved to Google Sheet."
    except Exception as e:
        return False, f"Sheets error: {e}"

# =========================
# Agentic AI helpers
# =========================
def run_validation_agent(row: Dict[str, Any]) -> Dict[str, Any]:
    errors = []
    warnings = []
    fixes = []

    def get_num(k, default=0.0):
        try:
            return float(row.get(k, default))
        except Exception:
            return float(default)

    vl = get_num("viral_load_T0", 0.0)
    if vl <= 0:
        warnings.append("viral_load_T0 is 0 or missing; log10_vl_T0 will be set to 0.")
        fixes.append("Set viral_load_T0 > 0 if available.")

    state = str(row.get("stateProvince_T0", "")).strip()
    fac = str(row.get("facilityName_T0", "")).strip()
    if not state or state.lower() == "unknown":
        warnings.append("stateProvince_T0 not provided.")
        fixes.append("Enter State/Province to improve retraining data quality.")
    if not fac or fac.lower() == "unknown":
        warnings.append("facilityName_T0 not provided.")
        fixes.append("Enter Facility Name to improve retraining data quality.")

    ok = len(errors) == 0
    return {
        "agent_ok": ok,
        "agent_errors": errors,
        "agent_warnings": warnings,
        "agent_fixes": fixes,
    }

def llm_enabled() -> bool:
    return bool(st.secrets.get("LLM_BASE_URL", "")) and bool(st.secrets.get("LLM_MODEL", "")) and bool(st.secrets.get("LLM_API_KEY", ""))

def generate_explanation_template(prob: float, thr: float, row: Dict[str, Any]) -> str:
    vl = row.get("viral_load_T0", None)
    logvl = row.get("log10_vl_T0", None)
    missed = row.get("missed_appointments_T0", None)
    late = row.get("days_late_T0", None)
    cd4 = row.get("cd4_T0", None)
    adh = row.get("adherence_prop_T0", None)

    lines = []
    lines.append(f"Summary: {'Higher' if prob >= thr else 'Lower'} risk of non-suppression. Probability (unsuppressed risk) = {prob:.3f} using threshold {thr:.3f}.")
    lines.append("")
    lines.append("Latest inputs used (program interpretation):")
    if vl is not None:
        lines.append(f"- Viral load: {vl} (log10: {logvl}) — higher values generally increase risk.")
    if adh is not None:
        lines.append(f"- Refill adherence: {float(adh)*100:.1f}% — lower adherence generally increases risk.")
    if missed is not None:
        lines.append(f"- Missed appointments: {missed} — more missed visits can increase risk.")
    if late is not None:
        lines.append(f"- Days late: {late} — frequent delays can signal gaps in continuity.")
    if cd4 is not None:
        lines.append(f"- CD4: {cd4} — may correlate with risk depending on context.")
    lines.append("")
    lines.append("Suggested next program actions (non-clinical):")
    lines.append("- Prioritize adherence support / follow-up in routine workflow.")
    lines.append("- Verify data completeness (VL, refill period, missed visits).")
    lines.append("- Schedule routine review based on your SOP.")
    lines.append("")
    lines.append("Note: Decision-support for program workflows; not diagnosis/treatment advice.")
    return "\n".join(lines)

def generate_explanation_llm(prob: float, thr: float, row: Dict[str, Any]) -> Tuple[bool, str]:
    if not llm_enabled():
        return False, "LLM not configured."
    try:
        base = st.secrets["LLM_BASE_URL"].rstrip("/")
        model = st.secrets["LLM_MODEL"]
        api_key = st.secrets["LLM_API_KEY"]

        prompt = (
            "You are a public health program decision-support assistant.\n"
            "Write a short, clear explanation (no clinical advice) of HIV viral suppression risk prediction.\n"
            "Use bullet points. Mention it is for program workflow support, not diagnosis.\n\n"
            f"Probability (unsuppressed risk): {prob:.4f}\n"
            f"Threshold used: {thr:.4f}\n"
            f"Inputs JSON: {json.dumps(row)}\n"
        )

        url = f"{base}/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.3}
        r = requests.post(url, headers=headers, json=payload, timeout=18)
        if r.status_code != 200:
            return False, f"LLM error HTTP {r.status_code}"

        data = r.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        text = (text or "").strip()
        if not text:
            return False, "LLM returned empty text."
        return True, text

    except Exception as e:
        return False, f"LLM exception: {e}"

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["🧍 Single patient", "📤 Upload CSV (batch)"])

# =========================
# TAB 1: Single patient
# =========================
with tab1:
    st.markdown(
        """
        <div class="card">
          <div class="small-note">
            Fill <b>Baseline (T0)</b> first. Add follow-up years only if available. The tool auto-selects the appropriate internal model.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    step = st.radio(
        "Workflow",
        options=["Step 1: Window", "Step 2: Baseline (T0)", "Step 3: Follow-up (optional)", "Step 4: Predict & Results"],
        horizontal=True,
        key="sp_step",
    )

    use_llm = st.toggle("Use AI narrative (LLM) for explanation", value=True, key="use_llm_toggle")

    max_followup_year = st.session_state.get("sp_max_followup", 0)
    chosen_key = None
    model_alias = None
    meta = None
    schema_cols = None

    if step == "Step 1: Window":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        max_followup_year = st.selectbox(
            "What is the latest FOLLOW-UP year you have available?",
            options=[0, 1, 2, 3, 4],
            format_func=lambda x: "T0 only (predict next year)" if x == 0 else f"T0 + Y1..Y{x} (predict next year)",
            index=0,
            key="sp_max_followup",
        )
        chosen_key = choose_model_key_by_year(max_followup_year, available_keys)
        if not chosen_key:
            st.error("No matching model found for the selected window. Check your model filenames in models/.")
            st.stop()
        model_alias = MODEL_ALIAS.get(chosen_key, "Model")
        st.success(f"Auto-selected model: {model_alias}")
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        max_followup_year = st.session_state.get("sp_max_followup", 0)
        chosen_key = choose_model_key_by_year(max_followup_year, available_keys)
        if not chosen_key:
            st.error("No matching model found for the selected window. Go back to Step 1.")
            st.stop()
        model_alias = MODEL_ALIAS.get(chosen_key, "Model")
        meta = available[chosen_key]["meta"]
        schema_cols = set(meta["feature_cols"])

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write(f"**Auto-selected model:** {model_alias}")
        st.write(f"**Selected window:** {'T0 only' if max_followup_year == 0 else f'T0 + Y1..Y{max_followup_year}'}")
        st.markdown("</div>", unsafe_allow_html=True)

    row = st.session_state.get("sp_row_cache", {})

    def render_timepoint_inputs(suf: str, expanded: bool):
        nonlocal row
        label = "Baseline (T0)" if suf == "_T0" else f"Year {suf.replace('_Y','')} (Follow-up)"

        with st.expander(f"{label} inputs", expanded=expanded):
            st.markdown("#### Clinical / adherence metrics")

            n1, n2, n3 = st.columns(3)
            with n1:
                age = st.number_input(
                    f"age{suf}",
                    min_value=0,
                    max_value=120,
                    value=int(row.get(f"age{suf}", 35)),
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
                    value=int(row.get(f"cd4{suf}", 350)),
                    step=10,
                    key=f"sp_cd4{suf}",
                    help=VAR_HELP["cd4"],
                )
                add_if_in_schema(row, f"cd4{suf}", float(cd4), schema_cols)

            with n3:
                vl = st.number_input(
                    f"viral_load{suf}",
                    min_value=0.0,
                    value=float(row.get(f"viral_load{suf}", 1000.0)),
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
                    value=float(row.get(f"weight{suf}", 60.0)),
                    step=0.5,
                    key=f"sp_wt{suf}",
                    help=VAR_HELP["weight"],
                )
                add_if_in_schema(row, f"weight{suf}", float(wt), schema_cols)

            with n6:
                missed = st.number_input(
                    f"missed_appointments{suf}",
                    min_value=0,
                    value=int(row.get(f"missed_appointments{suf}", 0)),
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
                    value=int(st.session_state.get(f"days_in_{suf}", 30)),
                    step=1,
                    key=f"sp_days_in_period{suf}",
                    help=VAR_HELP["days_in_period"],
                )
                st.session_state[f"days_in_{suf}"] = days_in_period

            with cB:
                days_covered = st.number_input(
                    f"Days covered by refills ({label})",
                    min_value=0,
                    max_value=365,
                    value=int(st.session_state.get(f"days_cov_{suf}", 0)),
                    step=1,
                    key=f"sp_days_covered{suf}",
                    help=VAR_HELP["days_covered"],
                )
                st.session_state[f"days_cov_{suf}"] = days_covered

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
                value=int(row.get(f"days_late{suf}", 0)),
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
                    value=int(row.get("age_baseline_T0", 35)),
                    step=1,
                    key="sp_base_age_T0",
                    help=VAR_HELP["age_baseline"],
                )
                add_if_in_schema(row, "age_baseline_T0", float(base_age), schema_cols)

            st.markdown("#### Program / clinical context")

            c1, c2, c3 = st.columns(3)
            with c1:
                g_val = row.get(f"gender{suf}", "Male")
                g_idx = OPTIONS["gender"].index(g_val) if g_val in OPTIONS["gender"] else 0
                gender = st.selectbox(
                    f"gender{suf}",
                    OPTIONS["gender"],
                    index=g_idx,
                    key=f"sp_gender{suf}",
                    help=VAR_HELP["gender"],
                )
                add_if_in_schema(row, f"gender{suf}", gender, schema_cols)

            with c2:
                f_val = row.get(f"functional_status{suf}", "Ambulatory")
                f_idx = OPTIONS["functional_status"].index(f_val) if f_val in OPTIONS["functional_status"] else 0
                func = st.selectbox(
                    f"functional_status{suf}",
                    OPTIONS["functional_status"],
                    index=f_idx,
                    key=f"sp_func{suf}",
                    help=VAR_HELP["functional_status"],
                )
                add_if_in_schema(row, f"functional_status{suf}", func, schema_cols)

            with c3:
                rl_val = row.get(f"regimen_line{suf}", "1st-line")
                rl_idx = OPTIONS["regimen_line"].index(rl_val) if rl_val in OPTIONS["regimen_line"] else 0
                regimen_line = st.selectbox(
                    f"regimen_line{suf}",
                    OPTIONS["regimen_line"],
                    index=rl_idx,
                    key=f"sp_line{suf}",
                    help=VAR_HELP["regimen_line"],
                )
                add_if_in_schema(row, f"regimen_line{suf}", regimen_line, schema_cols)

            c4, c5, c6 = st.columns(3)
            with c4:
                rt_val = row.get(f"regimen_type{suf}", OPTIONS["regimen_type"][0])
                rt_idx = OPTIONS["regimen_type"].index(rt_val) if rt_val in OPTIONS["regimen_type"] else 0
                regimen_type = st.selectbox(
                    f"regimen_type{suf}",
                    OPTIONS["regimen_type"],
                    index=rt_idx,
                    key=f"sp_type{suf}",
                    help=VAR_HELP["regimen_type"],
                )
                add_if_in_schema(row, f"regimen_type{suf}", regimen_type, schema_cols)

            with c5:
                tb_val = row.get(f"tb_status{suf}", OPTIONS["tb_status"][0])
                tb_idx = OPTIONS["tb_status"].index(tb_val) if tb_val in OPTIONS["tb_status"] else 0
                tb = st.selectbox(
                    f"tb_status{suf}",
                    OPTIONS["tb_status"],
                    index=tb_idx,
                    key=f"sp_tb{suf}",
                    help=VAR_HELP["tb_status"],
                )
                add_if_in_schema(row, f"tb_status{suf}", tb, schema_cols)

            with c6:
                who_val = int(row.get(f"who_stage{suf}", 1))
                who_idx = OPTIONS["who_stage"].index(who_val) if who_val in OPTIONS["who_stage"] else 0
                who = st.selectbox(
                    f"who_stage{suf}",
                    OPTIONS["who_stage"],
                    index=who_idx,
                    key=f"sp_who{suf}",
                    help=VAR_HELP["who_stage"],
                )
                add_if_in_schema(row, f"who_stage{suf}", int(who), schema_cols)

            # Open entry
            c7, c8, c9 = st.columns(3)
            with c7:
                state = st.text_input(
                    f"stateProvince{suf}",
                    value=str(row.get(f"stateProvince{suf}", "")),
                    key=f"sp_state{suf}",
                    help=VAR_HELP["stateProvince"],
                )
                add_if_in_schema(row, f"stateProvince{suf}", state.strip() if state else "Unknown", schema_cols)

            with c8:
                fac = st.text_input(
                    f"facilityName{suf}",
                    value=str(row.get(f"facilityName{suf}", "")),
                    key=f"sp_fac{suf}",
                    help=VAR_HELP["facilityName"],
                )
                add_if_in_schema(row, f"facilityName{suf}", fac.strip() if fac else "Unknown", schema_cols)

            with c9:
                sup_val = int(row.get(f"suppressed_lt1000{suf}", 0))
                sup_idx = OPTIONS["suppressed_lt1000"].index(sup_val) if sup_val in OPTIONS["suppressed_lt1000"] else 0
                sup = st.selectbox(
                    f"suppressed_lt1000{suf}",
                    OPTIONS["suppressed_lt1000"],
                    index=sup_idx,
                    key=f"sp_sup{suf}",
                    help=VAR_HELP["suppressed_lt1000"],
                )
                st.caption("0 = No, 1 = Yes")
                add_if_in_schema(row, f"suppressed_lt1000{suf}", int(sup), schema_cols)

    # --- Render Steps
    if step == "Step 2: Baseline (T0)":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        render_timepoint_inputs("_T0", expanded=True)
        st.markdown("</div>", unsafe_allow_html=True)

    if step == "Step 3: Follow-up (optional)":
        if max_followup_year == 0:
            st.info("No follow-up selected (T0 only). Go to Step 1 if you want to add follow-up years.")
        else:
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            for suf in timepoints(max_followup_year):
                if suf == "_T0":
                    continue
                render_timepoint_inputs(suf, expanded=False)
            st.markdown("</div>", unsafe_allow_html=True)

    # Cache row continuously
    st.session_state["sp_row_cache"] = row

    # --- Predict & Results
    if step == "Step 4: Predict & Results":
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        colA, colB, colC = st.columns([1, 1, 1])
        with colA:
            do_predict = st.button("Predict (Single Patient)", type="primary", key="sp_predict_btn")
        with colB:
            do_explain = st.button("Generate / Re-generate Explanation", key="sp_explain_btn")
        with colC:
            if st.button("Reset inputs", key="sp_reset_btn"):
                st.session_state["sp_row_cache"] = {}
                for k in ["last_res", "agent_bundle", "last_prob", "last_pred", "last_thr",
                          "last_model_key", "last_model_alias", "last_max_followup"]:
                    st.session_state.pop(k, None)
                st.success("Reset done. Go back to Step 2 to re-enter.")
        st.markdown("</div>", unsafe_allow_html=True)

        # --- Prediction
        if do_predict:
            with st.spinner("Running prediction..."):
                df1 = pd.DataFrame([row])
                res = predict_df(df1, available[chosen_key])

            prob = float(res.loc[0, "pred_prob_unsuppressed"])
            pred = int(res.loc[0, "pred_class"])
            thr = float(res.loc[0, "used_threshold"])

            st.session_state["last_res"] = res
            st.session_state["last_prob"] = prob
            st.session_state["last_pred"] = pred
            st.session_state["last_thr"] = thr
            st.session_state["last_model_key"] = chosen_key
            st.session_state["last_model_alias"] = model_alias
            st.session_state["last_max_followup"] = int(max_followup_year)

            # Agentic validation
            agent_bundle = run_validation_agent(row)

            # Explanation default (fast template) to avoid timeout
            agent_bundle["agent_explanation"] = generate_explanation_template(prob, thr, row)
            agent_bundle["explanation_source"] = "template"
            st.session_state["agent_bundle"] = agent_bundle

            # Missing features warning
            missing_txt = str(res.loc[0, "missing_features_filled"] or "").strip()
            if missing_txt:
                st.warning("Some model features were not provided and were auto-filled with defaults.")
                with st.expander("See auto-filled features"):
                    st.write(missing_txt)

            # Save to Google Sheets (non-blocking-ish with shorter payload)
            payload = {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "model_alias": model_alias,
                "prediction": pred,
                "probability": prob,
                "threshold": thr,
                "max_followup_year": int(max_followup_year),
                "state": row.get("stateProvince_T0", "Unknown"),
                "facility": row.get("facilityName_T0", "Unknown"),
                "agent_ok": agent_bundle.get("agent_ok", True),
                "agent_errors": json.dumps(agent_bundle.get("agent_errors", [])),
                "agent_warnings": json.dumps(agent_bundle.get("agent_warnings", [])),
                "agent_fixes": json.dumps(agent_bundle.get("agent_fixes", [])),
                "inputs_json": json.dumps(row),
            }
            ok, msg = send_to_gsheet(payload)
            if ok:
                st.success(msg)
            else:
                st.info(msg)

        # --- On-demand LLM Explanation (prevents timeouts on predict)
        if do_explain and "last_prob" in st.session_state:
            prob = float(st.session_state["last_prob"])
            thr = float(st.session_state["last_thr"])
            with st.spinner("Generating explanation (LLM)..."):
                if use_llm:
                    ok_llm, text = generate_explanation_llm(prob, thr, row)
                    if ok_llm:
                        st.session_state["agent_bundle"]["agent_explanation"] = text
                        st.session_state["agent_bundle"]["explanation_source"] = "llm"
                        st.success("LLM explanation generated.")
                    else:
                        st.session_state["agent_bundle"]["agent_explanation"] = generate_explanation_template(prob, thr, row)
                        st.session_state["agent_bundle"]["explanation_source"] = "template"
                        st.warning(text)
                else:
                    st.info("LLM toggle is off. Enable it to use LLM narrative.")

        # ---------- Results panel
        if "last_res" in st.session_state:
            prob = float(st.session_state["last_prob"])
            pred = int(st.session_state["last_pred"])
            thr = float(st.session_state["last_thr"])
            model_alias = st.session_state.get("last_model_alias", "Model")

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Result")
            st.write(f"**Model used:** {model_alias}")

            # Visual risk bar
            st.progress(min(max(prob, 0.0), 1.0))

            label = "High risk (likely not suppressed)" if pred == 1 else "Lower risk (likely suppressed)"
            st.write(f"**Risk score (unsuppressed probability):** `{prob:.3f}`")
            st.write(f"**Threshold used:** `{thr:.3f}`")
            st.write(f"**Classification:** **{label}**")
            st.markdown("</div>", unsafe_allow_html=True)

            # Technical details (collapsed to reduce render cost)
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Technical details")
            with st.expander("Show prediction dataframe"):
                st.dataframe(st.session_state["last_res"], use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Agentic AI section
            agent_bundle = st.session_state.get("agent_bundle", None)
            if agent_bundle:
                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("## 🤖 Agentic AI (Validation → Explanation → Audit)")
                tV, tE, tA = st.tabs(["✅ Validation", "📝 Explanation", "🧾 Audit"])

                with tV:
                    st.write("**Agent status:**", "OK ✅" if agent_bundle.get("agent_ok") else "Needs review ⚠️")
                    errs = agent_bundle.get("agent_errors", [])
                    warns = agent_bundle.get("agent_warnings", [])
                    fixes = agent_bundle.get("agent_fixes", [])
                    if errs:
                        st.error("Errors")
                        st.write(errs)
                    if warns:
                        st.warning("Warnings")
                        st.write(warns)
                    if fixes:
                        st.info("Suggested fixes")
                        st.write(fixes)

                with tE:
                    src = agent_bundle.get("explanation_source", "template")
                    st.caption(f"Explanation source: {src}")
                    st.write(agent_bundle.get("agent_explanation", ""))

                with tA:
                    st.write("Audit snapshot (what was stored to Sheets):")
                    st.json({
                        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                        "model_alias": st.session_state.get("last_model_alias"),
                        "max_followup_year": st.session_state.get("last_max_followup"),
                        "pred_prob_unsuppressed": st.session_state.get("last_prob"),
                        "pred_class": st.session_state.get("last_pred"),
                        "threshold": st.session_state.get("last_thr"),
                        "state_T0": row.get("stateProvince_T0", "Unknown"),
                        "facility_T0": row.get("facilityName_T0", "Unknown"),
                        "agent_ok": agent_bundle.get("agent_ok", True),
                    })
                st.markdown("</div>", unsafe_allow_html=True)

            # Admin-only download (restricted)
            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("## 🔐 Admin-only: download latest inputs (debugging only)")
            if not admin_ok():
                st.info("Admin locked. Use sidebar to unlock.")
            else:
                blob = json.dumps(st.session_state.get("sp_row_cache", {}), indent=2).encode("utf-8")
                st.download_button(
                    "Download JSON",
                    data=blob,
                    file_name="latest_inputs.json",
                    mime="application/json",
                )
            st.markdown("</div>", unsafe_allow_html=True)

# =========================
# TAB 2: Batch CSV
# =========================
with tab2:
    st.markdown(
        """
        <div class="card">
          <div class="small-note">
            Upload a CSV file. The app auto-selects the best chain model based on available follow-up columns.
            Output includes prediction probability, class, and used threshold.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_uploader")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        batch_use_llm = st.checkbox("Generate LLM explanation for a sample row (optional)", value=False)
    with col2:
        log_to_sheet = st.checkbox("Log summary to Google Sheets (recommended)", value=True)
    with col3:
        show_preview = st.checkbox("Show dataframe preview", value=False)

    if uploaded is not None:
        try:
            df_raw = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            st.stop()

        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.write(f"Rows: **{len(df_raw):,}**  |  Columns: **{len(df_raw.columns):,}**")
        st.markdown("</div>", unsafe_allow_html=True)

        model_key = choose_model_key_from_df(df_raw, available_keys)
        if not model_key:
            st.error("Could not auto-select a model based on your CSV columns. Ensure columns include _T0 and follow-up suffixes.")
            st.stop()

        model_alias = MODEL_ALIAS.get(model_key, "Model")
        st.success(f"Auto-selected model: {model_alias}")

        if show_preview:
            with st.expander("Preview uploaded data"):
                st.dataframe(df_raw.head(25), use_container_width=True)

        run_batch = st.button("Run batch prediction", type="primary", key="run_batch_btn")

        if run_batch:
            with st.spinner("Running batch prediction..."):
                res = predict_df(df_raw, available[model_key])

            st.markdown("<div class='card'>", unsafe_allow_html=True)
            st.markdown("### Batch results (preview)")
            st.dataframe(res.head(50), use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Download output
            out_csv = res.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions CSV",
                data=out_csv,
                file_name="hiv_predictions.csv",
                mime="text/csv",
            )

            # Optional: LLM explanation for a sample row
            if batch_use_llm:
                meta = available[model_key]["meta"]
                thr = extract_bestf1_threshold(meta, 0.5)
                sample_idx = 0
                sample_prob = float(res.loc[sample_idx, "pred_prob_unsuppressed"])
                sample_row = df_raw.iloc[sample_idx].to_dict()

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown("### Sample explanation (row 1)")
                if llm_enabled():
                    ok_llm, text = generate_explanation_llm(sample_prob, thr, sample_row)
                    if ok_llm:
                        st.write(text)
                    else:
                        st.warning(text)
                        st.write(generate_explanation_template(sample_prob, thr, sample_row))
                else:
                    st.info("LLM not configured; using template explanation.")
                    st.write(generate_explanation_template(sample_prob, thr, sample_row))
                st.markdown("</div>", unsafe_allow_html=True)

            # Optional logging: summary only (avoid massive payload)
            if log_to_sheet:
                meta = available[model_key]["meta"]
                thr = extract_bestf1_threshold(meta, 0.5)

                summary_payload = {
                    "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                    "batch": True,
                    "model_alias": model_alias,
                    "rows": int(len(res)),
                    "threshold": float(thr),
                    "mean_prob": float(np.mean(res["pred_prob_unsuppressed"].values)),
                    "pct_pred_1": float(np.mean(res["pred_class"].values) * 100.0),
                }
                ok, msg = send_to_gsheet(summary_payload)
                if ok:
                    st.success("Batch summary saved to Google Sheet.")
                else:
                    st.info(msg)

    else:
        st.info("Upload a CSV to run batch predictions.")


