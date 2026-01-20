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
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="title-card">
      <h2 style="margin:0;">HIV Viral Suppression Risk (DeepANN Chain Models)</h2>
      <div class="small-note">
        T0 baseline + follow-up years • Auto-selects correct chain model • Uses BestF1 threshold from metadata • Optional Google Sheets logging
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
# Admin gate (for restricted downloads)
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
    """
    Returns max follow-up year present among Y1..Y4.
    Requires T0 to be present for chain models, but we just detect Y-year max here.
    """
    max_year = 0
    for y in [1, 2, 3, 4]:
        if any(str(c).endswith(f"_Y{y}") for c in cols):
            max_year = y
    return max_year

def choose_model_key_by_year(max_followup_year: int, keys: set):
    """
    max_followup_year = 0 means only T0 available -> predict Y1
    1 means T0+Y1 -> predict Y2
    ...
    4 means T0..Y4 -> predict Y5
    """
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

    # Fill missing with safe defaults
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
    """Return list of suffixes in order: T0, Y1..Ymax"""
    suffixes = ["_T0"]
    for y in range(1, max_followup_year + 1):
        suffixes.append(f"_Y{y}")
    return suffixes

# -------------------------
# Google Sheets logging (webhook)
# -------------------------
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

        r = requests.post(url, json=payload, timeout=10)
        if r.status_code != 200:
            return False, f"Webhook HTTP {r.status_code}"
        data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
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
        "<div class='small-note'>You MUST enter T0 baseline first. Then add follow-up years if you have them. The app auto-selects the correct chain model.</div>",
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
        st.error("No matching model found for the selected time window. Confirm your model filenames in models/.")
        st.stop()

    meta = available[chosen_key]["meta"]
    schema_cols = set(meta["feature_cols"])

    st.success(f"Auto-selected model: {chosen_key}")

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

                # log10
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

                # baseline age only at T0 (if exists in schema)
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

                # ✅ OPEN ENTRY (not dropdown)
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

    if st.button("Predict (Single Patient)", type="primary", key="sp_predict_btn"):
        df1 = pd.DataFrame([row])
        res = predict_df(df1, available[chosen_key])

        st.session_state["last_row"] = row
        st.session_state["last_res"] = res
        st.session_state["last_prob"] = float(res.loc[0, "pred_prob_unsuppressed"])
        st.session_state["last_pred"] = int(res.loc[0, "pred_class"])
        st.session_state["last_thr"] = float(res.loc[0, "used_threshold"])
        st.session_state["last_model_key"] = chosen_key
        st.session_state["last_max_followup"] = int(max_followup_year)

        # Missing features warning (now correct)
        missing_txt = str(res.loc[0, "missing_features_filled"] or "").strip()
        if missing_txt:
            st.warning("Some model features were not provided and were auto-filled with defaults.")
            st.write("Auto-filled features:", missing_txt)

        # ✅ Google Sheets logging (one row per prediction)
        payload = {
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
            "model_name": chosen_key,
            "prediction": int(st.session_state["last_pred"]),
            "probability": float(st.session_state["last_prob"]),
            "threshold": float(st.session_state["last_thr"]),
            "max_followup_year": int(max_followup_year),
            "state": row.get("stateProvince_T0", "Unknown"),
            "facility": row.get("facilityName_T0", "Unknown"),
            "agent_note": st.session_state.get("agent_explanation", ""),
            "inputs_json": json.dumps(row),
        }

        ok, msg = send_to_gsheet(payload)
        if ok:
            st.success(msg)
        else:
            st.info(msg)

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
                "BestF1 threshold used": thr,
                "Predicted class (1 = at-risk/not suppressed, 0 = likely suppressed)": pred,
            })
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown("<div class='metric-card'>", unsafe_allow_html=True)
            st.markdown("### Technical details")
            st.dataframe(st.session_state["last_res"])
            st.markdown("</div>", unsafe_allow_html=True)

        # ✅ Admin-only dataset export (restricted)
        st.markdown("## 🔐 Admin-only: download rows for retraining")
        if not admin_ok():
            st.info("Admin locked. Use sidebar to unlock.")
        else:
            # This exports the *current session* predictions if you store them,
            # but your main source of truth is the Google Sheet.
            if st.button("Download current-session prediction JSON (admin)"):
                blob = json.dumps(st.session_state.get("last_row", {}), indent=2).encode("utf-8")
                st.download_button(
                    "Download JSON",
                    data=blob,
                    file_name="latest_prediction_inputs.json",
                    mime="application/json",
                )

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
            st.error("Could not auto-select a model. Ensure your CSV has T0 and optional Y1..Y4 columns.")
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

