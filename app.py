# app.py (Streamlit frontend only)

import json
import math
import hashlib
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import requests


# ============================================================
# Secrets expected:
#   PREMIUM_ACCESS_KEY, ADMIN_ACCESS_KEY
#   FASTAPI_BASE_URL  e.g. http://127.0.0.1:8000
#   FASTAPI_API_KEY   (optional; matches API_KEY in FastAPI env)
#   GSHEETS_WEBHOOK_URL (optional)
# ============================================================

def _secret(name: str) -> str:
    try:
        return str(st.secrets.get(name, "")).strip()
    except Exception:
        return ""

FASTAPI_BASE_URL = _secret("FASTAPI_BASE_URL") or "http://127.0.0.1:8000"
FASTAPI_API_KEY = _secret("FASTAPI_API_KEY")  # optional

def api_headers() -> dict:
    h = {"Content-Type": "application/json"}
    if FASTAPI_API_KEY:
        h["X-API-Key"] = FASTAPI_API_KEY
    return h


# -------------------------
# Access control (Premium + Admin)
# -------------------------
def is_premium() -> bool:
    required = _secret("PREMIUM_ACCESS_KEY")
    if not required:
        return True
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

        required_premium = _secret("PREMIUM_ACCESS_KEY")
        if required_premium:
            premium_input = st.text_input("Premium access code", type="password", key="premium_input")
            c1, c2 = st.columns([1, 1])
            with c1:
                if st.button("Unlock Premium"):
                    st.session_state["premium_unlocked"] = premium_input.strip() == required_premium
            with c2:
                if st.button("Lock Premium"):
                    st.session_state["premium_unlocked"] = False
        else:
            st.session_state["premium_unlocked"] = True

        st.divider()

        required_admin = _secret("ADMIN_ACCESS_KEY")
        if required_admin:
            admin_input = st.text_input("Admin access code", type="password", key="admin_input")
            c3, c4 = st.columns([1, 1])
            with c3:
                if st.button("Unlock Admin"):
                    st.session_state["admin_unlocked"] = admin_input.strip() == required_admin
            with c4:
                if st.button("Lock Admin"):
                    st.session_state["admin_unlocked"] = False
        else:
            st.session_state["admin_unlocked"] = True

        st.divider()
        st.markdown("### API")
        st.write({"FASTAPI_BASE_URL": FASTAPI_BASE_URL})
        st.write({"premium": is_premium(), "admin": is_admin()})


# -------------------------
# UI options
# -------------------------
OPTIONS = {
    "who_stage": [1, 2, 3, 4],
    "suppressed_lt1000": [0, 1],
    "functional_status": ["Ambulatory", "Bedridden", "Working"],
    "regimen_line": ["1st-line", "2nd-line"],
    "regimen_type": ["AZT/3TC/NVP", "TDF/3TC/DTG", "TDF/3TC/EFV", "AZT/3TC/LPV/r"],
    "tb_status": ["History of TB", "Active TB", "No TB"],
    "gender": ["Male", "Female"],
}

MODEL_ALIAS = {
    "DeepANN_T0_to_Y1": "Baseline → Year 1 Risk Model (v1)",
    "DeepANN_T0Y1_to_Y2": "Baseline+Y1 → Year 2 Risk Model (v1)",
    "DeepANN_T0Y1Y2_to_Y3": "Baseline+Y1+Y2 → Year 3 Risk Model (v1)",
    "DeepANN_T0Y1Y2Y3_to_Y4": "Baseline+Y1+Y2+Y3 → Year 4 Risk Model (v1)",
    "DeepANN_T0Y1Y2Y3Y4_to_Y5": "Baseline+Y1–Y4 → Year 5 Risk Model (v1)",
}

VAR_HELP = {
    "age": "Client age in completed years (0–120).",
    "cd4": "CD4 cell count (cells/mm³).",
    "viral_load": "Viral load in copies/mL (must be ≥ 0). log10(VL) is computed automatically when VL > 0.",
    "log10_vl": "Auto-calculated as log10(viral_load).",
    "weight": "Weight in kilograms (kg).",
    "missed_appointments": "Number of missed appointments within the assessment period.",
    "days_late": "Total number of days late for visits/pickups within the assessment period.",
    "days_in_period": "Assessment window length used for refill adherence (e.g., 30/60/90 days).",
    "days_covered": "Number of days the client had ART available within the assessment period (from refill records).",
    "pharmacy_refill_adherence_pct": "Auto-calculated as (Days covered ÷ Days in period) × 100, capped 0–100%.",
    "adherence_prop": "Auto-calculated as pharmacy_refill_adherence_pct ÷ 100 (0–1).",
    "gender": "Sex recorded at enrollment.",
    "functional_status": "Client functional status.",
    "regimen_line": "ART regimen line.",
    "regimen_type": "ART regimen type.",
    "tb_status": "TB status.",
    "who_stage": "WHO clinical stage (1–4).",
    "stateProvince": "State/Province where the client receives care.",
    "facilityName": "Facility where the client receives care.",
    "suppressed_lt1000": "Whether VL is suppressed below 1000 (0=No, 1=Yes).",
}


# ============================================================
# Google Sheets logging (optional)
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
        return False, "GSHEETS_WEBHOOK_URL not configured."
    try:
        r = requests.post(url, json=payload, timeout=12)
        r.raise_for_status()
        return True, "Saved to Google Sheet."
    except Exception as e:
        return False, f"Sheet save failed: {e}"


# ============================================================
# Backend helpers
# ============================================================
def horizon_from_max_year(max_year: int) -> str:
    # max_year: 0..4 maps to Y1..Y5
    return {0: "Y1", 1: "Y2", 2: "Y3", 3: "Y4", 4: "Y5"}[int(max_year)]

def model_alias_from_horizon(h: str) -> str:
    tag = {
        "Y1": "DeepANN_T0_to_Y1",
        "Y2": "DeepANN_T0Y1_to_Y2",
        "Y3": "DeepANN_T0Y1Y2_to_Y3",
        "Y4": "DeepANN_T0Y1Y2Y3_to_Y4",
        "Y5": "DeepANN_T0Y1Y2Y3Y4_to_Y5",
    }[h]
    return MODEL_ALIAS.get(tag, "HIV Risk Model")

def call_api_predict(features: dict, horizon: str, threshold_strategy: str = "youden", custom_threshold=None) -> dict:
    url = f"{FASTAPI_BASE_URL.rstrip('/')}/predict"
    payload = {
        "horizon": horizon,
        "threshold_strategy": threshold_strategy,
        "custom_threshold": custom_threshold,
        "features": features,
    }
    try:
        r = requests.post(url, headers=api_headers(), json=payload, timeout=20)
        if r.status_code >= 400:
            raise Exception(r.text)
        return r.json()
    except Exception as e:
        raise RuntimeError(f"FastAPI /predict failed: {e}")

def call_api_predict_batch(rows: list[dict], horizon: str, threshold_strategy: str = "youden", custom_threshold=None) -> dict:
    url = f"{FASTAPI_BASE_URL.rstrip('/')}/predict_batch"
    payload = {
        "horizon": horizon,
        "threshold_strategy": threshold_strategy,
        "custom_threshold": custom_threshold,
        "rows": rows,
    }
    try:
        r = requests.post(url, headers=api_headers(), json=payload, timeout=60)
        if r.status_code >= 400:
            raise Exception(r.text)
        return r.json()
    except Exception as e:
        raise RuntimeError(f"FastAPI /predict_batch failed: {e}")


# ============================================================
# Agent tools (stay on Streamlit side)
# ============================================================
def time_suffix_for_schema(schema_cols: set, timepoint: int) -> str:
    # UI just builds keys; backend will fill missing anyway.
    # Keep your T0 vs Y0 behavior if needed.
    if timepoint == 0:
        has_t0 = any(str(c).endswith("_T0") for c in schema_cols)
        return "T0" if has_t0 else "Y0"
    return f"Y{timepoint}"

def col(schema_cols: set, base: str, timepoint: int) -> str:
    suf = time_suffix_for_schema(schema_cols, timepoint)
    return f"{base}_{suf}"

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
                warnings.append(f"{vl_key}: VL is 0 → log10_vl will be 0.000 (check missing).")

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
        lines.append(f"- CD4: **{cd4}**. Interpretation depends on context and timing.")
    lines.append("\n*Decision-support only; not a diagnosis or treatment recommendation.*")
    return "\n".join(lines)

def render_results_dashboard(prob: float, pred: int, thr: float, model_name: str):
    risk_label = "At-risk (likely NOT suppressed)" if pred == 1 else "Likely suppressed"
    risk_badge = "HIGH RISK" if pred == 1 else "LOW RISK"

    st.markdown("## Results dashboard")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Unsuppressed risk (prob.)", f"{prob:.3f}")
    with c2:
        st.metric("Threshold used", f"{thr:.3f}")
    with c3:
        st.metric("Prediction", risk_badge)
    with c4:
        st.metric("Model", model_name)

    st.markdown(f"**Interpretation:** {risk_label}")
    st.progress(min(max(prob, 0.0), 1.0))


# ============================================================
# UI
# ============================================================
st.set_page_config(page_title="HIV Viral Suppression Predictor", layout="wide")
render_access_sidebar()

st.title("HIV Viral Suppression Risk")
st.caption("Streamlit UI → FastAPI brain (models + validation + prediction)")

# Quick API check
with st.sidebar:
    if st.button("Ping API (/health)"):
        try:
            r = requests.get(f"{FASTAPI_BASE_URL.rstrip('/')}/health", headers=api_headers(), timeout=10)
            st.write(r.json())
        except Exception as e:
            st.error(f"API ping failed: {e}")

tab1, tab2 = st.tabs(["🧍 Single patient form", "📤 Upload CSV (batch)"])


# ===========================
# SINGLE PATIENT
# ===========================
with tab1:
    st.subheader("Single patient form")

    max_year = st.selectbox(
        "Data available up to which timepoint?",
        options=[0, 1, 2, 3, 4],
        index=1,
        help="Baseline=0, Y1..Y4. Backend horizon becomes Y1..Y5.",
    )
    horizon = horizon_from_max_year(max_year)

    # We keep a schema_cols placeholder for your agent tools (since no local meta now).
    # In v2, we can fetch actual schema_cols from FastAPI if you want.
    schema_cols = set()  # optional; agent tools still work for keys that exist in row

    threshold_strategy = st.selectbox(
        "Threshold strategy",
        options=["youden", "f1", "roc_top_left", "custom"],
        index=0,
    )
    custom_threshold = None
    if threshold_strategy == "custom":
        custom_threshold = st.slider("Custom threshold", 0.0, 1.0, 0.5, 0.01)

    st.info(f"Backend horizon selected: **{horizon}** | Model label: **{model_alias_from_horizon(horizon)}**")

    row = {}

    for y in range(0, max_year + 1):
        suf = "T0" if y == 0 else f"Y{y}"
        title = "Baseline (T0) inputs" if y == 0 else f"Year {y} inputs"

        with st.expander(title, expanded=(y in [0, 1])):
            n1, n2, n3 = st.columns(3)
            with n1:
                row[f"age_{suf}"] = float(st.number_input(f"age_{suf}", 0, 120, 35, 1, help=VAR_HELP["age"]))
            with n2:
                row[f"cd4_{suf}"] = float(st.number_input(f"cd4_{suf}", 0, 5000, 350, 10, help=VAR_HELP["cd4"]))
            with n3:
                vl = float(st.number_input(f"viral_load_{suf}", min_value=0.0, value=1000.0, step=50.0, help=VAR_HELP["viral_load"]))
                row[f"viral_load_{suf}"] = vl

            logvl = round(math.log10(vl), 3) if vl > 0 else 0.0
            row[f"log10_vl_{suf}"] = float(logvl)
            st.caption(f"log10_vl_{suf} auto = {logvl:.3f}")

            cA, cB, cC = st.columns(3)
            with cA:
                days_in_period = st.number_input(f"Days in assessment period ({suf})", 1, 365, 30, 1, help=VAR_HELP["days_in_period"])
            with cB:
                days_covered = st.number_input(f"Days covered by refills ({suf})", 0, 365, 0, 1, help=VAR_HELP["days_covered"])
            pct = (days_covered / days_in_period) * 100.0 if days_in_period > 0 else 0.0
            pct = round(max(0.0, min(100.0, pct)), 2)
            prop = round(pct / 100.0, 4)
            with cC:
                st.write(f"adherence %: {pct:.2f}%")
                st.write(f"adherence prop: {prop:.4f}")

            row[f"pharmacy_refill_adherence_pct_{suf}"] = float(pct)
            row[f"adherence_prop_{suf}"] = float(prop)
            row[f"weight_{suf}"] = float(st.number_input(f"weight_{suf}", min_value=0.0, value=60.0, step=0.5, help=VAR_HELP["weight"]))
            row[f"missed_appointments_{suf}"] = float(st.number_input(f"missed_appointments_{suf}", min_value=0, value=0, step=1, help=VAR_HELP["missed_appointments"]))
            row[f"days_late_{suf}"] = float(st.number_input(f"days_late_{suf}", min_value=0, value=0, step=1, help=VAR_HELP["days_late"]))

            c1, c2, c3 = st.columns(3)
            with c1:
                row[f"gender_{suf}"] = st.selectbox(f"gender_{suf}", OPTIONS["gender"], help=VAR_HELP["gender"])
            with c2:
                row[f"functional_status_{suf}"] = st.selectbox(f"functional_status_{suf}", OPTIONS["functional_status"], help=VAR_HELP["functional_status"])
            with c3:
                row[f"regimen_line_{suf}"] = st.selectbox(f"regimen_line_{suf}", OPTIONS["regimen_line"], help=VAR_HELP["regimen_line"])

            c4, c5, c6 = st.columns(3)
            with c4:
                row[f"regimen_type_{suf}"] = st.selectbox(f"regimen_type_{suf}", OPTIONS["regimen_type"], help=VAR_HELP["regimen_type"])
            with c5:
                row[f"tb_status_{suf}"] = st.selectbox(f"tb_status_{suf}", OPTIONS["tb_status"], help=VAR_HELP["tb_status"])
            with c6:
                row[f"who_stage_{suf}"] = st.selectbox(f"who_stage_{suf}", OPTIONS["who_stage"], help=VAR_HELP["who_stage"])

            c7, c8, c9 = st.columns(3)
            with c7:
                row[f"stateProvince_{suf}"] = st.text_input(f"stateProvince_{suf}", value="", help=VAR_HELP["stateProvince"])
            with c8:
                row[f"facilityName_{suf}"] = st.text_input(f"facilityName_{suf}", value="", help=VAR_HELP["facilityName"])
            with c9:
                row[f"suppressed_lt1000_{suf}"] = int(st.selectbox(f"suppressed_lt1000_{suf}", OPTIONS["suppressed_lt1000"], help=VAR_HELP["suppressed_lt1000"]))

    if st.button("Predict (Single Patient)", type="primary"):
        # Optional: basic agent validation before sending
        if "schema_cols" in st.session_state:
            schema_cols = st.session_state["schema_cols"]

        try:
            api_res = call_api_predict(
                features=row,
                horizon=horizon,
                threshold_strategy=threshold_strategy,
                custom_threshold=custom_threshold,
            )
        except Exception as e:
            st.error(str(e))
            st.stop()

        prob = float(api_res["probability"])
        pred = int(api_res["predicted_class"])
        thr = float(api_res["threshold_used"])
        missing_filled = api_res.get("missing_features_filled", "")

        st.session_state["last_row"] = row
        st.session_state["last_prob"] = prob
        st.session_state["last_pred"] = pred
        st.session_state["last_thr"] = thr
        st.session_state["last_horizon"] = horizon
        st.session_state["missing_features_filled"] = missing_filled
        st.session_state["last_max_year"] = int(max_year)
        st.session_state["last_schema_cols"] = set(row.keys())  # for agent functions to reference keys present

        # Save log record (same as your previous logic)
        record = flatten_for_sheet({
            **row,
            "entry_id": "",
            "timestamp": _now_iso(),
            "horizon": horizon,
            "model_alias": model_alias_from_horizon(horizon),
            "pred_prob_unsuppressed": prob,
            "pred_class": pred,
            "threshold_used": thr,
            "missing_features_filled": missing_filled,
        })
        record["entry_id"] = make_entry_id(record)

        st.session_state.setdefault("retrain_log", [])
        st.session_state["retrain_log"].append(record)

        if sheets_enabled():
            ok, msg = append_to_gsheet(record)
            st.success(msg) if ok else st.warning(msg)

    # Results
    if "last_prob" in st.session_state:
        render_results_dashboard(
            st.session_state["last_prob"],
            st.session_state["last_pred"],
            st.session_state["last_thr"],
            model_alias_from_horizon(st.session_state["last_horizon"]),
        )

        if st.session_state.get("missing_features_filled"):
            st.warning("Backend auto-filled missing features:")
            st.write(st.session_state["missing_features_filled"])

        st.markdown("## Agent tools (Premium)")
        if not is_premium():
            st.info("Agent tools are Premium-only. Unlock in the sidebar.")
        else:
            if st.button("1) Validation"):
                schema_cols = st.session_state["last_schema_cols"]
                v = agent_validate_row(st.session_state["last_row"], schema_cols, st.session_state["last_max_year"])
                st.session_state["agent_validation"] = v
                if v["errors"]:
                    st.error("\n".join(v["errors"]))
                elif v["warnings"]:
                    st.warning("\n".join(v["warnings"]))
                else:
                    st.success("Validation passed.")

            if st.button("2) Explanation"):
                schema_cols = st.session_state["last_schema_cols"]
                exp = agent_template_explanation(
                    st.session_state["last_row"],
                    schema_cols,
                    st.session_state["last_prob"],
                    st.session_state["last_pred"],
                    st.session_state["last_thr"],
                    st.session_state["last_max_year"],
                )
                st.markdown(exp)

        st.markdown("## Download entries (Admin)")
        if not is_admin():
            st.caption("Downloads are Admin-only.")
        else:
            if st.session_state.get("retrain_log"):
                df_all = pd.DataFrame(st.session_state["retrain_log"])
                st.download_button(
                    "Download ALL session prediction rows (CSV)",
                    data=df_all.to_csv(index=False).encode("utf-8"),
                    file_name=f"hiv_session_predictions_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )


# ===========================
# BATCH CSV
# ===========================
with tab2:
    st.subheader("Batch scoring (Upload CSV)")
    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    horizon = st.selectbox("Horizon for batch", options=["Y1", "Y2", "Y3", "Y4", "Y5"], index=4)
    threshold_strategy = st.selectbox("Threshold strategy (batch)", options=["youden", "f1", "roc_top_left", "custom"], index=0)
    custom_threshold = None
    if threshold_strategy == "custom":
        custom_threshold = st.slider("Custom threshold (batch)", 0.0, 1.0, 0.5, 0.01)

    if uploaded is not None:
        df = pd.read_csv(uploaded)
        st.write("Preview:", df.head())

        if st.button("Run batch predictions", type="primary"):
            try:
                payload_rows = df.to_dict(orient="records")
                api_res = call_api_predict_batch(payload_rows, horizon, threshold_strategy, custom_threshold)
                res_df = pd.DataFrame(api_res["results"])
                st.dataframe(res_df.head(50))

                st.download_button(
                    "Download predictions CSV",
                    data=res_df.to_csv(index=False).encode("utf-8"),
                    file_name=f"predictions_{horizon}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )
            except Exception as e:
                st.error(str(e))


