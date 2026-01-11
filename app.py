import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf

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
# Disclaimer (Simulated dataset + non-clinical use)
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

    # Categorical -> tf.string tensor
    for c in cat_cols:
        arr = X[c].fillna("Unknown").astype(str).to_numpy().reshape(-1, 1)
        d[c] = tf.convert_to_tensor(arr, dtype=tf.string)

    # Numeric -> tf.float32 tensor
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


# -------------------------
# Single Patient Form Builder (uses your fixed dropdown values)
# -------------------------
def add_if_in_schema(row: dict, colname: str, value, schema_cols: set):
    if colname in schema_cols:
        row[colname] = value


# -------------------------
# Tabs
# -------------------------
tab1, tab2 = st.tabs(["🧍 Single patient form", "📤 Upload CSV (batch)"])

with tab1:
    st.subheader("Single patient form (executive-friendly)")
    st.markdown(
        "<div class='small-note'>Select how many years of data you have. The app will pick the right model and use the BestF1 threshold from its metadata.</div>",
        unsafe_allow_html=True,
    )

    max_year = st.selectbox("Data available up to which year?", options=[1, 2, 3, 4], index=0, key="sp_max_year")
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

        # Create Year sections
        for y in range(1, max_year + 1):
            with st.expander(f"Year {y} inputs", expanded=(y == 1)):
                st.markdown("#### Clinical / adherence metrics")

                # ---- Numeric inputs (Year y) ----
                n1, n2, n3 = st.columns(3)
                with n1:
                    age = st.number_input(
                        f"age_Y{y}", min_value=0, max_value=120, value=35, step=1, key=f"sp_age_{y}"
                    )
                    add_if_in_schema(row, f"age_Y{y}", float(age), schema_cols)

                with n2:
                    cd4 = st.number_input(
                        f"cd4_Y{y}", min_value=0, max_value=5000, value=350, step=10, key=f"sp_cd4_{y}"
                    )
                    add_if_in_schema(row, f"cd4_Y{y}", float(cd4), schema_cols)

                with n3:
                    vl = st.number_input(
                        f"viral_load_Y{y}", min_value=0.0, value=1000.0, step=50.0, key=f"sp_vl_{y}"
                    )
                    add_if_in_schema(row, f"viral_load_Y{y}", float(vl), schema_cols)

                n4, n5, n6 = st.columns(3)
                with n4:
                    logvl = st.number_input(
                        f"log10_vl_Y{y}", min_value=0.0, value=3.0, step=0.1, key=f"sp_logvl_{y}"
                    )
                    add_if_in_schema(row, f"log10_vl_Y{y}", float(logvl), schema_cols)

                with n5:
                    wt = st.number_input(
                        f"weight_Y{y}", min_value=0.0, value=60.0, step=0.5, key=f"sp_wt_{y}"
                    )
                    add_if_in_schema(row, f"weight_Y{y}", float(wt), schema_cols)

                with n6:
                    adh = st.number_input(
                        f"adherence_prop_Y{y}", min_value=0.0, max_value=1.0, value=0.9, step=0.01, key=f"sp_adh_{y}"
                    )
                    add_if_in_schema(row, f"adherence_prop_Y{y}", float(adh), schema_cols)

                n7, n8, n9 = st.columns(3)
                with n7:
                    refill = st.number_input(
                        f"pharmacy_refill_adherence_pct_Y{y}",
                        min_value=0.0, max_value=100.0, value=90.0, step=1.0,
                        key=f"sp_refill_{y}",
                    )
                    add_if_in_schema(row, f"pharmacy_refill_adherence_pct_Y{y}", float(refill), schema_cols)

                with n8:
                    missed = st.number_input(
                        f"missed_appointments_Y{y}", min_value=0, value=0, step=1, key=f"sp_missed_{y}"
                    )
                    add_if_in_schema(row, f"missed_appointments_Y{y}", float(missed), schema_cols)

                with n9:
                    late = st.number_input(
                        f"days_late_Y{y}", min_value=0, value=0, step=1, key=f"sp_late_{y}"
                    )
                    add_if_in_schema(row, f"days_late_Y{y}", float(late), schema_cols)

                # age_baseline: usually only Year 1
                if y == 1:
                    base_age = st.number_input(
                        "age_baseline_Y1", min_value=0, max_value=120, value=35, step=1, key="sp_base_age"
                    )
                    add_if_in_schema(row, "age_baseline_Y1", float(base_age), schema_cols)

                st.markdown("#### Program / clinical context")

                # ---- REQUIRED DROPDOWNS ----
                c1, c2, c3 = st.columns(3)
                with c1:
                    gender = st.selectbox(f"gender_Y{y}", OPTIONS["gender"], key=f"sp_gender_{y}")
                    add_if_in_schema(row, f"gender_Y{y}", gender, schema_cols)

                with c2:
                    func = st.selectbox(f"functional_status_Y{y}", OPTIONS["functional_status"], key=f"sp_func_{y}")
                    add_if_in_schema(row, f"functional_status_Y{y}", func, schema_cols)

                with c3:
                    regimen_line = st.selectbox(f"regimen_line_Y{y}", OPTIONS["regimen_line"], key=f"sp_line_{y}")
                    add_if_in_schema(row, f"regimen_line_Y{y}", regimen_line, schema_cols)

                c4, c5, c6 = st.columns(3)
                with c4:
                    regimen_type = st.selectbox(f"regimen_type_Y{y}", OPTIONS["regimen_type"], key=f"sp_type_{y}")
                    add_if_in_schema(row, f"regimen_type_Y{y}", regimen_type, schema_cols)

                with c5:
                    tb = st.selectbox(f"tb_status_Y{y}", OPTIONS["tb_status"], key=f"sp_tb_{y}")
                    add_if_in_schema(row, f"tb_status_Y{y}", tb, schema_cols)

                with c6:
                    who = st.selectbox(f"who_stage_Y{y}", OPTIONS["who_stage"], key=f"sp_who_{y}")
                    add_if_in_schema(row, f"who_stage_Y{y}", who, schema_cols)

                c7, c8, c9 = st.columns(3)
                with c7:
                    state = st.selectbox(f"stateProvince_Y{y}", OPTIONS["stateProvince"], key=f"sp_state_{y}")
                    add_if_in_schema(row, f"stateProvince_Y{y}", state, schema_cols)

                with c8:
                    fac = st.selectbox(f"facilityName_Y{y}", OPTIONS["facilityName"], key=f"sp_fac_{y}")
                    add_if_in_schema(row, f"facilityName_Y{y}", fac, schema_cols)

                with c9:
                    sup = st.selectbox(f"suppressed_lt1000_Y{y}", OPTIONS["suppressed_lt1000"], key=f"sp_sup_{y}")
                    st.caption("Note: 0 = No, 1 = Yes")
                    add_if_in_schema(row, f"suppressed_lt1000_Y{y}", int(sup), schema_cols)

        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")

    if st.button("Predict (Single Patient)", type="primary", key="sp_predict_btn"):
        df1 = pd.DataFrame([row])

        # Run prediction once
        res = predict_df(df1, available[chosen_key])

        # If missing features were auto-filled, block and request completion
        missing_txt = str(res.loc[0, "missing_features_filled"] or "").strip()
        if missing_txt:
            st.error("Please complete the missing inputs before predicting.")
            st.write("Missing features detected:", missing_txt)
            st.stop()

        # Display results
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

