import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from typing import Dict, Tuple, Optional

st.set_page_config(page_title="HIV Viral Suppression Risk (Deep ANN)", layout="wide")

from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

# Works whether app.py is in root or in a subfolder
CANDIDATES = [
    APP_DIR / "models",
    APP_DIR.parent / "models",
    APP_DIR / "Models",
    APP_DIR.parent / "Models",
]

MODELS_DIR = next((p for p in CANDIDATES if p.exists() and p.is_dir()), None)

with st.expander("🔎 Debug: where am I looking for models?", expanded=True):
    st.write("Working dir (os.getcwd):", os.getcwd())
    st.write("app.py folder (APP_DIR):", str(APP_DIR))
    st.write("MODELS_DIR:", str(MODELS_DIR))

    if MODELS_DIR is not None:
        st.write("Files in MODELS_DIR:")
        st.write([p.name for p in MODELS_DIR.iterdir()])
    else:
        st.error("No models directory found from candidates.")

# =========================
# CONFIG
# =========================
MODELS_DIR = "models"  # folder in your project
SUPPORTED_MODEL_KEYS = [
    "DeepANN_Y1_to_Y2",
    "DeepANN_Y1Y2_to_Y3",
    "DeepANN_Y1Y2Y3_to_Y4",
    "DeepANN_Y1Y2Y3Y4_to_Y5",
]
OPTIONAL_BASELINE_KEY = "DeepANN_FirstVisit_to_Y1"  # optional

# A simple, practical per-year field set for the Single Patient Form.
# (These should match your dataset columns; if some are missing in your metadata, the app will ignore them.)
PER_YEAR_FIELDS_NUM = [
    "age", "cd4", "who_stage", "weight",
    "adherence_prop", "pharmacy_refill_adherence_pct",
    "missed_appointments", "days_late",
]
PER_YEAR_FIELDS_CAT = [
    "gender", "stateProvince", "facilityName",
    "functional_status", "regimen_line", "regimen_type", "tb_status",
]

# For imputation in the app when a user leaves numeric empty (real deployment: replace with train medians)
NUMERIC_IMPUTE_DEFAULT = 0.0


# =========================
# UTILITIES
# =========================
def list_available_models(models_dir: str) -> Dict[str, Dict]:
    """
    Returns dict:
      model_key -> {
        "model_path": "...keras",
        "meta_path": "...json",
        "meta": {...}
      }
    """
    out = {}
    if not os.path.isdir(models_dir):
        return out

    for fname in os.listdir(models_dir):
        if fname.endswith("_metadata.json"):
            key = fname.replace("_metadata.json", "")
            meta_path = os.path.join(models_dir, fname)
            keras_path = os.path.join(models_dir, f"{key}.keras")
            if os.path.exists(keras_path):
                try:
                    with open(meta_path, "r") as f:
                        meta = json.load(f)
                    out[key] = {"model_path": keras_path, "meta_path": meta_path, "meta": meta}
                except Exception:
                    continue
    return out


def extract_bestf1_threshold(meta: dict, fallback: float = 0.5) -> float:
    """
    Robustly pull BestF1 threshold from metadata JSON created in your Colab training.
    Supports:
      meta["best_thresholds"]["F1"]["threshold"]
    or  meta["best_thresholds"]["F1"]["threshold"] as string/float
    """
    try:
        thr = meta.get("best_thresholds", {}).get("F1", {}).get("threshold", None)
        if thr is None:
            return fallback
        return float(thr)
    except Exception:
        return fallback


@st.cache_resource
def load_model_cached(path: str):
    return tf.keras.models.load_model(path)


def to_model_input(df: pd.DataFrame, cat_cols: list, num_cols: list) -> Dict[str, np.ndarray]:
    """
    Convert a dataframe into the dict of arrays expected by the Keras model:
      each feature -> shape (N, 1)
    """
    X = {}

    for c in cat_cols:
        # ensure string
        arr = df[c].fillna("Unknown").astype(str).to_numpy()
        X[c] = arr.reshape(-1, 1)

    for c in num_cols:
        # ensure float32
        arr = pd.to_numeric(df[c], errors="coerce").astype(np.float32).to_numpy()
        # if any NaN remains -> impute default
        if np.isnan(arr).any():
            arr = np.nan_to_num(arr, nan=np.float32(NUMERIC_IMPUTE_DEFAULT))
        X[c] = arr.reshape(-1, 1)

    return X


def detect_years_present(df: pd.DataFrame) -> int:
    """
    Detect the max year present from columns like *_Y1..*_Y4
    Returns max year among [1..4] that exists in df columns.
    """
    years_found = []
    for y in [1, 2, 3, 4]:
        if any(col.endswith(f"_Y{y}") for col in df.columns):
            years_found.append(y)
    return max(years_found) if years_found else 0


def choose_model_key_for_df(df: pd.DataFrame, available_keys: set, allow_baseline=False) -> Optional[str]:
    """
    Auto-select model based on the maximum year available in the input.
    - If input contains Year4 cols => use Y1..Y4 model
    - If Year3 => use Y1..Y3
    - If Year2 => use Y1..Y2
    - If only Year1 => use Y1-only model (Y1_to_Y2)
    - Baseline/first-visit: optional, only used if allow_baseline=True and no year columns or user selects it.
    """
    max_year = detect_years_present(df)

    if max_year >= 4 and "DeepANN_Y1Y2Y3Y4_to_Y5" in available_keys:
        return "DeepANN_Y1Y2Y3Y4_to_Y5"
    if max_year == 3 and "DeepANN_Y1Y2Y3_to_Y4" in available_keys:
        return "DeepANN_Y1Y2Y3_to_Y4"
    if max_year == 2 and "DeepANN_Y1Y2_to_Y3" in available_keys:
        return "DeepANN_Y1Y2_to_Y3"
    if max_year == 1 and "DeepANN_Y1_to_Y2" in available_keys:
        return "DeepANN_Y1_to_Y2"

    # Optional baseline model
    if allow_baseline and OPTIONAL_BASELINE_KEY in available_keys:
        return OPTIONAL_BASELINE_KEY

    return None


def align_to_feature_schema(df: pd.DataFrame, feature_cols: list, cat_cols: list, num_cols: list) -> Tuple[pd.DataFrame, list]:
    """
    Ensure df has exactly the required feature columns.
    - Adds missing cols (filled with Unknown for cat, default numeric for num)
    - Drops extra cols
    Returns aligned df, and list of missing cols created.
    """
    missing = [c for c in feature_cols if c not in df.columns]
    X = df.copy()

    for c in missing:
        if c in cat_cols:
            X[c] = "Unknown"
        elif c in num_cols:
            X[c] = NUMERIC_IMPUTE_DEFAULT
        else:
            # if unknown type, treat as categorical
            X[c] = "Unknown"

    # keep only feature_cols in the exact order used in training
    X = X[feature_cols].copy()

    # ensure proper dtypes
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].fillna("Unknown").astype(str)

    for c in num_cols:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce")
            X[c] = X[c].fillna(NUMERIC_IMPUTE_DEFAULT).astype(np.float32)

    return X, missing


def predict_with_selected_model(df_in: pd.DataFrame, model_info: dict) -> pd.DataFrame:
    """
    Predict probabilities + classification using model metadata schema + BestF1 threshold.
    Returns dataframe with results appended.
    """
    meta = model_info["meta"]
    model = load_model_cached(model_info["model_path"])

    feature_cols = meta["feature_cols"]
    cat_cols = meta["cat_cols"]
    num_cols = meta["num_cols"]

    threshold = extract_bestf1_threshold(meta, fallback=0.5)

    X_aligned, missing = align_to_feature_schema(df_in, feature_cols, cat_cols, num_cols)
    X_input = to_model_input(X_aligned, cat_cols, num_cols)

    prob = model.predict(X_input, verbose=0).ravel()
    pred = (prob >= threshold).astype(int)

    out = df_in.copy()
    out["pred_prob_unsuppressed"] = prob  # interpret as risk (depends on how your y was coded)
    out["pred_class"] = pred
    out["used_threshold"] = threshold
    out["missing_features_filled"] = ", ".join(missing) if missing else ""

    return out


# =========================
# UI
# =========================
st.title("AI-Driven HIV Viral Suppression Risk (Deep ANN Model Family)")
st.caption("Single patient scoring + batch CSV scoring with auto-model selection (Y1 / Y1–Y2 / Y1–Y3 / Y1–Y4).")

available = list_available_models(MODELS_DIR)
available_keys = set(available.keys())

if not available:
    st.error(
        f"No models found in `{MODELS_DIR}`.\n\n"
        "Make sure you created a `models/` folder and placed your `.keras` and `_metadata.json` files inside."
    )
    st.stop()

with st.expander("✅ Models found", expanded=False):
    st.write(pd.DataFrame(
        [{"model_key": k, "model_path": v["model_path"], "meta_path": v["meta_path"]} for k, v in available.items()]
    ))

tab1, tab2 = st.tabs(["🧍 Single Patient Form", "📤 Upload CSV (Batch)"])


# =========================
# TAB 1: SINGLE PATIENT FORM
# =========================
with tab1:
    st.subheader("Single Patient Form")

    colA, colB = st.columns([1, 1])
    with colA:
        allow_baseline = st.checkbox("Allow baseline-only (First-visit) model if available", value=False)
    with colB:
        show_optional_years = st.checkbox("Show Year 3–4 fields", value=True)

    st.markdown("Fill **Year 1** fields at minimum. If you have Year 2/3/4, fill them too — the app will auto-select the best matching model.")

    # Build a single-row dict of inputs with Year 1..4 keys
    form_data = {}

    def year_section(year: int, expanded=True):
        with st.expander(f"Year {year} inputs", expanded=expanded):
            c1, c2, c3 = st.columns(3)

            # Numeric fields
            for i, f in enumerate(PER_YEAR_FIELDS_NUM):
                colname = f"{f}_Y{year}"
                with [c1, c2, c3][i % 3]:
                    val = st.number_input(colname, value=float(NUMERIC_IMPUTE_DEFAULT), step=1.0, format="%.3f")
                    form_data[colname] = val

            # Categorical fields
            for i, f in enumerate(PER_YEAR_FIELDS_CAT):
                colname = f"{f}_Y{year}"
                with [c1, c2, c3][(i + 1) % 3]:
                    # keep it simple; user can type freeform
                    val = st.text_input(colname, value="")
                    form_data[colname] = val if val.strip() else "Unknown"

    # Always show Year 1, Year 2. Year 3/4 optional for UI
    year_section(1, expanded=True)
    year_section(2, expanded=False)
    if show_optional_years:
        year_section(3, expanded=False)
        year_section(4, expanded=False)

    if st.button("Predict (Single Patient)", type="primary"):
        single_df = pd.DataFrame([form_data])

        # Auto-select based on which year columns are present (they are all present, but might be 'Unknown' / 0)
        # So for form, we use a smarter rule: detect if user actually provided non-default values for Year2+.
        def year_has_signal(df_row: pd.Series, year: int) -> bool:
            # If ANY non-default numeric OR categorical not 'Unknown' is present, treat as available
            has_num = False
            for f in PER_YEAR_FIELDS_NUM:
                col = f"{f}_Y{year}"
                if col in df_row and float(df_row[col]) != float(NUMERIC_IMPUTE_DEFAULT):
                    has_num = True
                    break
            has_cat = False
            for f in PER_YEAR_FIELDS_CAT:
                col = f"{f}_Y{year}"
                if col in df_row and str(df_row[col]).strip() not in ["", "Unknown", "unknown"]:
                    has_cat = True
                    break
            return has_num or has_cat

        row = single_df.iloc[0]
        max_year = 1
        if year_has_signal(row, 4): max_year = 4
        elif year_has_signal(row, 3): max_year = 3
        elif year_has_signal(row, 2): max_year = 2
        else: max_year = 1

        # Choose model key by max_year
        if max_year >= 4:
            chosen_key = "DeepANN_Y1Y2Y3Y4_to_Y5"
        elif max_year == 3:
            chosen_key = "DeepANN_Y1Y2Y3_to_Y4"
        elif max_year == 2:
            chosen_key = "DeepANN_Y1Y2_to_Y3"
        else:
            chosen_key = "DeepANN_Y1_to_Y2"

        # Fallback to baseline model only if user enabled and the chosen key isn't available
        if chosen_key not in available_keys:
            chosen_key = choose_model_key_for_df(single_df, available_keys, allow_baseline=allow_baseline)

        if not chosen_key or chosen_key not in available_keys:
            st.error("Could not auto-select a model. Check that your model files exist in /models.")
            st.stop()

        st.info(f"✅ Auto-selected model: **{chosen_key}**")

        result_df = predict_with_selected_model(single_df, available[chosen_key])

        meta = available[chosen_key]["meta"]
        thr = extract_bestf1_threshold(meta, 0.5)

        prob = float(result_df.loc[0, "pred_prob_unsuppressed"])
        pred = int(result_df.loc[0, "pred_class"])

        st.markdown("### Result")
        st.write({
            "Predicted probability (model output)": prob,
            "BestF1 threshold used": thr,
            "Predicted class (1 = at-risk / not suppressed, 0 = likely suppressed)": pred,
        })

        if result_df.loc[0, "missing_features_filled"]:
            st.warning("Some required model features were missing and were auto-filled. See `missing_features_filled` below.")

        st.dataframe(result_df)


# =========================
# TAB 2: UPLOAD CSV (BATCH)
# =========================
with tab2:
    st.subheader("Upload CSV (Batch Scoring)")
    st.markdown(
        "Upload a CSV containing **at least** the columns needed for one of the scenarios.\n\n"
        "The app will auto-select the best model based on year columns present.\n"
        "If some required columns are missing, they will be filled as **Unknown** (categorical) or **0.0** (numeric)."
    )

    uploaded = st.file_uploader("Upload CSV", type=["csv"])
    allow_baseline_batch = st.checkbox("Allow baseline-only (First-visit) model for batch (if available)", value=False)

    if uploaded is not None:
        df_up = pd.read_csv(uploaded)
        st.write("Preview:", df_up.head())

        chosen_key = choose_model_key_for_df(df_up, available_keys, allow_baseline=allow_baseline_batch)
        if not chosen_key or chosen_key not in available_keys:
            st.error(
                "Could not auto-select a model for this CSV. "
                "Make sure it includes year columns like *_Y1, *_Y2, *_Y3, *_Y4."
            )
            st.stop()

        st.info(f"✅ Auto-selected model: **{chosen_key}**")

        if st.button("Predict (Batch)", type="primary"):
            results = predict_with_selected_model(df_up, available[chosen_key])

            st.success("Done. Showing results preview:")
            st.dataframe(results.head(20))

            # Download
            out_csv = results.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download predictions CSV",
                data=out_csv,
                file_name=f"predictions_{chosen_key}.csv",
                mime="text/csv"
            )

            st.caption("Tip: The `missing_features_filled` column tells you if the uploaded CSV lacked required model inputs.")
