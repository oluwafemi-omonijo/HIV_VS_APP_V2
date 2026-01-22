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
        st.markdown("### 🔐 Access")
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
                        st.success("Premium unlocked ✅")
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
                        st.success("Admin unlocked ✅")
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
