# MarsGui.py — Single-file app: minimal controls + auto Dusty/Clear + inline model/JSON override
import math
import io, csv, json, os
from datetime import datetime, timedelta

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ---------- Fixed background constants (kept out of UI) ----------
DERATE        = 0.90     # wiring/inverter loss
I_PEAK        = 600.0    # W/m² clear-sky noon irradiance
DAYLIGHT_H    = 12.0     # hours of daylight
ALPHA_ATM     = 0.60     # atmospheric loss on dusty days (fraction)
BETA_SOILING  = 0.30     # panel soiling loss on dusty days (fraction)

# ---------- Core math ----------
def daily_energy_wh(is_dusty: bool, area_m2: float, eta: float) -> float:
    """Daily energy (Wh) with fixed environment and dust penalties."""
    # Approximate daily insolation using a sine-shaped daylight curve:
    ins_clear_Wh_m2 = I_PEAK * (2.0 / math.pi) * DAYLIGHT_H
    T_atm   = 1.0 - ALPHA_ATM    if is_dusty else 1.0
    T_panel = 1.0 - BETA_SOILING if is_dusty else 1.0
    return area_m2 * eta * DERATE * ins_clear_Wh_m2 * T_atm * T_panel

# ---------- Auto "dustiness" heuristic (fallback) ----------
def auto_dust_flags_for_week(base_prob: float = 0.35, jitter: float = 0.10):
    """Return 7 booleans (Dusty=True) for today..+6 using a smooth probability pattern."""
    today = datetime.now().date()
    rng = np.random.RandomState(today.toordinal())  # deterministic per day
    flags = []
    for i in range(7):
        p = base_prob + 0.15 * math.sin(i * math.pi / 3) + rng.uniform(-jitter, jitter)
        p = min(1.0, max(0.0, p))
        flags.append(rng.rand() < p)
    return flags

# ---------- Inline model hook (SINGLE-FILE OVERRIDE) ----------
def predict_dust_week_inline():
    """
    RETURN exactly 7 booleans for today..+6 (True=Dusty, False=Clear),
    OR return None to skip. Replace the body with your real model call.
    """
    # Example (disabled): return [True, False, False, True, False, True, False]
    return None

# ---------- JSON override loader (SINGLE-FILE, no import needed) ----------
def load_dust_flags_from_json(path="predictions.json"):
    """
    If a JSON file is present with {"dusty":[true/false x7]}, use it.
    Return list[bool] or None if missing/invalid.
    """
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        arr = data.get("dusty", [])
        if not isinstance(arr, list) or len(arr) < 7:
            return None
        return [bool(x) for x in arr[:7]]
    except Exception:
        return None

# ---------- Unified source of truth for dust flags ----------
def get_dust_flags_for_week():
    """
    Priority:
      1) predictions.json (if present & valid)
      2) predict_dust_week_inline() if it returns 7 booleans
      3) auto_dust_flags_for_week() fallback
    """
    flags = load_dust_flags_from_json()
    if flags is not None:
        return flags
    try:
        flags = predict_dust_week_inline()
        if isinstance(flags, (list, tuple)) and len(flags) >= 7:
            return [bool(x) for x in flags[:7]]
    except Exception:
        pass
    return auto_dust_flags_for_week(base_prob=0.35, jitter=0.10)

# ---------- UI ----------
st.set_page_config(page_title="Mars Dust Calendar", layout="wide")
st.title("🔴 Mars 7-Day Dust Calendar")

left_col, right_col = st.columns(2)

# ===== Left column =====
with left_col:
    # Top-left: Solar panel stats (only controls)
    with st.container():
        st.subheader("Solar Panels")
        c1, c2, c3 = st.columns([1, 1, 1])
        with c1:
            n_panels = st.number_input("Number of panels", 1, 5000, 12, 1)
        with c2:
            area_per_panel = st.number_input("Area per panel (m²)", 0.1, 20.0, 1.6, 0.1, format="%.2f")
        with c3:
            eta = st.slider("Efficiency η", 0.05, 0.30, 0.20, 0.01)

        total_area_m2 = float(n_panels) * float(area_per_panel)
        st.metric("Total array area", f"{total_area_m2:.2f} m²")

    # Bottom-left: Week view (auto or overridden)
    with st.container():
        st.subheader("This Week")
        dust_flags = get_dust_flags_for_week()
        today = datetime.now().date()
        days = [today + timedelta(days=i) for i in range(7)]

        # Weekly summary
        n_dusty = sum(dust_flags)
        n_clear = 7 - n_dusty
        st.metric("Weekly summary", f"{n_clear} clear / {n_dusty} dusty")

        # Date + badge list (serious looking pills)
        for d, dusty in zip(days, dust_flags):
            label = "Dusty" if dusty else "Clear"
            color = "#eab308" if dusty else "#22c55e"   # amber / green
            pill = (
                f'<span style="padding:2px 8px;border-radius:9999px;'
                f'background:{color};color:#0e1117;font-weight:600;'
                f'font-size:0.85rem;">{label}</span>'
            )
            st.markdown(f"**{d.strftime('%a %b %d')}** — {pill}", unsafe_allow_html=True)

# ===== Right column =====
with right_col:
    # Compute once for right-side panels
    daily_wh = [daily_energy_wh(dusty, total_area_m2, eta) for dusty in dust_flags]
    total_wh = float(np.sum(daily_wh))

    # Top-right: Daily Energy + total
    with st.container():
        st.subheader("Daily Energy (Wh)")
        st.metric("Total for 7 days (Wh)", f"{total_wh:,.0f}")
        for d, e, dusty in zip(days, daily_wh, dust_flags):
            st.write(f"**{d.strftime('%a %b %d')}** — {'Dusty' if dusty else 'Clear'}: {e:,.0f} Wh")

        # Download CSV (handy for judges)
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["Date", "Status", "Energy_Wh"])
        for d, dusty, e in zip(days, dust_flags, daily_wh):
            w.writerow([d.isoformat(), "Dusty" if dusty else "Clear", int(e)])
        st.download_button("Download CSV", buf.getvalue(), "week_energy.csv", "text/csv")

    # Bottom-right: Energy chart
    with st.container():
        st.subheader("Daily Energy Chart")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar([d.strftime("%a") for d in days], daily_wh)
        ax.set_ylabel("Energy (Wh)")
        ax.grid(True, axis="y", alpha=0.3)
        st.pyplot(fig)

# Footer
st.caption("Energy ≈ (Total Area × η × derate) × [I_peak × (2/π) × daylight] × (dust penalties when Dusty).")
