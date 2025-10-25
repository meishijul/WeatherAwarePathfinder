# MarsGui.py — 2×2 layout, minimal controls + CSV-backed Dusty/Clear
# Rolling week: ALWAYS uses tomorrow .. +6 days (7 total).
# Reads ten_day_forecast.csv (date,label,prob_dusty,...) and maps those dates to Dusty/Clear.

import math
import os, io, json, csv as _csv
from datetime import datetime, timedelta, date as _date

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# ==== Rolling week window (starts tomorrow) ====
START_DATE = _date.today() + timedelta(days=1)  # inclusive
NUM_DAYS   = 7                                  # tomorrow .. +6

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

# ---------- Fallback heuristic (only used if no CSV/inline/JSON) ----------
def auto_dust_flags_for_range(start: _date, num_days: int, base_prob=0.35, jitter=0.10):
    rng = np.random.RandomState(start.toordinal())  # deterministic per start date
    flags = []
    for i in range(num_days):
        p = base_prob + 0.15 * math.sin(i * math.pi / 3) + rng.uniform(-jitter, jitter)
        p = min(1.0, max(0.0, p))
        flags.append(rng.rand() < p)
    return flags

# ---------- Optional inline hook (single-file model override; return 7 booleans or None) ----------
def predict_dust_week_inline():
    # Example (disabled): return [True, False, False, True, True, False, False]
    return None

# ---------- Optional JSON override loader ----------
def load_dust_flags_from_json(path="predictions.json"):
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        arr = data.get("dusty", [])
        if isinstance(arr, list) and len(arr) >= NUM_DAYS:
            return [bool(x) for x in arr[:NUM_DAYS]]
    except Exception:
        pass
    return None

# ---------- CSV loader for your AI output (uses ONLY dates in the rolling window) ----------
def _parse_bool_from_row(row, prob_threshold=0.5):
    keys = {k.lower(): k for k in row.keys()}
    # Prefer an explicit label if present
    lbl_key = keys.get("label")
    if lbl_key is not None:
        v = str(row[lbl_key]).strip().lower()
        return v in ("dust", "dusty", "storm", "dust_storm", "true", "1", "yes")
    # Otherwise use probability if present
    pkey = keys.get("prob_dusty")
    if pkey is not None:
        try:
            return float(row[pkey]) >= float(prob_threshold)
        except Exception:
            return None
    return None

def load_dust_flags_from_csv(path="ten_day_forecast.csv", prob_threshold=0.5):
    if not os.path.exists(path):
        return None

    wanted = {START_DATE + timedelta(days=i) for i in range(NUM_DAYS)}
    mapping = {}  # date -> bool
    try:
        with open(path, newline="", encoding="utf-8") as f:
            reader = _csv.DictReader(f)
            for row in reader:
                dkey = next((k for k in row if k and k.lower() in ("date", "date_label", "date_str")), None)
                if not dkey:
                    continue
                try:
                    d = datetime.fromisoformat(str(row[dkey]).strip()).date()
                except Exception:
                    continue
                if d not in wanted:
                    continue
                val = _parse_bool_from_row(row, prob_threshold=prob_threshold)
                if val is not None:
                    mapping[d] = bool(val)
    except Exception:
        return None

    # Build ordered list for the rolling week; allow gaps
    flags = [mapping.get(START_DATE + timedelta(days=i), None) for i in range(NUM_DAYS)]
    if not any(x in (True, False) for x in flags):
        return None

    # Fill gaps: forward then backward; leftover Nones -> fallback heuristic for that day
    last = None
    for i in range(NUM_DAYS):
        if flags[i] is None and last is not None:
            flags[i] = last
        else:
            last = flags[i] if flags[i] is not None else last
    nxt = None
    for i in range(NUM_DAYS - 1, -1, -1):
        if flags[i] is None and nxt is not None:
            flags[i] = nxt
        else:
            nxt = flags[i] if flags[i] is not None else nxt
    if any(x is None for x in flags):
        auto = auto_dust_flags_for_range(START_DATE, NUM_DAYS)
        flags = [auto[i] if flags[i] is None else flags[i] for i in range(NUM_DAYS)]

    return [bool(x) for x in flags]

# ---------- Unified source of truth (CSV > JSON > inline > heuristic) ----------
def get_dust_flags_for_range():
    flags = load_dust_flags_from_csv("ten_day_forecast.csv", prob_threshold=0.5)
    if flags is not None:
        return flags

    flags = load_dust_flags_from_json()
    if flags is not None:
        return flags

    try:
        flags = predict_dust_week_inline()
        if isinstance(flags, (list, tuple)) and len(flags) >= NUM_DAYS:
            return [bool(x) for x in flags[:NUM_DAYS]]
    except Exception:
        pass

    return auto_dust_flags_for_range(START_DATE, NUM_DAYS)

# ---------- UI ----------
st.set_page_config(page_title="Mars Dust Calendar (Rolling Week)", layout="wide")

end_date = START_DATE + timedelta(days=NUM_DAYS - 1)
week_label = f"{START_DATE.strftime('%b %d')}–{end_date.strftime('%b %d')}"
st.title(f"🔴 Mars 7-Day Dust Calendar — {week_label}")

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

    # Bottom-left: Week view (rolling)
    with st.container():
        st.subheader(f"This Week ({START_DATE.strftime('%b %d')} → {end_date.strftime('%b %d')})")
        dust_flags = get_dust_flags_for_range()
        days = [START_DATE + timedelta(days=i) for i in range(NUM_DAYS)]

        # Weekly summary
        n_dusty = sum(dust_flags)
        n_clear = NUM_DAYS - n_dusty
        st.metric("Weekly summary", f"{n_clear} clear / {n_dusty} dusty")

        # Date + badge list
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
        st.metric(f"Total for {week_label} (Wh)", f"{total_wh:,.0f}")
        for d, e, dusty in zip(days, daily_wh, dust_flags):
            st.write(f"**{d.strftime('%a %b %d')}** — {'Dusty' if dusty else 'Clear'}: {e:,.0f} Wh")

        # Download CSV with results
        out = io.StringIO()
        w = _csv.writer(out)
        w.writerow(["Date", "Status", "Energy_Wh"])
        for d, dusty, e in zip(days, dust_flags, daily_wh):
            w.writerow([d.isoformat(), "Dusty" if dusty else "Clear", int(e)])
        st.download_button(f"Download week energy ({week_label})", out.getvalue(),
                           f"week_energy_{START_DATE.isoformat()}_to_{end_date.isoformat()}.csv", "text/csv")

    # Bottom-right: Energy chart
    with st.container():
        st.subheader("Daily Energy Chart")
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar([d.strftime("%a %d") for d in days], daily_wh)
        ax.set_ylabel("Energy (Wh)")
        ax.grid(True, axis="y", alpha=0.3)
        st.pyplot(fig)

# Footer
st.caption("Energy ≈ (Total Area × η × derate) × [I_peak × (2/π) × daylight] × dust penalties when Dusty.")
