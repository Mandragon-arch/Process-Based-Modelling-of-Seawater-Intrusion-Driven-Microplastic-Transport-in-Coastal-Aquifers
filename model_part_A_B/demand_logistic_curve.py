# =============================================================================
# NITK CAMPUS GROUNDWATER DEMAND — LOGISTIC (S-CURVE) MODEL
# =============================================================================
#
# LOGISTIC FUNCTION PARAMETERS — ASSUMPTIONS vs DATA
# ---------------------------------------------------
# Q(t) = Q_min + (Q_max - Q_min) / (1 + exp(-k * (t - t_mid)))
#
#   Q_min  = 180  m³/day  ASSUMPTION: 30% of 0.6 MLD minimum total campus demand
#                         (60% assumed to be met by municipal supply)
#   Q_max  = 480  m³/day  ASSUMPTION: 30% of 1.2 MLD maximum total campus demand
#                         (upper bound based on current campus population ~10,000)
#   t_mid  = 2006         DERIVED: NIT upgrade year; rapid campus expansion began
#                         ~2002-2004 (NIT status), hostel construction peaked 2005-08
#   k      = 0.30         ASSUMPTION: growth-rate tuning parameter; chosen so that
#                         80% of growth occurs within ±4 years of t_mid
#   section_width = 200 m MODELLING ASSUMPTION: representative 2D cross-section
#                         width; NOT a field-measured value
#
# Data sources: NITK annual reports (enrolment), CGWB W06543 field records,
#   Varsha S (2024-25) field survey, Karnataka Water Supply Board estimates.
# =============================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# =============================================================================
# 1. PARAMETERS
# =============================================================================
Q_min          = 180.0   # m³/day  — minimum GW demand (1991 baseline)
Q_max          = 480.0   # m³/day  — maximum GW demand (2025 plateau)
t_mid          = 2006.0  # year    — inflection point (NIT upgrade era)
k              = 0.30    # 1/year  — logistic growth rate
section_width  = 200.0   # m       — 2D model section width (ASSUMPTION)

YEAR_START  = 1991
YEAR_END    = 2025
PUMPING_START = 1996     # first year of modelled active pumping

# =============================================================================
# 2. LOGISTIC FUNCTION
# =============================================================================
def Q_GW(t):
    """
    Logistic (S-curve) groundwater demand as a function of calendar year t.
    Returns demand in m³/day.
    """
    return Q_min + (Q_max - Q_min) / (1.0 + np.exp(-k * (t - t_mid)))

# =============================================================================
# 3. COMPUTE INTEGER-YEAR VALUES  →  dictionary year: Q
# =============================================================================
years_int = list(range(YEAR_START, YEAR_END + 1))
Q_dict    = {yr: Q_GW(yr) for yr in years_int}   # year → Q (m³/day)

# =============================================================================
# 4. CONSOLE TABLE
# =============================================================================
print("\n" + "="*72)
print(f"  NITK Campus Groundwater Demand — Logistic Growth Model")
print("="*72)
print(f"  {'Year':<6} {'Q_GW (m³/day)':>15} {'Q_GW (MLD)':>12} {'2D flux (m²/day)':>18}")
print(f"  {'-'*6} {'-'*15} {'-'*12} {'-'*18}")
for yr in years_int:
    Q   = Q_dict[yr]
    mld = Q / 1000.0
    q2d = Q / section_width
    marker = "  ← pumping starts" if yr == PUMPING_START else ""
    print(f"  {yr:<6} {Q:>15.2f} {mld:>12.5f} {q2d:>18.4f}{marker}")
print("="*72)
print(f"\n  Section width (modelling assumption): {section_width:.0f} m")
print(f"  Q range : {Q_dict[YEAR_START]:.1f} → {Q_dict[YEAR_END]:.1f} m³/day")
print(f"  MLD range: {Q_dict[YEAR_START]/1000:.4f} → {Q_dict[YEAR_END]/1000:.4f} MLD\n")

# =============================================================================
# 5. SMOOTH CURVE FOR PLOTTING
# =============================================================================
t_smooth = np.linspace(YEAR_START, YEAR_END, 500)
Q_smooth = Q_GW(t_smooth)

# =============================================================================
# 6. PLOT
# =============================================================================
fig, ax1 = plt.subplots(figsize=(13, 6.5))

# ── Colour palette ─────────────────────────────────────────────────────────
col_curve   = "#1f77b4"   # steel blue
col_scatter = "#1f77b4"
col_hline   = "#888888"
col_vline   = "#e37222"   # orange
col_shade   = "#d0e8ff"   # light blue

# ── Pre-pumping shaded band ─────────────────────────────────────────────────
ax1.axvspan(YEAR_START, PUMPING_START,
            facecolor=col_shade, alpha=0.55, zorder=0,
            label=f"Pre-pumping / warm-up ({YEAR_START}–{PUMPING_START-1})")

# ── Horizontal asymptote lines ──────────────────────────────────────────────
ax1.axhline(Q_min, color=col_hline, lw=1.2, ls="--", zorder=1,
            label=f"$Q_{{min}}$ = {Q_min:.0f} m³/day (0.27 MLD)")
ax1.axhline(Q_max, color=col_hline, lw=1.2, ls=":", zorder=1,
            label=f"$Q_{{max}}$ = {Q_max:.0f} m³/day (0.72 MLD)")

# ── Smooth S-curve ──────────────────────────────────────────────────────────
ax1.plot(t_smooth, Q_smooth,
         color=col_curve, lw=2.5, zorder=3, label="Logistic demand curve")

# ── Integer year scatter points ─────────────────────────────────────────────
Q_int = np.array([Q_dict[yr] for yr in years_int])
ax1.scatter(years_int, Q_int,
            color=col_scatter, s=28, zorder=4, label="Annual values")

# ── Inflection vertical line ────────────────────────────────────────────────
ax1.axvline(t_mid, color=col_vline, lw=1.5, ls="--", zorder=2,
            label=f"Inflection — NIT upgrade ({int(t_mid)})")

Q_inflection = Q_GW(t_mid)
ax1.annotate(f"Inflection (NIT upgrade)\n$Q$ = {Q_inflection:.0f} m³/day",
             xy=(t_mid, Q_inflection),
             xytext=(t_mid + 2.2, Q_inflection - 38),
             fontsize=8.5, color=col_vline,
             arrowprops=dict(arrowstyle="->", color=col_vline, lw=0.9))

# ── Label Q_min / Q_max lines ───────────────────────────────────────────────
ax1.text(YEAR_END + 0.1, Q_min - 7, f"$Q_{{min}}$ = {Q_min:.0f}",
         color=col_hline, fontsize=8, va="top")
ax1.text(YEAR_END + 0.1, Q_max + 3, f"$Q_{{max}}$ = {Q_max:.0f}",
         color=col_hline, fontsize=8, va="bottom")

# ── Axes limits and ticks ───────────────────────────────────────────────────
ax1.set_xlim(YEAR_START - 0.5, YEAR_END + 1.0)
ax1.set_ylim(200, 750)
ax1.set_xlabel("Year", fontsize=12)
ax1.set_ylabel("Groundwater demand, $Q_{GW}$ [m³/day]", fontsize=11, color="black")
ax1.tick_params(axis="y")
ax1.grid(True, linestyle="--", alpha=0.35, zorder=0)

# ── Right-hand axis (MLD) ───────────────────────────────────────────────────
ax2 = ax1.twinx()
ax2.set_ylim(200 / 1000, 750 / 1000)
ax2.set_ylabel("Groundwater demand [MLD]", fontsize=11, color="#555555")
ax2.tick_params(axis="y", labelcolor="#555555")
# Sync y-ticks explicitly so minor grid aligns
ax2.set_yticks([y / 1000 for y in ax1.get_yticks() if 200 <= y <= 750])

# ── Title ───────────────────────────────────────────────────────────────────
ax1.set_title(
    "NITK Campus Groundwater Demand — Logistic Growth Model",
    fontsize=14, fontweight="bold", pad=12,
)

# ── Assumption text box (subtitle) ─────────────────────────────────────────
assumption_text = (
    "Assumption: 60% of total campus demand met by groundwater.\n"
    "Total demand range: 0.45–1.2 MLD.  "
    "Section width: 200 m (modelling assumption, not field measured)."
)
ax1.text(0.01, 0.03, assumption_text,
         transform=ax1.transAxes,
         fontsize=8.2, va="bottom", ha="left", style="italic",
         bbox=dict(boxstyle="round,pad=0.35", facecolor="lightyellow",
                   alpha=0.85, edgecolor="#cccc88"))

# ── Legend ──────────────────────────────────────────────────────────────────
ax1.legend(loc="upper left", fontsize=9, framealpha=0.9, ncol=2)

plt.tight_layout()

# =============================================================================
# 7. SAVE
# =============================================================================
out_path = os.path.join(os.path.dirname(__file__), "demand_logistic_curve.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Plot saved → {out_path}\n")
