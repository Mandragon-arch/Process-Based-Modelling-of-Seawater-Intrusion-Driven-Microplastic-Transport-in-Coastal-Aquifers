# =============================================================================
# C3_diag.py  —  SWI PROJECT  |  NITK / Surathkal Campus Aquifer
# =============================================================================
# Diagnostic: vertical salinity profiles at selected x-distances for OND 2025.
# Also prints full depth profile data to console.
# Helps diagnose whether paleo IC is dominating vs recharge freshening
# the upper layers, and whether seasonal signal exists in shallow layers.
#
# Run after C2_extract.py. No additional dependencies.
# =============================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os
import config

print("=" * 65)
print("  C3_diag.py  —  vertical profile diagnostic")
print("=" * 65)

# =============================================================================
# LOAD
# =============================================================================
npz_path = os.path.join(config.RESULTS_DIR, "C2_extracted.npz")
d        = np.load(npz_path, allow_pickle=True)

x_centers    = d['x_centers'].astype(float)
z_mid_layers = d['z_mid_layers'].astype(float)
bots         = d['bots'].astype(float)
conc_all     = d['conc_all'].astype(float)
head_all     = d['head_all'].astype(float)
sp_years     = d['sp_years']
sp_seasons   = d['sp_seasons']

OUT = config.RESULTS_DIR

def find_idx(year, season):
    for i, (y, s) in enumerate(zip(sp_years, sp_seasons)):
        if int(y) == year and str(s) == season:
            return i
    return None

def nearest_col(x_target):
    return int(np.argmin(np.abs(x_centers - x_target)))

# =============================================================================
# DIAGNOSTIC 1 — Vertical profiles at 5 distances, OND 2025
# Columns: 75, 200, 400, 600, 900 m from coast
# =============================================================================
probe_dists = [75, 200, 400, 600, 900]
probe_cols  = [nearest_col(x) for x in probe_dists]
probe_names = [f"{x} m" for x in probe_dists]

idx_ond2025 = find_idx(2025, 'OND')
c_ond = conc_all[idx_ond2025]   # (40, 120)
h_ond = head_all[idx_ond2025]

print("\n  Vertical concentration profiles — OND 2025")
print(f"  {'Depth (m MSL)':>14}", end="")
for name in probe_names:
    print(f"  {name:>10}", end="")
print()
print("  " + "-" * (14 + 12 * len(probe_dists)))

for k in range(40):
    z = z_mid_layers[k]
    print(f"  {z:>14.1f}", end="")
    for col in probe_cols:
        val = c_ond[k, col]
        if np.isnan(val):
            print(f"  {'dry':>10}", end="")
        else:
            print(f"  {val:>10.4f}", end="")
    print()

# Summary: top 5 layer min and bottom 5 layer mean
print()
print("  Summary:")
print(f"  {'Distance':>12}  {'Top-5 min (ppt)':>16}  {'Bot-5 mean (ppt)':>17}  "
      f"{'Head L1 (m)':>12}  {'Fully saline?':>14}")
print("  " + "-" * 78)
for name, col, dist in zip(probe_names, probe_cols, probe_dists):
    top5_min  = float(np.nanmin(c_ond[:5,  col]))
    bot5_mean = float(np.nanmean(c_ond[35:, col]))
    head_l1   = float(h_ond[0, col])
    saline    = "YES" if top5_min > 1.0 else "NO"
    print(f"  {name:>12}  {top5_min:>16.4f}  {bot5_mean:>17.4f}  "
          f"  {head_l1:>10.3f}  {saline:>14}")

# =============================================================================
# DIAGNOSTIC 2 — All 4 seasons at x=200 m (OND 2025 vs other seasons 2025)
# Shows whether seasonal signal exists in shallow layers
# =============================================================================
col_200 = nearest_col(200)
print(f"\n  Seasonal vertical profiles at x≈200 m — year 2025")
print(f"  {'Depth (m MSL)':>14}  {'JF':>10}  {'MAM':>10}  {'JJAS':>10}  {'OND':>10}")
print("  " + "-" * 58)

season_concs = {}
for seas in ['JF', 'MAM', 'JJAS', 'OND']:
    idx = find_idx(2025, seas)
    season_concs[seas] = conc_all[idx][:, col_200] if idx is not None else np.full(40, np.nan)

for k in range(40):
    z = z_mid_layers[k]
    print(f"  {z:>14.1f}", end="")
    for seas in ['JF', 'MAM', 'JJAS', 'OND']:
        val = season_concs[seas][k]
        print(f"  {val:>10.4f}" if not np.isnan(val) else f"  {'dry':>10}", end="")
    print()

# =============================================================================
# DIAGNOSTIC 3 — Toe evolution: compare 1996 vs 2010 vs 2025 OND profiles
# at x=200 m — shows how salinity has changed over 30 years
# =============================================================================
col_200 = nearest_col(200)
print(f"\n  Temporal evolution at x≈200 m — OND only")
print(f"  {'Depth (m MSL)':>14}  {'1996 OND':>10}  {'2010 OND':>10}  {'2025 OND':>10}")
print("  " + "-" * 50)

year_concs = {}
for yr in [1996, 2010, 2025]:
    idx = find_idx(yr, 'OND')
    year_concs[yr] = conc_all[idx][:, col_200] if idx is not None else np.full(40, np.nan)

for k in range(40):
    z = z_mid_layers[k]
    print(f"  {z:>14.1f}", end="")
    for yr in [1996, 2010, 2025]:
        val = year_concs[yr][k]
        print(f"  {val:>10.4f}" if not np.isnan(val) else f"  {'dry':>10}", end="")
    print()

# =============================================================================
# PLOT — Vertical profiles at 5 distances, OND 2025
# =============================================================================
fig, axes = plt.subplots(1, 5, figsize=(14, 6), sharey=True)
fig.suptitle(
    "Vertical Salinity Profiles — OND 2025  |  NITK Surathkal Campus Aquifer",
    fontsize=11, fontweight='bold'
)

colors_seas = {'JF': '#8c564b', 'MAM': '#d62728', 'JJAS': '#1f77b4', 'OND': '#ff7f0e'}

for ax, col, dist in zip(axes, probe_cols, probe_dists):
    c_prof = c_ond[:, col]
    c_plot = np.where(np.isnan(c_prof), 0.0, c_prof)
    ax.plot(c_plot, z_mid_layers, color='steelblue', lw=2.0, label='OND 2025')

    # Also plot all 4 seasons 2025 for comparison
    for seas in ['JF', 'MAM', 'JJAS']:
        idx_s = find_idx(2025, seas)
        if idx_s is not None:
            c_s = np.where(np.isnan(conc_all[idx_s][:, col]), 0.0,
                           conc_all[idx_s][:, col])
            ax.plot(c_s, z_mid_layers,
                    color=colors_seas[seas], lw=1.0, ls='--',
                    alpha=0.7, label=seas)

    ax.axvline(0.5, color='black', lw=0.8, ls=':', alpha=0.5)
    ax.set_xlim(0, 36)
    ax.set_ylim(-40, 2)
    ax.set_xlabel("Salinity (ppt)", fontsize=9)
    ax.set_title(f"x = {dist} m", fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    if ax == axes[0]:
        ax.set_ylabel("Elevation (m MSL)", fontsize=9)
        ax.legend(fontsize=7.5, loc='lower right')

fig.tight_layout()
p = os.path.join(OUT, "diag_vertical_profiles.png")
fig.savefig(p, dpi=150)
plt.close(fig)
print(f"\n  Plot saved -> {p}")

# =============================================================================
# PLOT 2 — Temporal evolution: 1996 / 2010 / 2025 OND at x=200 m
# =============================================================================
fig2, ax_t = plt.subplots(figsize=(6, 6))
colors_yr = {1996: 'navy', 2010: 'steelblue', 2025: 'tomato'}
for yr in [1996, 2010, 2025]:
    c_p = np.where(np.isnan(year_concs[yr]), 0.0, year_concs[yr])
    ax_t.plot(c_p, z_mid_layers,
              color=colors_yr[yr], lw=2.0, label=f"OND {yr}")

ax_t.axvline(0.5, color='black', lw=0.8, ls=':', alpha=0.5, label='0.5 ppt')
ax_t.set_xlim(0, 36)
ax_t.set_ylim(-40, 2)
ax_t.set_xlabel("Salinity (ppt)", fontsize=10)
ax_t.set_ylabel("Elevation (m MSL)", fontsize=10)
ax_t.set_title(
    "Temporal Evolution — OND  |  x ≈ 200 m\n"
    "NITK Surathkal Campus Aquifer",
    fontsize=10, fontweight='bold'
)
ax_t.grid(True, linestyle='--', alpha=0.3)
ax_t.legend(fontsize=10)
fig2.tight_layout()
p2 = os.path.join(OUT, "diag_temporal_200m.png")
fig2.savefig(p2, dpi=150)
plt.close(fig2)
print(f"  Plot saved -> {p2}")

print("\n" + "=" * 65)
print("  C3_diag.py COMPLETE")
print("=" * 65)