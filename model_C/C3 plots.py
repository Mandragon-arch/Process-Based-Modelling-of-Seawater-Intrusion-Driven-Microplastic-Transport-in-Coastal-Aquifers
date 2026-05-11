# =============================================================================
# C3_plots.py  —  SWI PROJECT  |  NITK / Surathkal Campus Aquifer
# =============================================================================
# Four thesis-quality plots from C2_extracted.npz.
#
# Plot 1 — Seasonal wedge cross-sections (2016 and 2025)
#           8 panels (4 seasons × 2 years), shared colorbar
#           Salinity field + 0.5 ppt contour + toe marker
#           x-axis 0–500 m (active wedge zone)
#
# Plot 2 — Wedge toe migration 1996–2025
#           4 seasonal lines, field data window marked
#
# Plot 3 — Salinity transect vs field data (OND 2025)
#           Model min-of-top-5-layers profile + field well scatter
#           NOTE on extraction: model domain top = 0 m MSL; real dug wells
#           sit on ground 6–24 m MSL and sample a freshwater lens that exists
#           above the model domain. Min of top 5 layers is the shallowest
#           freshwater value the model can represent for comparison.
#
# Plot 4 — Model vs field data comparison table
#           With Cl/HCO3 column; colour-coded agreement rows
#
# Run after: C2_extract.py
# Outputs saved to: config.RESULTS_DIR
# =============================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import config

print("=" * 65)
print("  C3_plots.py  —  generating plots")
print("=" * 65)

# =============================================================================
# LOAD
# =============================================================================
npz_path = os.path.join(config.RESULTS_DIR, "C2_extracted.npz")
if not os.path.isfile(npz_path):
    raise FileNotFoundError(
        f"C2_extracted.npz not found at:\n  {npz_path}\nRun C2_extract.py first."
    )

d = np.load(npz_path, allow_pickle=True)

x_centers    = d['x_centers'].astype(float)
z_mid_layers = d['z_mid_layers'].astype(float)
tops         = d['tops'].astype(float)
bots         = d['bots'].astype(float)
conc_all     = d['conc_all'].astype(float)    # (120, 40, 120)
sp_years     = d['sp_years']
sp_seasons   = d['sp_seasons']
toe_years    = d['toe_years']
toe_JF       = d['toe_JF'].astype(float)
toe_MAM      = d['toe_MAM'].astype(float)
toe_JJAS     = d['toe_JJAS'].astype(float)
toe_OND      = d['toe_OND'].astype(float)
fw_names     = d['fw_names']
fw_dist      = d['fw_dist'].astype(float)
fw_sal       = d['fw_sal'].astype(float)
fw_type      = d['fw_type']
fw_model     = d['fw_model'].astype(float)
fw_notes     = d['fw_notes']
fw_Cl        = d['fw_Cl'].astype(float)
fw_HCO3      = d['fw_HCO3'].astype(float)
fw_ratio     = d['fw_ratio'].astype(float)
fw_ec        = d['fw_ec'].astype(float)

nlay = conc_all.shape[1]
ncol = conc_all.shape[2]
OUT  = config.RESULTS_DIR
os.makedirs(OUT, exist_ok=True)

# =============================================================================
# GRID EDGES FOR PCOLORMESH
# =============================================================================
dx      = np.diff(x_centers)
x_edges = np.concatenate([[x_centers[0] - dx[0] / 2],
                           x_centers[:-1] + dx / 2,
                           [x_centers[-1] + dx[-1] / 2]])
z_edges = np.concatenate([[tops[0]], bots])   # (nlay+1,)

# =============================================================================
# STYLE
# =============================================================================
CMAP           = plt.cm.RdYlBu_r
NORM           = mcolors.LogNorm(vmin=0.05, vmax=35.0)
SEASON_COLORS  = config.SEASON_COLORS
SEASON_MARKERS = config.SEASON_MARKERS
SEASON_OFFSET  = config.SEASON_OFFSET
SEASON_LABELS  = {'JF':   'Jan–Feb',  'MAM':  'Mar–May',
                  'JJAS': 'Jun–Sep',  'OND':  'Oct–Dec'}

# =============================================================================
# HELPERS
# =============================================================================
def find_idx(year, season):
    for i, (y, s) in enumerate(zip(sp_years, sp_seasons)):
        if int(y) == year and str(s) == season:
            return i
    return None


def get_toe(c2d):
    """Most inland column where mean of bottom 5 layers >= 0.5 ppt."""
    base = np.nanmean(c2d[35:, :], axis=0)
    cols = np.where(base >= 0.5)[0]
    return float(x_centers[cols[-1]]) if len(cols) > 0 else 0.0


def draw_panel(ax, c2d, title, xlim=1000):
    """Salinity pcolormesh + 0.5 ppt contour + toe marker."""
    from matplotlib.lines import Line2D
    c_plot = np.where(np.isnan(c2d), 0.0, c2d)
    pcm = ax.pcolormesh(x_edges, z_edges, c_plot,
                        cmap=CMAP, norm=NORM, shading='flat')
    try:
        ax.contour(x_centers, z_mid_layers, c_plot,
                   levels=[0.5], colors='black',
                   linewidths=1.0, linestyles='--')
    except Exception:
        pass
    toe = get_toe(c2d)
    if 0 < toe <= xlim:
        ax.axvline(toe, color='black', lw=0.8, ls=':', alpha=0.7)
    ax.set_xlim(0, xlim)
    ax.set_ylim(-40, 2)
    ax.set_title(title, fontsize=8.5, pad=3)
    return pcm


# =============================================================================
# PLOT 1 — Seasonal wedge cross-sections: 2016 and 2025
# Rows = seasons, Cols = years
# =============================================================================
print("  Plot 1 — seasonal wedge cross-sections ...")

fig1, axes1 = plt.subplots(4, 2, figsize=(11, 13),
                            sharex=True, sharey=True)
fig1.suptitle(
    "Seasonal Saltwater Wedge  |  NITK Surathkal Campus Aquifer\n"
    "Dashed contour: 0.5 ppt  |  Dotted line: wedge toe  |  x: 0-1000 m from coast",
    fontsize=10, fontweight='bold'
)

pcm_ref = None
for col, yr in enumerate([2016, 2025]):
    for row, seas in enumerate(['JF', 'MAM', 'JJAS', 'OND']):
        ax  = axes1[row, col]
        idx = find_idx(yr, seas)
        if idx is None:
            ax.text(500, -20, "not found", ha='center', fontsize=8, color='red')
            continue
        c2d  = conc_all[idx]
        ghb  = config.get_ghb_head(yr, seas)
        toe  = get_toe(c2d)
        title = (f"{yr}  {SEASON_LABELS[seas]}\n"
                 f"GHB = {ghb:.2f} m   Toe = {toe:.0f} m")
        pcm_ref = draw_panel(ax, c2d, title, xlim=1000)

for row in range(4):
    axes1[row, 0].set_ylabel("Elevation (m MSL)", fontsize=8)
for col in range(2):
    axes1[3, col].set_xlabel("Distance from coast (m)", fontsize=9)

fig1.subplots_adjust(right=0.85, hspace=0.40, wspace=0.10, top=0.91)
cbar_ax = fig1.add_axes([0.87, 0.08, 0.018, 0.82])
cb = fig1.colorbar(pcm_ref, cax=cbar_ax, extend='both')
cb.set_label("Salinity (ppt)", fontsize=9)
cb.set_ticks([0.05, 0.1, 0.5, 1, 5, 10, 35])
cb.set_ticklabels(['0.05', '0.1', '0.5', '1', '5', '10', '35'])

p1 = os.path.join(OUT, "plot1_wedge_sections.png")
fig1.savefig(p1, dpi=150, bbox_inches='tight')
plt.close(fig1)
print(f"    Saved -> {p1}")

# =============================================================================
# PLOT 2 — Wedge toe migration 1996–2025
# =============================================================================
print("  Plot 2 — toe migration timeline ...")

fig2, ax2 = plt.subplots(figsize=(12, 5))

for seas, toe_vals in [('JF',   toe_JF),
                        ('MAM',  toe_MAM),
                        ('JJAS', toe_JJAS),
                        ('OND',  toe_OND)]:
    t_x = toe_years + SEASON_OFFSET[seas]
    ax2.plot(t_x, toe_vals,
             color=SEASON_COLORS[seas],
             marker=SEASON_MARKERS[seas],
             markersize=4, lw=1.5,
             label=SEASON_LABELS[seas])

# Field data window
ax2.axvline(2024 + SEASON_OFFSET['OND'], color='green',
            lw=1.5, ls='--', label='Field data (OND 2024–25)')

ax2.set_xlabel("Year", fontsize=11)
ax2.set_ylabel("Wedge toe — distance from coast (m)", fontsize=11)
ax2.set_title(
    "Seasonal Wedge Toe Migration  |  NITK Surathkal Campus Aquifer  |  1996–2025",
    fontsize=11, fontweight='bold'
)
ax2.set_xlim(1995, 2026)
ax2.set_ylim(bottom=0)
ax2.grid(True, linestyle='--', alpha=0.4)
ax2.legend(fontsize=10, loc='upper left', framealpha=0.9)
fig2.tight_layout()

p2 = os.path.join(OUT, "plot2_toe_timeline.png")
fig2.savefig(p2, dpi=150)
plt.close(fig2)
print(f"    Saved -> {p2}")

# =============================================================================
# PLOT 3 — Salinity transect vs field data (OND 2025)
#
# Model extraction: min of top 5 layers (indices 0-4, elevations 0 to -5 m MSL)
# This is the shallowest freshwater value the model can represent.
#
# Limitation note: model domain top = 0 m MSL. Dug wells at 75–310 m sit on
# ground 6–17 m MSL and sample a freshwater lens that exists above the model
# domain. The model cannot represent this lens; layer 1 at these columns is
# already within the saltwater wedge. The min-top-5 extraction gives the
# most favourable model value possible for comparison at near-coast wells.
# This limitation is documented in the thesis and in the comparison table.
# =============================================================================
print("  Plot 3 — transect vs field data ...")

idx_ond = find_idx(2025, 'OND')
c_ond   = conc_all[idx_ond]   # (40, 120)

# Min of top 5 layers — best-case shallow freshwater value
model_min5 = np.nanmin(c_ond[:5, :], axis=0)
model_min5 = np.where(np.isnan(model_min5), 0.0, model_min5)

# Also compute min-top-5 per well for the comparison
fw_model_min5 = np.zeros(len(fw_names))
for i, dist in enumerate(fw_dist):
    cidx = int(np.argmin(np.abs(x_centers - dist)))
    vals = c_ond[:5, cidx]
    valid = vals[~np.isnan(vals)]
    fw_model_min5[i] = float(np.min(valid)) if len(valid) > 0 else 0.0

fig3, ax3 = plt.subplots(figsize=(11, 5))

ax3.plot(x_centers, model_min5,
         color='steelblue', lw=2.0,
         label='Model — min of top 5 layers  (OND 2025)', zorder=3)

dw = fw_type == 'DW'
tw = fw_type == 'TW'

ax3.scatter(fw_dist[dw], fw_sal[dw],
            s=65, c='tomato', edgecolors='black', lw=0.7,
            zorder=5, label='Field data — dug well')
ax3.scatter(fw_dist[tw], fw_sal[tw],
            s=65, c='darkorange', edgecolors='black', marker='^', lw=0.7,
            zorder=5, label='Field data — tubewell')

for i, name in enumerate(fw_names):
    xyoff = (4, 5) if fw_type[i] == 'DW' else (4, -12)
    col   = 'tomato' if fw_type[i] == 'DW' else 'darkorange'
    ax3.annotate(name, (fw_dist[i], fw_sal[i]),
                 textcoords='offset points', xytext=xyoff,
                 fontsize=7.5, color=col)

ax3.set_xlabel("Distance from coast (m)", fontsize=11)
ax3.set_ylabel("Salinity (ppt)", fontsize=11)
ax3.set_title(
    "Salinity Transect — Model vs Field Data  |  OND 2025\n"
    "NITK Surathkal Campus Aquifer",
    fontsize=11, fontweight='bold'
)
ax3.set_xlim(0, 1200)
ax3.set_ylim(bottom=0)
ax3.grid(True, linestyle='--', alpha=0.4)
ax3.legend(fontsize=10, loc='upper right', framealpha=0.9)

# Add domain limitation note as text box
ax3.text(0.02, 0.97,
         "Note: model domain top = 0 m MSL.\n"
         "Near-coast dug wells (75–310 m) sample\n"
         "freshwater lens above model domain.",
         transform=ax3.transAxes, fontsize=7.5, va='top',
         bbox=dict(boxstyle='round', facecolor='lightyellow',
                   edgecolor='gray', alpha=0.9))

fig3.tight_layout()

p3 = os.path.join(OUT, "plot3_transect_validation.png")
fig3.savefig(p3, dpi=150)
plt.close(fig3)
print(f"    Saved -> {p3}")

# =============================================================================
# PLOT 4 — Comparison table
# Uses fw_model_min5 (min top 5 layers) for the model column
# =============================================================================
print("  Plot 4 — comparison table ...")

n_w = len(fw_names)
fig4, ax4 = plt.subplots(figsize=(15, 0.50 * n_w + 2.5))
ax4.axis('off')
ax4.set_title(
    "Model vs Field Data — Salinity Comparison  |  OND 2025\n"
    "NITK Surathkal Campus Aquifer  |  Model: min of top 5 layers (0 to −5 m MSL)",
    fontsize=11, fontweight='bold', pad=14
)

col_labels = [
    'Well', 'Dist\n(m)', 'Type',
    'Field\n(ppt)', 'Model\n(ppt)', 'Diff\n(ppt)',
    'Cl⁻\n(mg/L)', 'HCO₃⁻\n(mg/L)', 'Cl/HCO₃',
    'Notes'
]

rows   = []
colors = []

for i in range(n_w):
    field = float(fw_sal[i])
    model = float(fw_model_min5[i])
    diff  = model - field
    ratio = fw_ratio[i]

    cl_s  = f"{fw_Cl[i]:.0f}"    if not np.isnan(fw_Cl[i])   else '—'
    hco_s = f"{fw_HCO3[i]:.1f}"  if not np.isnan(fw_HCO3[i]) else '—'
    rat_s = f"{ratio:.1f}"        if not np.isnan(ratio)       else '—'

    rows.append([
        fw_names[i],
        f"{int(fw_dist[i])}",
        fw_type[i],
        f"{field:.4f}",
        f"{model:.4f}",
        f"{diff:+.4f}",
        cl_s, hco_s, rat_s,
        str(fw_notes[i]),
    ])

    # Colour by model/field ratio
    if field > 0.001:
        r = model / field
        bg = ('#d4edda' if 0.5 <= r <= 2.0 else
              '#fff3cd' if 0.2 <= r <= 5.0 else '#f8d7da')
    else:
        bg = '#e2e3e5'
    colors.append([bg] * len(col_labels))

tbl = ax4.table(
    cellText=rows,
    colLabels=col_labels,
    cellColours=colors,
    cellLoc='center',
    loc='center',
    bbox=[0, 0, 1, 1],
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)

for j in range(len(col_labels)):
    tbl[0, j].set_facecolor('#2c3e50')
    tbl[0, j].set_text_props(color='white', fontweight='bold')

for j, w in enumerate([0.07, 0.06, 0.05, 0.07, 0.07, 0.07,
                        0.08, 0.09, 0.08, 0.26]):
    for i in range(n_w + 1):
        tbl[i, j].set_width(w)

fig4.tight_layout()
p4 = os.path.join(OUT, "plot4_comparison_table.png")
fig4.savefig(p4, dpi=150, bbox_inches='tight')
plt.close(fig4)
print(f"    Saved -> {p4}")

# =============================================================================
# RMSE SUMMARY — using min-top-5 extraction
# =============================================================================
f_arr = np.array(fw_sal,      dtype=float)
m_arr = np.array(fw_model_min5, dtype=float)
rmse_all    = float(np.sqrt(np.mean((m_arr - f_arr)**2)))
inland      = fw_dist > 200
rmse_inland = float(np.sqrt(np.mean((m_arr[inland] - f_arr[inland])**2)))
rmse_coast  = float(np.sqrt(np.mean((m_arr[~inland] - f_arr[~inland])**2)))

print()
print("=" * 65)
print("  RMSE (model min-top-5 vs field, OND 2025):")
print(f"    All wells       : {rmse_all:.4f} ppt")
print(f"    Coastal ≤200 m  : {rmse_coast:.4f} ppt")
print(f"    Inland  >200 m  : {rmse_inland:.4f} ppt")
print()
print("  NOTE: Near-coast RMSE is expected to be high.")
print("  Model domain top = 0 m MSL; dug wells (75–310 m) are on")
print("  ground 6–17 m MSL and sample a freshwater lens above the")
print("  model domain. This is a documented domain limitation.")
print()
print("  Mismatch note codes:")
print("    T = Tidal / beach-face processes (not in 2D model)")
print("    H = Lateral heterogeneity / preferential flow")
print("    D = Depth — shallow open well vs model layer average")
print("    S = 3D lateral mixing not captured in 2D section")
print("    P = Paleo-marine residual (Holocene highstand signature)")
print("        not reproducible in 30-year operational model")
print()
print("  Next step: run C4_validate.py")
print("=" * 65)