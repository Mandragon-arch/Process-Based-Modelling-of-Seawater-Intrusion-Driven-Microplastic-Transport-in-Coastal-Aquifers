# =============================================================================
# C4_sensitivity.py  —  SWI PROJECT  |  NITK / Surathkal Campus Aquifer
# =============================================================================
# Sensitivity analysis — two scenarios compared against the base run:
#
# SCENARIO 1 — GHB CONDUCTANCE SENSITIVITY
#   Base : K_regional = 3.5 m/d  (C_GHB_BASE from config)
#   Sens : K_regional = 7.0 m/d  (doubled — upper plausible estimate)
#   Rationale: regional K is the least-constrained parameter; doubling it
#   tests whether the inland freshwater flux can significantly suppress
#   wedge advance. If the model is insensitive, the base value is robust.
#
# SCENARIO 2 — SEA LEVEL RISE (SLR)
#   Base : CHD head = 0.0 m MSL
#   SLR  : CHD head = +0.15 m MSL (15 cm rise, IPCC AR6 Arabian Sea 2050)
#   CHD salinity unchanged at 35 ppt (SLR does not change ocean chemistry).
#   GHB heads held at base values (conservative: regional water table does
#   not rise with sea level — upper-bound SLR impact estimate).
#
# Both scenarios run for YEAR 2025 ONLY (4 seasons) using the same grid,
# transport parameters, and pumping as the base operational model.
# Concentration IC for sensitivity runs = base model OND 2024 field (SP 116).
#
# OUTPUTS (in results/):
#   plot_sens_ghb_sections.png   — cross-sections base vs GHB sens (2025)
#   plot_sens_ghb_toe.png        — toe bar chart base vs GHB sens
#   plot_sens_slr_sections.png   — cross-sections base vs SLR (2025)
#   plot_sens_slr_toe.png        — toe bar chart base vs SLR
#   plot_sens_summary.png        — combined summary panel
#
# Run after: C2_extract.py (needs C2_extracted.npz for base IC)
# Runtime: ~3–5 minutes per scenario (4 SPs each)
# =============================================================================

import numpy as np
import flopy
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
import config

print("=" * 65)
print("  C4_sensitivity.py  —  GHB conductance + SLR scenarios")
print("=" * 65)

# =============================================================================
# LOAD BASE C2 OUTPUT (for IC and base toe values)
# =============================================================================
npz_path = os.path.join(config.RESULTS_DIR, "C2_extracted.npz")
if not os.path.isfile(npz_path):
    raise FileNotFoundError(f"C2_extracted.npz not found: {npz_path}\nRun C2_extract.py first.")

d = np.load(npz_path, allow_pickle=True)
conc_all   = d['conc_all'].astype(float)
sp_years   = d['sp_years']
sp_seasons = d['sp_seasons']
toe_OND    = d['toe_OND'].astype(float)
toe_JF     = d['toe_JF'].astype(float)
toe_MAM    = d['toe_MAM'].astype(float)
toe_JJAS   = d['toe_JJAS'].astype(float)
toe_years  = d['toe_years']
x_centers  = d['x_centers'].astype(float)
top_surface= d['top_surface'].astype(float)

# Base toes for 2025
_toe_map_base = {'JF': toe_JF, 'MAM': toe_MAM, 'JJAS': toe_JJAS, 'OND': toe_OND}
base_toes_2025 = {}
yr_idx = int(np.searchsorted(toe_years, 2025))
for s in config.SEASONS:
    base_toes_2025[s] = float(_toe_map_base[s][yr_idx])

print(f"  Base toes (2025): "
      f"JF={base_toes_2025['JF']:.0f}  MAM={base_toes_2025['MAM']:.0f}  "
      f"JJAS={base_toes_2025['JJAS']:.0f}  OND={base_toes_2025['OND']:.0f} m")

# =============================================================================
# GRID AND CONFIG
# =============================================================================
nlay, nrow, ncol = config.nlay, config.nrow, config.ncol
delr, delc       = config.delr, config.delc
dis_top          = config.dis_top
dis_botm         = config.dis_botm
IDOMAIN          = config.IDOMAIN
bots_3d          = config.bots_3d
mf6_exe          = config.MF6_EXE
OUT              = config.RESULTS_DIR

_active_layers = {j: [k for k in range(nlay) if IDOMAIN[k, 0, j] == 1]
                  for j in range(ncol)}

# IC for sensitivity runs: OND 2025 base concentration field (SP 120)
_sp_ond2024 = config.sp_of(2024, 'OND')   # SP 116 — use as IC for 2025 runs
_idx_ic = None
for i, (y, s) in enumerate(zip(sp_years, sp_seasons)):
    if int(y) == 2024 and str(s) == 'OND':
        _idx_ic = i
        break
if _idx_ic is None:
    raise RuntimeError("OND 2024 not found in C2_extracted.npz — check base run.")

_sens_ic = conc_all[_idx_ic]   # (nlay, ncol) — concentration at OND 2024
_sens_ic_3d = _sens_ic[:, np.newaxis, :]   # (nlay, 1, ncol)
_sens_ic_3d[IDOMAIN == 0] = 0.0
print(f"  Sensitivity IC: OND 2024 conc  "
      f"range={float(np.nanmin(_sens_ic)):.3f}–{float(np.nanmax(_sens_ic)):.3f} ppt")

# =============================================================================
# HELPER: build and run a 4-SP sensitivity simulation (year 2025 only)
# =============================================================================
def build_sens_sim(ws_sens, gwf_name, gwt_name, ghb_sp, chd_sp, cnc_sp,
                   ic_conc_3d, label="SENS"):
    """Build, write and run a 4-SP (2025 seasons) MODFLOW 6 sensitivity sim."""
    os.makedirs(ws_sens, exist_ok=True)

    # 4 SPs: JF, MAM, JJAS, OND 2025
    seasons_2025 = config.SEASONS
    perlen_s = [float(config.SEASON_DAYS[s]) for s in seasons_2025]
    nstp_s   = [12, 18, 20, 18]
    nper_s   = 4

    # Pumping for 2025
    wel_sp_s = {i: config.make_wel_data(2025) for i in range(nper_s)}

    sim = flopy.mf6.MFSimulation(
        sim_name=gwf_name, version="mf6",
        exe_name=mf6_exe, sim_ws=ws_sens,
    )
    flopy.mf6.ModflowTdis(
        sim, time_units="days", nper=nper_s,
        perioddata=list(zip(perlen_s, nstp_s, [1.0]*nper_s)),
    )

    ims_gwf = flopy.mf6.ModflowIms(
        sim, pname="ims_gwf", filename="gwf.ims",
        print_option="SUMMARY", complexity="MODERATE",
        outer_maximum=500, inner_maximum=200,
        outer_dvclose=1e-5, inner_dvclose=1e-6,
        rcloserecord=1e-4, linear_acceleration="BICGSTAB",
        relaxation_factor=0.98,
    )
    gwf = flopy.mf6.ModflowGwf(sim, modelname=gwf_name, save_flows=True)
    sim.register_ims_package(ims_gwf, [gwf_name])

    flopy.mf6.ModflowGwfdis(
        gwf, nlay=nlay, nrow=nrow, ncol=ncol,
        delr=delr, delc=delc,
        top=dis_top, botm=dis_botm, idomain=IDOMAIN,
        length_units="METERS",
    )
    flopy.mf6.ModflowGwfnpf(
        gwf, save_specific_discharge=True,
        icelltype=1, k=config.K33_3D, k33=config.K33_3D, wetdry=1.0,
    )
    # IC: OND 2024 head from base model (use GHB mean as proxy)
    _h_ic = float(np.mean([config.get_ghb_head(2024, s) for s in config.SEASONS]))
    flopy.mf6.ModflowGwfic(gwf, strt=_h_ic)
    flopy.mf6.ModflowGwfsto(
        gwf, sy=config.SY, ss=config.SS,
        iconvert=1, transient={i: True for i in range(nper_s)},
    )
    flopy.mf6.ModflowGwfchd(
        gwf, auxiliary=["CONCENTRATION"],
        stress_period_data=chd_sp, pname="CHD",
    )
    flopy.mf6.ModflowGwfghb(
        gwf, auxiliary=["CONCENTRATION"],
        stress_period_data=ghb_sp, pname="GHB",
    )
    flopy.mf6.ModflowGwfrcha(
        gwf,
        recharge={i: np.full((nrow, ncol),
                             config.get_recharge(2025, s), dtype=float)
                  for i, s in enumerate(seasons_2025)},
        pname="RCH",
    )
    flopy.mf6.ModflowGwfwel(
        gwf, auxiliary=["CONCENTRATION"],
        stress_period_data=wel_sp_s, pname="WEL",
    )
    flopy.mf6.ModflowGwfbuy(
        gwf, nrhospecies=1, denseref=config.RHO_REF,
        packagedata=[(0, config.DRHODC, config.C_FRESH, gwt_name, "CONCENTRATION")],
    )
    flopy.mf6.ModflowGwfoc(
        gwf,
        head_filerecord  =f"{gwf_name}.hds",
        budget_filerecord=f"{gwf_name}.cbb",
        saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
        printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    )

    gwt = flopy.mf6.ModflowGwt(sim, modelname=gwt_name, save_flows=True)
    ims_gwt = flopy.mf6.ModflowIms(
        sim, pname="ims_gwt", filename="gwt.ims",
        print_option="SUMMARY", complexity="MODERATE",
        outer_maximum=500, inner_maximum=150,
        outer_dvclose=1e-6, inner_dvclose=1e-7,
        rcloserecord=1e-5, linear_acceleration="BICGSTAB",
        relaxation_factor=0.97,
    )
    sim.register_ims_package(ims_gwt, [gwt_name])

    flopy.mf6.ModflowGwtdis(
        gwt, nlay=nlay, nrow=nrow, ncol=ncol,
        delr=delr, delc=delc,
        top=dis_top, botm=dis_botm, idomain=IDOMAIN,
        length_units="METERS",
    )
    flopy.mf6.ModflowGwtic(gwt, strt=ic_conc_3d)
    flopy.mf6.ModflowGwtadv(gwt, scheme="TVD")
    flopy.mf6.ModflowGwtdsp(
        gwt, alh=config.ALPHA_L, ath1=config.ALPHA_T,
        atv=config.ALPHA_T, diffc=0.0,
    )
    flopy.mf6.ModflowGwtmst(gwt, porosity=config.POROSITY)
    flopy.mf6.ModflowGwtcnc(gwt, stress_period_data=cnc_sp, pname="CNC")
    flopy.mf6.ModflowGwtssm(
        gwt,
        sources=[("CHD","AUX","CONCENTRATION"),
                 ("GHB","AUX","CONCENTRATION"),
                 ("WEL","AUX","CONCENTRATION")],
        pname="SSM",
    )
    flopy.mf6.ModflowGwtoc(
        gwt,
        concentration_filerecord=f"{gwt_name}.ucn",
        budget_filerecord        =f"{gwt_name}.cbb",
        saverecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
        printrecord=[("CONCENTRATION", "LAST")],
    )
    flopy.mf6.ModflowGwfgwt(
        sim, exgtype="GWF6-GWT6",
        exgmnamea=gwf_name, exgmnameb=gwt_name,
        filename=f"{gwf_name}_{gwt_name}.exg",
    )

    sim.write_simulation(silent=True)

    # Patch REWET
    _npf_path = os.path.join(ws_sens, f"{gwf_name}.npf")
    _txt = open(_npf_path).read()
    if "REWET" not in _txt:
        _txt = _txt.replace("BEGIN options",
                            "BEGIN options\n  REWET\n  WETFCT 0.1\n  IWETIT 5\n  IHDWET 1")
        open(_npf_path, "w").write(_txt)

    print(f"  Running {label} ...")
    success, buff = sim.run_simulation(silent=True, report=True)
    if not success:
        for line in buff:
            print(line)
        raise RuntimeError(f"{label} simulation failed.")
    print(f"  {label} — completed OK")
    return ws_sens


def read_sens_conc(ws_sens, gwt_name, sp_idx):
    """Read concentration for a given 0-based SP index from sensitivity run."""
    ucn_path = os.path.join(ws_sens, f"{gwt_name}.ucn")
    ucn_obj  = flopy.utils.HeadFile(ucn_path, text='concentration')
    records  = ucn_obj.get_kstpkper()
    kper_map = {}
    for kstp, kper in records:
        if kper not in kper_map or kstp > kper_map[kper][0]:
            kper_map[kper] = (kstp, kper)
    ksp = kper_map.get(sp_idx)
    if ksp is None:
        return None
    c3d = ucn_obj.get_data(kstpkper=ksp)
    c2d = c3d[:, 0, :]
    c2d[IDOMAIN[:, 0, :] == 0] = np.nan
    return c2d


def compute_toe_2d(c2d):
    col_means = np.full(ncol, np.nan)
    for j in range(ncol):
        act  = _active_layers[j]
        bot5 = act[-5:] if len(act) >= 5 else act
        vals = [c2d[k, j] for k in bot5 if not np.isnan(c2d[k, j])]
        if vals:
            col_means[j] = float(np.mean(vals))
    cols = np.where(col_means >= 0.5)[0]
    return float(x_centers[cols[-1]]) if len(cols) > 0 else 0.0


# Grid edges for pcolormesh (coast column)
dx      = np.diff(x_centers)
x_edges = np.concatenate([[x_centers[0]-dx[0]/2],
                           x_centers[:-1]+dx/2,
                           [x_centers[-1]+dx[-1]/2]])
_tops_coast = config.tops_3d[:, 0, 0]
_bots_coast = config.bots_3d[:, 0, 0]
z_edges = np.concatenate([[_tops_coast[0]], _bots_coast])
Z_BOT   = float(_bots_coast[-1])
Z_TOP   = float(top_surface.max()) + 0.5

CMAP = plt.cm.RdYlBu_r
NORM = mcolors.LogNorm(vmin=0.05, vmax=35.0)

# =============================================================================
# SCENARIO 1 — GHB CONDUCTANCE SENSITIVITY (K = 7.0 m/d)
# =============================================================================
print("\n" + "="*65)
print("  SCENARIO 1 — GHB Conductance Sensitivity  K=7.0 m/d")
print("="*65)

ws_ghb  = os.path.join(config._HERE, "modflow_workspace", "ghb_sens")
gwf_s   = "gwf_sens"
gwt_s   = "gwt_sens"

ghb_sp_sens = {}
chd_sp_sens = {}
cnc_sp_sens = {}
for i, seas in enumerate(config.SEASONS):
    h_ghb = config.get_ghb_head(2025, seas)
    ghb_sp_sens[i] = config.make_ghb_list(h_ghb, c_per_layer=config.C_GHB_SENS)
    chd_sp_sens[i] = config.make_chd_sea(0.0)
    cnc_sp_sens[i] = config.make_cnc_sea()

build_sens_sim(ws_ghb, gwf_s, gwt_s,
               ghb_sp_sens, chd_sp_sens, cnc_sp_sens,
               _sens_ic_3d, label="GHB SENS K=7.0")

ghb_sens_toes = {}
ghb_sens_c2d  = {}
for i, seas in enumerate(config.SEASONS):
    c2d = read_sens_conc(ws_ghb, gwt_s, i)
    if c2d is not None:
        ghb_sens_toes[seas] = compute_toe_2d(c2d)
        ghb_sens_c2d[seas]  = c2d
    else:
        ghb_sens_toes[seas] = 0.0

print(f"\n  GHB Sens toes (2025): "
      f"JF={ghb_sens_toes['JF']:.0f}  MAM={ghb_sens_toes['MAM']:.0f}  "
      f"JJAS={ghb_sens_toes['JJAS']:.0f}  OND={ghb_sens_toes['OND']:.0f} m")

# =============================================================================
# SCENARIO 2 — SEA LEVEL RISE (CHD = +0.15 m)
# =============================================================================
print("\n" + "="*65)
print("  SCENARIO 2 — Sea Level Rise  CHD = +0.15 m MSL")
print("="*65)

H_SLR   = 0.15
ws_slr  = os.path.join(config._HERE, "modflow_workspace", "slr")
gwf_slr = "gwf_slr"
gwt_slr = "gwt_slr"

ghb_sp_slr = {}
chd_sp_slr = {}
cnc_sp_slr = {}
for i, seas in enumerate(config.SEASONS):
    h_ghb = config.get_ghb_head(2025, seas)
    ghb_sp_slr[i] = config.make_ghb_list(h_ghb)
    # SLR: raise CHD head to +0.15 m MSL but keep salinity at 35 ppt
    chd_sp_slr[i] = [[(k, 0, 0), H_SLR, config.C_SEA]
                     for k in range(nlay)
                     if IDOMAIN[k, 0, 0] == 1
                     and H_SLR > float(bots_3d[k, 0, 0]) + 1e-6]
    cnc_sp_slr[i] = config.make_cnc_sea()

build_sens_sim(ws_slr, gwf_slr, gwt_slr,
               ghb_sp_slr, chd_sp_slr, cnc_sp_slr,
               _sens_ic_3d, label=f"SLR CHD=+{H_SLR}m")

slr_toes = {}
slr_c2d  = {}
for i, seas in enumerate(config.SEASONS):
    c2d = read_sens_conc(ws_slr, gwt_slr, i)
    if c2d is not None:
        slr_toes[seas] = compute_toe_2d(c2d)
        slr_c2d[seas]  = c2d
    else:
        slr_toes[seas] = 0.0

print(f"\n  SLR toes (2025): "
      f"JF={slr_toes['JF']:.0f}  MAM={slr_toes['MAM']:.0f}  "
      f"JJAS={slr_toes['JJAS']:.0f}  OND={slr_toes['OND']:.0f} m")

# =============================================================================
# PLOTS
# =============================================================================
SCOL = config.SEASON_COLORS
SLBL = {'JF':'Jan-Feb','MAM':'Mar-May','JJAS':'Jun-Sep','OND':'Oct-Dec'}

def _draw_cross(ax, c2d, toe, title, color='steelblue', xlim=800):
    c_plot = np.where(np.isnan(c2d), 0.0, c2d)
    pcm = ax.pcolormesh(x_edges, z_edges, c_plot, cmap=CMAP, norm=NORM, shading='flat')
    try:
        ax.contour(x_centers, config.z_mid_layers, c_plot,
                   levels=[0.5], colors='black', linewidths=1.0, linestyles='--')
    except Exception:
        pass
    ax.plot(x_centers, top_surface, color='saddlebrown', lw=1.0, ls='-')
    if 0 < toe <= xlim:
        ax.axvline(toe, color='red', lw=1.2, ls=':', label=f"Toe {toe:.0f} m")
    ax.set_xlim(0, xlim)
    ax.set_ylim(Z_BOT, Z_TOP)
    ax.set_title(title, fontsize=8.5, color=color, pad=3)
    return pcm

# --- PLOT: GHB Sensitivity cross-sections (2025 all seasons) ---
fig_g, axes_g = plt.subplots(4, 2, figsize=(12, 13), sharex=True, sharey=True)
fig_g.suptitle(
    "GHB Conductance Sensitivity — Year 2025  |  NITK Surathkal\n"
    f"Left: Base K=3.5 m/d  |  Right: Sensitivity K=7.0 m/d",
    fontsize=10, fontweight='bold'
)
pcm_ref = None
for row, seas in enumerate(config.SEASONS):
    idx_base = None
    for i, (y, s) in enumerate(zip(sp_years, sp_seasons)):
        if int(y) == 2025 and str(s) == seas:
            idx_base = i; break
    c_base = conc_all[idx_base] if idx_base is not None else np.zeros((nlay, ncol))
    c_sens = ghb_sens_c2d.get(seas, np.zeros((nlay, ncol)))
    toe_b  = base_toes_2025.get(seas, 0.0)
    toe_s  = ghb_sens_toes.get(seas, 0.0)

    ax_b = axes_g[row, 0]
    ax_s = axes_g[row, 1]
    _draw_cross(ax_b, c_base, toe_b, f"{SLBL[seas]}  Base  Toe={toe_b:.0f} m", 'steelblue')
    pcm_ref = _draw_cross(ax_s, c_sens, toe_s,
                          f"{SLBL[seas]}  Sens K=7.0  Toe={toe_s:.0f} m", 'darkorange')

for row in range(4):
    axes_g[row, 0].set_ylabel("Elevation (m MSL)", fontsize=8)
for col in range(2):
    axes_g[3, col].set_xlabel("Distance from coast (m)", fontsize=9)

fig_g.subplots_adjust(right=0.85, hspace=0.40, wspace=0.10, top=0.91)
if pcm_ref is not None:
    cbar_ax = fig_g.add_axes([0.87, 0.08, 0.018, 0.82])
    cb = fig_g.colorbar(pcm_ref, cax=cbar_ax, extend='both')
    cb.set_label("Salinity (ppt)", fontsize=9)
    cb.set_ticks([0.05, 0.1, 0.5, 1, 5, 10, 35])
    cb.set_ticklabels(['0.05','0.1','0.5','1','5','10','35'])

p_g1 = os.path.join(OUT, "plot_sens_ghb_sections.png")
fig_g.savefig(p_g1, dpi=150, bbox_inches='tight')
plt.close(fig_g)
print(f"\n  Saved -> {p_g1}")

# --- PLOT: GHB sensitivity toe bar chart ---
fig_gt, ax_gt = plt.subplots(figsize=(9, 5))
bar_x = np.arange(len(config.SEASONS))
bar_w = 0.35
bars_b = ax_gt.bar(bar_x - bar_w/2,
                   [base_toes_2025[s] for s in config.SEASONS],
                   bar_w, color='steelblue', edgecolor='black', lw=0.8,
                   label='Base  K=3.5 m/d')
bars_s = ax_gt.bar(bar_x + bar_w/2,
                   [ghb_sens_toes[s] for s in config.SEASONS],
                   bar_w, color='darkorange', edgecolor='black', lw=0.8,
                   label='Sensitivity  K=7.0 m/d')
for bars, col in [(bars_b,'steelblue'),(bars_s,'darkorange')]:
    for b in bars:
        h = b.get_height()
        ax_gt.text(b.get_x()+b.get_width()/2, h+3, f"{h:.0f}",
                   ha='center', va='bottom', fontsize=8.5, color=col)
ax_gt.set_xticks(bar_x)
ax_gt.set_xticklabels(config.SEASONS, fontsize=11)
ax_gt.set_ylabel("Wedge toe — distance from coast (m)", fontsize=11)
ax_gt.set_title("GHB Conductance Sensitivity — Wedge Toe 2025\n"
                "NITK Surathkal Campus Aquifer",
                fontsize=11, fontweight='bold')
ax_gt.set_ylim(0, max(max(base_toes_2025.values()), max(ghb_sens_toes.values())) * 1.2)
ax_gt.grid(True, axis='y', linestyle='--', alpha=0.4)
ax_gt.legend(fontsize=10)
fig_gt.tight_layout()
p_g2 = os.path.join(OUT, "plot_sens_ghb_toe.png")
fig_gt.savefig(p_g2, dpi=150)
plt.close(fig_gt)
print(f"  Saved -> {p_g2}")

# --- PLOT: SLR cross-sections ---
fig_s, axes_s = plt.subplots(4, 2, figsize=(12, 13), sharex=True, sharey=True)
fig_s.suptitle(
    f"Sea Level Rise Sensitivity — Year 2025  |  NITK Surathkal\n"
    f"Left: Base CHD=0.0 m MSL  |  Right: SLR CHD=+{H_SLR} m MSL",
    fontsize=10, fontweight='bold'
)
pcm_ref2 = None
for row, seas in enumerate(config.SEASONS):
    idx_base = None
    for i, (y, s) in enumerate(zip(sp_years, sp_seasons)):
        if int(y) == 2025 and str(s) == seas:
            idx_base = i; break
    c_base = conc_all[idx_base] if idx_base is not None else np.zeros((nlay, ncol))
    c_slr  = slr_c2d.get(seas, np.zeros((nlay, ncol)))
    toe_b  = base_toes_2025.get(seas, 0.0)
    toe_s  = slr_toes.get(seas, 0.0)

    _draw_cross(axes_s[row, 0], c_base, toe_b,
                f"{SLBL[seas]}  Base  Toe={toe_b:.0f} m", 'steelblue')
    pcm_ref2 = _draw_cross(axes_s[row, 1], c_slr, toe_s,
                           f"{SLBL[seas]}  SLR +{H_SLR}m  Toe={toe_s:.0f} m", 'firebrick')

for row in range(4):
    axes_s[row, 0].set_ylabel("Elevation (m MSL)", fontsize=8)
for col in range(2):
    axes_s[3, col].set_xlabel("Distance from coast (m)", fontsize=9)

fig_s.subplots_adjust(right=0.85, hspace=0.40, wspace=0.10, top=0.91)
if pcm_ref2 is not None:
    cbar_ax2 = fig_s.add_axes([0.87, 0.08, 0.018, 0.82])
    cb2 = fig_s.colorbar(pcm_ref2, cax=cbar_ax2, extend='both')
    cb2.set_label("Salinity (ppt)", fontsize=9)
    cb2.set_ticks([0.05, 0.1, 0.5, 1, 5, 10, 35])
    cb2.set_ticklabels(['0.05','0.1','0.5','1','5','10','35'])

p_s1 = os.path.join(OUT, "plot_sens_slr_sections.png")
fig_s.savefig(p_s1, dpi=150, bbox_inches='tight')
plt.close(fig_s)
print(f"  Saved -> {p_s1}")

# --- PLOT: SLR toe bar chart ---
fig_st, ax_st = plt.subplots(figsize=(9, 5))
bars_b2 = ax_st.bar(bar_x - bar_w/2,
                    [base_toes_2025[s] for s in config.SEASONS],
                    bar_w, color='steelblue', edgecolor='black', lw=0.8,
                    label='Base  CHD=0.0 m')
bars_slr = ax_st.bar(bar_x + bar_w/2,
                     [slr_toes[s] for s in config.SEASONS],
                     bar_w, color='firebrick', edgecolor='black', lw=0.8,
                     label=f'SLR  CHD=+{H_SLR} m')
for bars, col in [(bars_b2,'steelblue'),(bars_slr,'firebrick')]:
    for b in bars:
        h = b.get_height()
        ax_st.text(b.get_x()+b.get_width()/2, h+3, f"{h:.0f}",
                   ha='center', va='bottom', fontsize=8.5, color=col)
ax_st.set_xticks(bar_x)
ax_st.set_xticklabels(config.SEASONS, fontsize=11)
ax_st.set_ylabel("Wedge toe — distance from coast (m)", fontsize=11)
ax_st.set_title(f"Sea Level Rise Sensitivity — Wedge Toe 2025\n"
                "NITK Surathkal Campus Aquifer", fontsize=11, fontweight='bold')
ax_st.set_ylim(0, max(max(base_toes_2025.values()), max(slr_toes.values())) * 1.2)
ax_st.grid(True, axis='y', linestyle='--', alpha=0.4)
ax_st.legend(fontsize=10)
fig_st.tight_layout()
p_s2 = os.path.join(OUT, "plot_sens_slr_toe.png")
fig_st.savefig(p_s2, dpi=150)
plt.close(fig_st)
print(f"  Saved -> {p_s2}")

# =============================================================================
# PRINTED SUMMARY TABLES
# =============================================================================
print("\n" + "="*65)
print("  SENSITIVITY SUMMARY — Year 2025")
print("="*65)
print(f"\n  GHB Conductance Sensitivity (K_base=3.5 m/d vs K_sens=7.0 m/d):")
print(f"  {'Season':<8} {'Base toe (m)':>14} {'Sens toe (m)':>14} {'Δtoe (m)':>10}")
print(f"  {'-'*8} {'-'*14} {'-'*14} {'-'*10}")
for s in config.SEASONS:
    tb = base_toes_2025[s]; ts = ghb_sens_toes[s]
    print(f"  {s:<8} {tb:>14.1f} {ts:>14.1f} {ts-tb:>+10.1f}")

print(f"\n  Sea Level Rise Sensitivity (CHD=0.0 m vs CHD=+{H_SLR} m):")
print(f"  {'Season':<8} {'Base toe (m)':>14} {'SLR toe (m)':>14} {'Δtoe (m)':>10}")
print(f"  {'-'*8} {'-'*14} {'-'*14} {'-'*10}")
for s in config.SEASONS:
    tb = base_toes_2025[s]; ts = slr_toes[s]
    print(f"  {s:<8} {tb:>14.1f} {ts:>14.1f} {ts-tb:>+10.1f}")

print("\n" + "="*65)
print("  C4 COMPLETE")
print("="*65)
print(f"  GHB sensitivity plots : {p_g1}  {p_g2}")
print(f"  SLR sensitivity plots : {p_s1}  {p_s2}")
print("  Next: run C5_summary.py")