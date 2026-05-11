# =============================================================================
# C1_build_paleo.py  —  SWI PROJECT  |  NITK / Surathkal Campus Aquifer
# =============================================================================
# PALEO SPIN-UP MODEL
#
# PURPOSE
# -------
# This script simulates the long-term freshwater flushing of a fully marine
# aquifer following the Holocene sea-level highstand (~6000–4000 yr BP).
# Geological evidence (Hashimi 1995, Bhatt 2006) shows that sea level along
# the west coast of India stood 1–3 m above present MSL during this period,
# flooding coastal aquifers to full marine salinity.  When sea level fell to
# its present level, freshwater recharge began flushing the aquifer seaward.
# After ~750 years of pre-anthropogenic flushing (no pumping), the residual
# saline field produced by this model is used as the initial condition for
# the operational run (C1_build_run.py with USE_PALEO_IC = True).
#
# SCIENTIFIC RATIONALE
# --------------------
# The base operational model (C1_build_run.py) starts with a fully fresh
# initial condition.  This explains near-coast salinity well but cannot
# reproduce the elevated Cl/HCO3 ratios observed at 745–1100 m from the
# coast (Field data, OND 2024-25), where active seawater intrusion is
# physically implausible given present-day hydraulic gradients.  Those
# inland observations are best explained as paleo-marine residuals — saline
# water trapped in low-permeability zones during the Holocene transgression
# and incompletely flushed in the subsequent centuries.
#
# MODEL DESIGN
# ------------
#   Duration     : 750 years (273,750 days) — conservative pre-colonial,
#                  pre-NITK estimate; covers the period between ~the late
#                  Holocene regression and the onset of large-scale extraction
#   IC (GWT)     : C_SEA = 35.0 ppt everywhere (fully marine aquifer)
#   IC (GWF)     : head equilibrated to CHD_HIGHSTAND
#   CHD (coast)  : +1.5 m MSL (midpoint of 1–3 m highstand range)
#                  constant for full 750 yr ("sustained highstand" assumption)
#   GHB (inland) : long-term mean head — same warmup value as C1_build_run
#   Recharge     : long-term annual mean (WARMUP_R from config) — no
#                  year-specific or seasonal forcing; we have no pre-1960
#                  rainfall data for this region
#   Pumping      : NONE — pre-anthropogenic era
#   Time stepping: single SP, nstp=50, tsmult=1.2 — logarithmic expansion
#                  gives fine resolution early (sharp interface migration)
#                  coarsening as system approaches quasi-steady state
#
# AQUIFER NOTE
# ------------
# The modelled unit is the laterite/coastal sandy aquifer, 0 to -40 m MSL.
# Below -40 m lies fractured Deccan basalt / granitic basement — not
# modelled.  The base of the modelled section is treated as a no-flow
# boundary, consistent with the low hydraulic connectivity of the
# fractured basement at this depth.
#
# OUTPUTS
# -------
#   modflow_workspace/paleo/gwf_paleo.hds   — hydraulic heads
#   modflow_workspace/paleo/gwt_paleo.ucn   — concentrations
#   paleo_final_conc.npy                    — final concentration array
#                                             (nlay x nrow x ncol)
#                                             used as IC in operational run
#   results/paleo_crosssection.png          — residual saline field at t=750 yr
#   results/paleo_comparison_table.csv      — model vs field data salinity
#
# WORKFLOW
# --------
#   Step 1 (this script) : run paleo spin-up -> save paleo_final_conc.npy
#   Step 2               : run C1_build_run.py with USE_PALEO_IC = True
#                          (set flag at top of that script)
#   Step 3               : run C2_extract.py, C3, C4, C5 as normal
#
# Runtime: ~5–8 minutes (single SP, no iteration convergence issues expected
#          because the interface moves slowly after the first few decades).
#
# Requires:
#   config.py          — all shared parameters
#   input_ghb.csv      — to compute warmup GHB head
#   rainfall CSV       — to compute WARMUP_R (already done in config)
# =============================================================================

import numpy as np
import flopy
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import csv
import config

# =============================================================================
# PALEO-SPECIFIC CONSTANTS
# =============================================================================

# Holocene sea-level highstand head [m above MSL]
# Source: Hashimi (1995), Bhatt (2006) — west coast India highstand 1–3 m.
# Using midpoint as the "sustained highstand" assumption.
CHD_HIGHSTAND     = 1.5          # m above MSL

# Duration of pre-anthropogenic flushing [years]
# Represents the interval between the late-Holocene regression (~750 yr BP)
# and the onset of significant groundwater extraction at the NITK campus.
# Conservative lower bound — longer durations would produce a more flushed
# (fresher) residual field.
PALEO_YEARS       = 750          # years

PALEO_DAYS        = PALEO_YEARS * 365.25   # 273,937.5 days

# Time stepping — single stress period, logarithmic expansion
# nstp=50 with tsmult=1.2 gives time steps from ~4 days (early, fast interface
# migration) to ~3,400 days (late, near-equilibrium slow flushing).
PALEO_NSTP        = 50
PALEO_TSMULT      = 1.2

# JF drawdown factor — January-February (dry winter) head relative to OND
# of the preceding season.  In the absence of JF-specific CGWB records,
# JF head is approximated as JF_DRAWDOWN_FACTOR × OND head of the same year.
# Physically: the water table has been depleting since the post-monsoon peak
# (OND) under dry conditions.  Value 0.6 means ~40% drawdown from OND peak.
# Adjustable here without touching any other script.
JF_DRAWDOWN_FACTOR = 0.6

# Paleo workspace and output paths
ws_paleo          = os.path.join(config._HERE, "modflow_workspace", "paleo")
PALEO_CONC_NPY    = os.path.join(config._HERE, "paleo_final_conc.npy")
PALEO_PLOT        = os.path.join(config.RESULTS_DIR, "paleo_crosssection.png")
PALEO_TABLE_CSV   = os.path.join(config.RESULTS_DIR, "paleo_comparison_table.csv")

# Model names (distinct from operational run names)
GWF_PALEO = "gwf_paleo"
GWT_PALEO = "gwt_paleo"

os.makedirs(ws_paleo,       exist_ok=True)
os.makedirs(config.RESULTS_DIR, exist_ok=True)

# =============================================================================
# UNPACK CONFIG
# =============================================================================
nlay, nrow, ncol = config.nlay, config.nrow, config.ncol
delr, delc       = config.delr, config.delc
tops, bots       = config.tops, config.bots
x_centers        = config.x_centers
z_mid            = config.z_mid_layers
mf6_exe          = config.MF6_EXE

# =============================================================================
# BOUNDARY CONDITION DATA
# =============================================================================
print("  Building paleo boundary condition data ...")

# GHB head: long-term mean, same logic as C1_build_run warmup
_warmup_ghb = float(np.mean(
    [config.get_ghb_head(y, s)
     for y in config.PUMPING_YEARS
     for s in config.SEASONS]
))
print(f"  Paleo GHB head (warmup mean) : {_warmup_ghb:.4f} m")
print(f"  Paleo CHD head (highstand)   : {CHD_HIGHSTAND:.1f} m")
print(f"  Paleo recharge (annual mean) : {config.WARMUP_R * 1000:.4f} mm/d")
print(f"  JF drawdown factor           : {JF_DRAWDOWN_FACTOR}  (JF = {JF_DRAWDOWN_FACTOR} × OND)")

# Single stress period — all BCs constant
nper_paleo = 1

# CHD: coast at highstand level, salinity = C_SEA
chd_paleo  = [[(k, 0, 0), CHD_HIGHSTAND, config.C_SEA] for k in range(nlay)]

# CNC: fix coast concentration at C_SEA
cnc_paleo  = [[(k, 0, 0), config.C_SEA] for k in range(nlay)]

# GHB: inland boundary at long-term mean head, fresh
ghb_paleo  = config.make_ghb_list(_warmup_ghb)

# Recharge: long-term annual mean to top layer
rch_paleo  = {0: np.full((nrow, ncol), config.WARMUP_R, dtype=float)}

# No wells during paleo period (pre-anthropogenic)
# wel_paleo intentionally omitted — no WEL package in this simulation

# =============================================================================
# BUILD SIMULATION
# =============================================================================
print("\n  Building MODFLOW 6 paleo simulation ...")

sim = flopy.mf6.MFSimulation(
    sim_name=GWF_PALEO,
    version="mf6",
    exe_name=mf6_exe,
    sim_ws=ws_paleo,
)

# Single long stress period — logarithmic time stepping
flopy.mf6.ModflowTdis(
    sim,
    time_units="days",
    nper=nper_paleo,
    perioddata=[(PALEO_DAYS, PALEO_NSTP, PALEO_TSMULT)],
)

# -------------------------------------------------------------------------
# GWF
# -------------------------------------------------------------------------
ims_gwf = flopy.mf6.ModflowIms(
    sim,
    pname="ims_gwf",
    filename="gwf.ims",
    print_option="SUMMARY",
    complexity="MODERATE",
    outer_maximum=500,
    inner_maximum=200,
    outer_dvclose=1e-5,
    inner_dvclose=1e-6,
    rcloserecord=1e-4,
    linear_acceleration="BICGSTAB",
    relaxation_factor=0.98,
)

gwf = flopy.mf6.ModflowGwf(
    sim,
    modelname=GWF_PALEO,
    save_flows=True,
    # newtonoptions="NEWTON UNDER_RELAXATION",
)
sim.register_ims_package(ims_gwf, [GWF_PALEO])

flopy.mf6.ModflowGwfdis(
    gwf,
    nlay=nlay, nrow=nrow, ncol=ncol,
    delr=delr, delc=delc,
    top=tops[0], botm=bots,
    idomain=1,
    length_units="METERS",
)

flopy.mf6.ModflowGwfnpf(
    gwf,
    save_specific_discharge=True,
    icelltype=1,
    k=config.K33_3D,
    k33=config.K33_3D,
    wetdry=1.0,
)

# GWF IC: head equilibrated to highstand CHD
# Using CHD_HIGHSTAND as a uniform start is physically reasonable —
# during the transgression the water table was at or above sea level
# everywhere in this low-lying coastal strip.
flopy.mf6.ModflowGwfic(gwf, strt=CHD_HIGHSTAND)

flopy.mf6.ModflowGwfsto(
    gwf,
    sy=config.SY,
    ss=config.SS,
    iconvert=1,
    transient={0: True},
)

# Coast: Holocene highstand CHD with seawater concentration
flopy.mf6.ModflowGwfchd(
    gwf,
    auxiliary=["CONCENTRATION"],
    stress_period_data={0: chd_paleo},
    pname="CHD",
)

# Inland: GHB at long-term mean head, fresh
flopy.mf6.ModflowGwfghb(
    gwf,
    auxiliary=["CONCENTRATION"],
    stress_period_data={0: ghb_paleo},
    pname="GHB",
)

# Recharge: long-term annual mean to top layer
flopy.mf6.ModflowGwfrcha(
    gwf,
    recharge=rch_paleo,
    pname="RCH",
)

# NOTE: No WEL package — pre-anthropogenic model has zero pumping.

# Buoyancy coupling
flopy.mf6.ModflowGwfbuy(
    gwf,
    nrhospecies=1,
    denseref=config.RHO_REF,
    packagedata=[(0, config.DRHODC, config.C_FRESH, GWT_PALEO, "CONCENTRATION")],
)

flopy.mf6.ModflowGwfoc(
    gwf,
    head_filerecord  =f"{GWF_PALEO}.hds",
    budget_filerecord=f"{GWF_PALEO}.cbb",
    saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
)

# -------------------------------------------------------------------------
# GWT
# -------------------------------------------------------------------------
gwt = flopy.mf6.ModflowGwt(sim, modelname=GWT_PALEO, save_flows=True)

ims_gwt = flopy.mf6.ModflowIms(
    sim,
    pname="ims_gwt",
    filename="gwt.ims",
    print_option="SUMMARY",
    complexity="MODERATE",
    outer_maximum=400,
    inner_maximum=120,
    outer_dvclose=1e-6,
    inner_dvclose=1e-7,
    rcloserecord=1e-5,
    linear_acceleration="BICGSTAB",
    relaxation_factor=0.97,
)
sim.register_ims_package(ims_gwt, [GWT_PALEO])

flopy.mf6.ModflowGwtdis(
    gwt,
    nlay=nlay, nrow=nrow, ncol=ncol,
    delr=delr, delc=delc,
    top=tops[0], botm=bots,
    idomain=1,
    length_units="METERS",
)

# GWT IC: fully marine — entire aquifer at seawater salinity
# This represents the Holocene transgression end-member: the coastal aquifer
# was saturated with marine water when sea level was 1–3 m above present MSL.
flopy.mf6.ModflowGwtic(gwt, strt=config.C_SEA)

flopy.mf6.ModflowGwtadv(gwt, scheme="TVD")

flopy.mf6.ModflowGwtdsp(
    gwt,
    alh=config.ALPHA_L,
    ath1=config.ALPHA_T,
    atv=config.ALPHA_T,
    diffc=0.0,
)

flopy.mf6.ModflowGwtmst(gwt, porosity=config.POROSITY)

# Coast concentration fixed at seawater (highstand maintained throughout)
flopy.mf6.ModflowGwtcnc(
    gwt,
    stress_period_data={0: cnc_paleo},
    pname="CNC",
)

# SSM: CHD and GHB carry concentrations via auxiliary
# No WEL, so no well source-sink mixing needed
flopy.mf6.ModflowGwtssm(
    gwt,
    sources=[
        ("CHD", "AUX", "CONCENTRATION"),
        ("GHB", "AUX", "CONCENTRATION"),
    ],
    pname="SSM",
)

flopy.mf6.ModflowGwtoc(
    gwt,
    concentration_filerecord=f"{GWT_PALEO}.ucn",
    budget_filerecord        =f"{GWT_PALEO}.cbb",
    saverecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
    printrecord=[("CONCENTRATION", "LAST")],
)

# GWF-GWT exchange
flopy.mf6.ModflowGwfgwt(
    sim,
    exgtype="GWF6-GWT6",
    exgmnamea=GWF_PALEO,
    exgmnameb=GWT_PALEO,
    filename=f"{GWF_PALEO}_{GWT_PALEO}.exg",
)

# =============================================================================
# WRITE & RUN
# =============================================================================
print(f"\n  Writing paleo model files to: {ws_paleo}")
sim.write_simulation(silent=False)

print(f"\n  Running paleo simulation ({PALEO_YEARS} yr = {PALEO_DAYS:.0f} days) ...")
print(f"  Time steps: {PALEO_NSTP}, tsmult={PALEO_TSMULT}  (logarithmic expansion)")
print(f"  Expected runtime: ~5–8 minutes.\n")

success, buff = sim.run_simulation(silent=False, report=True)

if not success:
    print("\n" + "=" * 60)
    print("  PALEO SIMULATION FAILED — MODFLOW output:")
    print("=" * 60)
    for line in buff:
        print(line)
    raise RuntimeError("Paleo simulation failed. Check output above.")

print("\n" + "=" * 60)
print("  PALEO SIMULATION COMPLETED SUCCESSFULLY")
print("=" * 60)

# =============================================================================
# EXTRACT FINAL CONCENTRATION FIELD
# =============================================================================
print("\n  Extracting final concentration field ...")

ucn_file = flopy.utils.HeadFile(
    os.path.join(ws_paleo, f"{GWT_PALEO}.ucn"),
    text="CONCENTRATION",
)
# Last time step = end of 750-yr flushing
paleo_conc = ucn_file.get_data(kstpkper=(PALEO_NSTP - 1, 0))  # (nlay, nrow, ncol)

np.save(PALEO_CONC_NPY, paleo_conc)
print(f"  Paleo final concentration saved -> {PALEO_CONC_NPY}")
print(f"  Array shape : {paleo_conc.shape}  (nlay x nrow x ncol)")
print(f"  Conc range  : {paleo_conc.min():.3f} – {paleo_conc.max():.3f} ppt")

# Quick sanity checks
_coast_mean = float(paleo_conc[:, 0, 0].mean())
_inland_mean = float(paleo_conc[:, 0, -1].mean())
_mid_mean    = float(paleo_conc[:, 0, ncol // 2].mean())
print(f"  Column-mean salinity — coast (x~0): {_coast_mean:.2f} ppt  "
      f"| mid (x~750 m): {_mid_mean:.2f} ppt  "
      f"| inland (x~1500 m): {_inland_mean:.2f} ppt")

# =============================================================================
# PLOT — PALEO CROSS-SECTION (residual saline field at t = 750 yr)
# =============================================================================
print("\n  Generating paleo cross-section plot ...")

C2d    = paleo_conc[:, 0, :]   # (nlay, ncol)
x_edges = np.concatenate([[0.0], np.cumsum(delr)])
z_edges = np.concatenate([[tops[0]], bots])

cmap = plt.cm.RdYlBu_r
norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=config.C_SEA / 2.0, vmax=config.C_SEA)

fig, axes = plt.subplots(1, 2, figsize=(18, 7),
                          gridspec_kw={"width_ratios": [2, 1]})

fig.suptitle(
    f"Paleo Spin-Up — Residual Saline Field after {PALEO_YEARS} yr Freshwater Flushing\n"
    f"IC: fully marine (35 ppt)  |  CHD = +{CHD_HIGHSTAND} m MSL (Holocene highstand)  |  "
    f"No pumping  |  NITK Campus Aquifer",
    fontsize=12, fontweight="bold", y=1.01,
)

# Left panel: full-width cross-section
ax = axes[0]
pcm = ax.pcolormesh(x_edges, z_edges, C2d, cmap=cmap, norm=norm, shading="flat")
_c50 = ax.contour(x_centers, z_mid, C2d,
           levels=[config.C_SEA / 2.0], colors="black", linewidths=1.5,
           linestyles="--")
_c1  = ax.contour(x_centers, z_mid, C2d,
           levels=[1.0], colors="lime", linewidths=1.2,
           linestyles=":")
plt.colorbar(pcm, ax=ax, label="Salinity [ppt]", shrink=0.85)
ax.set_xlabel("Distance from coast [m]", fontsize=11)
ax.set_ylabel("Elevation [m MSL]", fontsize=11)
ax.set_title(f"Full section (0-1500 m)", fontsize=10)
ax.set_xlim(0, 1500)
ax.set_ylim(-40, 2)

# Legend via proxy artists (ax.contour does not accept label= kwarg)
from matplotlib.lines import Line2D as _Line2D
ax.legend(
    handles=[
        _Line2D([0], [0], color="black", lw=1.5, ls="--"),
        _Line2D([0], [0], color="lime",  lw=1.2, ls=":"),
    ],
    labels=["50% seawater interface", "1 ppt contour (fresh limit)"],
    fontsize=8, loc="upper right", framealpha=0.8,
)

# Mark field-well positions on cross-section
_well_colors = config.WELL_COLORS
for i, (wname, wd) in enumerate(sorted(config.FIELD_WELLS.items(),
                                        key=lambda x: x[1]["dist"])):
    if wd["dist"] <= 1500:
        ax.axvline(wd["dist"], color="grey", lw=0.7, ls=":", alpha=0.6)
        ax.text(wd["dist"], 0.5, wname, rotation=90, fontsize=6.5,
                va="bottom", ha="center", color="grey")

# Right panel: coastal zoom (0–500 m) — where field wells are
ax2 = axes[1]
pcm2 = ax2.pcolormesh(x_edges, z_edges, C2d, cmap=cmap, norm=norm, shading="flat")
ax2.contour(x_centers, z_mid, C2d,
            levels=[config.C_SEA / 2.0], colors="black", linewidths=1.5, linestyles="--")
ax2.contour(x_centers, z_mid, C2d,
            levels=[1.0], colors="lime", linewidths=1.2, linestyles=":")
for i, (wname, wd) in enumerate(sorted(config.FIELD_WELLS.items(),
                                        key=lambda x: x[1]["dist"])):
    if wd["dist"] <= 500:
        ax2.axvline(wd["dist"], color="grey", lw=0.8, ls=":", alpha=0.7)
        ax2.text(wd["dist"], 0.5, wname, rotation=90, fontsize=7,
                 va="bottom", ha="center", color="grey")
ax2.set_xlabel("Distance from coast [m]", fontsize=11)
ax2.set_ylabel("Elevation [m MSL]", fontsize=11)
ax2.set_title("Coastal zoom (0–500 m)", fontsize=10)
ax2.set_xlim(0, 500)
ax2.set_ylim(-40, 2)

# Annotate key paleo parameters
_txt = (
    f"Duration     : {PALEO_YEARS} yr\n"
    f"CHD (coast)  : +{CHD_HIGHSTAND} m MSL\n"
    f"GHB (inland) : {_warmup_ghb:.3f} m MSL\n"
    f"Recharge     : {config.WARMUP_R * 1000:.3f} mm/d\n"
    f"Pumping      : None\n"
    f"IC (GWT)     : {config.C_SEA:.0f} ppt (fully marine)\n"
    f"Conc range   : {paleo_conc.min():.2f}–{paleo_conc.max():.2f} ppt"
)
axes[0].text(0.72, 0.04, _txt, transform=axes[0].transAxes,
             fontsize=8, va="bottom", ha="left",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                       edgecolor="grey", alpha=0.85))

plt.tight_layout()
fig.savefig(PALEO_PLOT, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Paleo cross-section saved -> {PALEO_PLOT}")

# =============================================================================
# COMPARISON TABLE
# Field data  vs  Operational IC model (fresh start)  vs  Paleo IC model
#
# For the paleo IC model column, we extract the concentration from the
# *paleo_final_conc* array at each well's distance and representative depth.
# The operational model column reads from the base run outputs; if those
# files are not yet available, that column is filled with NaN and a note
# is printed.
#
# Well-type depth assignment (consistent with C4/C5 plots):
#   DW (open dug well)   -> layers 1-5  (~1-5 m depth)
#   TW (tubewell)        -> layers 10-30 (~10-30 m depth, screened interval)
# =============================================================================
print("\n  Building comparison table ...")

# Try to load operational model final concentration (base run)
_op_conc_path = os.path.join(config.WORKSPACE_BASE, f"{config.GWT_NAME}.ucn")
_op_conc = None
if os.path.isfile(_op_conc_path):
    try:
        _op_ucn   = flopy.utils.HeadFile(_op_conc_path, text="CONCENTRATION")
        _op_times = _op_ucn.get_times()
        _op_conc  = _op_ucn.get_data(totim=_op_times[-1])
        print(f"  Operational model output loaded (t={_op_times[-1]:.0f} d)")
    except Exception as _e:
        print(f"  WARNING: could not load operational conc — {_e}")
        print("  Operational column will show NaN. Run C1_build_run first.")
else:
    print(f"  Operational model output not found at:\n  {_op_conc_path}")
    print("  Operational column will show NaN. Run C1_build_run.py first.")

# Mismatch note legend
MISMATCH_LEGEND = {
    "T": "Tidal / beach-face processes not captured in 2D model",
    "H": "Lateral heterogeneity / preferential flow paths",
    "D": "Depth mismatch (shallow open well vs model layer average)",
    "S": "2D section cannot capture 3D lateral mixing signal",
    "P": "Paleo-intrusion residual — geochemical fingerprint of historical "
         "seawater contact not reproduced in a 30-year fresh-IC window",
}

rows = []
for wname, wd in sorted(config.FIELD_WELLS.items(), key=lambda x: x[1]["dist"]):
    dist  = wd["dist"]
    fsal  = wd["sal"]
    wtype = wd["type"]
    notes = config.WELL_MISMATCH_NOTES[wname]

    cidx  = int(np.argmin(np.abs(x_centers - dist)))

    # Depth-representative layer slice
    if wtype == "DW":
        layers = slice(1, 6)    # 1–5 m  (surface dug well)
    else:
        layers = slice(10, 30)  # 10–30 m (tubewell screened interval)

    # Paleo IC concentration at this well location
    p_sal = float(paleo_conc[layers, 0, cidx].mean())

    # Operational IC concentration (NaN if base run not available)
    if _op_conc is not None:
        o_sal = float(_op_conc[layers, 0, cidx].mean())
    else:
        o_sal = float("nan")

    rows.append({
        "Well"               : wname,
        "Type"               : wtype,
        "Dist_m"             : dist,
        "Field_sal_ppt"      : fsal,
        "OpIC_model_ppt"     : round(o_sal, 4),
        "PaleoIC_model_ppt"  : round(p_sal, 4),
        "Delta_OpIC_ppt"     : round(o_sal - fsal, 4) if not np.isnan(o_sal) else float("nan"),
        "Delta_PaleoIC_ppt"  : round(p_sal - fsal, 4),
        "Mismatch_notes"     : notes,
    })

# Write CSV
_fields = ["Well", "Type", "Dist_m", "Field_sal_ppt",
           "OpIC_model_ppt", "PaleoIC_model_ppt",
           "Delta_OpIC_ppt", "Delta_PaleoIC_ppt", "Mismatch_notes"]

with open(PALEO_TABLE_CSV, "w", newline="", encoding="utf-8") as _f:
    _w = csv.DictWriter(_f, fieldnames=_fields)
    _w.writeheader()
    _w.writerows(rows)

    # Append legend
    _f.write("\n# MISMATCH NOTE CODES\n")
    for code, desc in MISMATCH_LEGEND.items():
        _f.write(f"# {code}: {desc}\n")
    _f.write(
        f"\n# Field data source: In-situ measurements, Thermo Scientific Orion "
        f"multiparameter meter, OND 2024-25\n"
        f"# OpIC  = Operational model with fresh initial condition (C1_build_run)\n"
        f"# PaleoIC = This script — paleo spin-up IC, {PALEO_YEARS} yr flushing, "
        f"CHD = +{CHD_HIGHSTAND} m\n"
        f"# Delta = Model - Field  (positive = model over-predicts salinity)\n"
        f"# DW depth representative: layers 1-5 (~1-5 m)\n"
        f"# TW depth representative: layers 10-30 (~10-30 m)\n"
    )

print(f"  Comparison table saved -> {PALEO_TABLE_CSV}")

# Print table to console
_hdr = (f"{'Well':>6}  {'Type':>3}  {'Dist':>5}  "
        f"{'Field':>7}  {'OpIC':>8}  {'PaleoIC':>9}  "
        f"{'ΔOp':>8}  {'ΔPaleo':>8}  Notes")
print("\n  " + _hdr)
print("  " + "-" * len(_hdr))
for r in rows:
    _delta_op    = f"{r['Delta_OpIC_ppt']:+.4f}" if not np.isnan(r['Delta_OpIC_ppt']) else "   N/A  "
    _delta_paleo = f"{r['Delta_PaleoIC_ppt']:+.4f}"
    print(f"  {r['Well']:>6}  {r['Type']:>3}  {r['Dist_m']:>5}  "
          f"{r['Field_sal_ppt']:>7.4f}  {r['OpIC_model_ppt']:>8.4f}  "
          f"{r['PaleoIC_model_ppt']:>9.4f}  "
          f"{_delta_op:>8}  {_delta_paleo:>8}  {r['Mismatch_notes']}")

# RMSE summary
_paleo_arr = np.array([r["PaleoIC_model_ppt"] for r in rows])
_field_arr = np.array([r["Field_sal_ppt"]     for r in rows])
_inland    = np.array([r["Dist_m"] for r in rows]) > 400
print(f"\n  RMSE (all wells)        — Paleo IC : {np.sqrt(np.mean((_paleo_arr - _field_arr)**2)):.5f} ppt")
print(f"  RMSE (inland >400 m)   — Paleo IC : {np.sqrt(np.mean((_paleo_arr[_inland] - _field_arr[_inland])**2)):.5f} ppt")
if _op_conc is not None:
    _op_arr = np.array([r["OpIC_model_ppt"] for r in rows])
    print(f"  RMSE (all wells)        — Op IC    : {np.sqrt(np.mean((_op_arr - _field_arr)**2)):.5f} ppt")
    print(f"  RMSE (inland >400 m)   — Op IC    : {np.sqrt(np.mean((_op_arr[_inland] - _field_arr[_inland])**2)):.5f} ppt")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 70)
print("  PALEO BUILD COMPLETE")
print("=" * 70)
print(f"  Paleo conc array   : {PALEO_CONC_NPY}")
print(f"  Cross-section plot : {PALEO_PLOT}")
print(f"  Comparison table   : {PALEO_TABLE_CSV}")
print()
print("  NEXT STEPS:")
print("  1. Review paleo_cross section.png — check residual wedge extent.")
print("  2. In C1_build_run.py, set  USE_PALEO_IC = True  at the top of")
print("     the script (flag to be added).  This will load paleo_final_conc.npy")
print("     as the GWT initial condition instead of C_FRESH = 0.0.")
print("  3. Re-run C1_build_run.py, then C2_extract.py, C3, C4, C5.")
print("  4. The comparison table will auto-populate the OpIC column once")
print("     the base run output (gwt_swi.ucn) exists.")
print("=" * 70)