# =============================================================================
# C1_build_run.py  —  SWI PROJECT  |  NITK / Surathkal Campus Aquifer
# =============================================================================
# Builds all MODFLOW 6 input files and runs the base (operational) simulation.
#
# Run ONCE. Output files are saved to modflow_workspace/base/ and read by
# C2_extract.py. Re-run if model parameters or boundary conditions change.
#
# Runtime: ~10 minutes.
#
# --- INITIAL CONDITION SWITCH ---
# USE_PALEO_IC (below) controls the GWT initial condition:
#
#   False  — GWT starts fully fresh (C = 0.0 ppt everywhere).
#            Standard fresh-IC operational run.
#
#   True   — GWT starts from the paleo spin-up residual saline field
#            stored in paleo_final_conc.npy (produced by C1_build_paleo.py).
#            This represents the Holocene paleo-marine residual after 750 yr
#            of pre-anthropogenic flushing, and is the physically preferred IC.
#            Run C1_build_paleo.py first to generate paleo_final_conc.npy.
#
# Requires:
#   config.py              — all parameters, boundary condition builders
#   input_ghb.csv          — GHB heads 1996-2025
#   rainfall CSV           — seasonal rainfall 1960-2025
#   paleo_final_conc.npy   — required only when USE_PALEO_IC = True
#
# Outputs (in modflow_workspace/base/):
#   gwf_swi.hds        — hydraulic heads (binary)
#   gwt_swi.ucn        — concentrations (binary)
#   gwf_swi.cbb        — GWF budget (binary)
#   gwt_swi.cbb        — GWT budget (binary)
# =============================================================================

import numpy as np
import flopy
import os
import config

# =============================================================================
# INITIAL CONDITION SWITCH
# Set True to use the paleo spin-up residual field as GWT initial condition.
# Set False for the standard fully-fresh start (original behaviour).
# paleo_final_conc.npy must exist (run C1_build_paleo.py first).
# =============================================================================
USE_PALEO_IC = True

# =============================================================================
# UNPACK CONFIG — all names match config.py exports exactly
# =============================================================================
ws       = config.WORKSPACE_BASE
gwf_name = config.GWF_NAME
gwt_name = config.GWT_NAME
mf6_exe  = config.MF6_EXE

nlay, nrow, ncol = config.nlay, config.nrow, config.ncol
delr, delc       = config.delr, config.delc
tops, bots       = config.tops, config.bots
nper             = config.nper
perlen           = config.perlen
nstp             = config.nstp
tsmult           = config.tsmult

os.makedirs(ws, exist_ok=True)

# =============================================================================
# STRESS PERIOD DATA
# Build all boundary condition arrays for all 121 stress periods.
# Warm-up SP (index 0): long-term mean recharge, no pumping, mean GHB head.
# Pumping SPs (1-120): year/season specific recharge, GHB head, pumping.
# =============================================================================
print("  Building stress period data ...")

# Mean GHB head used for warm-up SP (average of all seasonal means 1996-2025)
_warmup_ghb = float(np.mean(
    [config.get_ghb_head(y, s)
     for y in config.PUMPING_YEARS
     for s in config.SEASONS]
))

rch_sp = {}
wel_sp = {}
ghb_sp = {}
chd_sp = {}
cnc_sp = {}

for sp in range(nper):
    meta = config.SP_META[sp]
    yr   = meta["year"]
    seas = meta["season"]

    if yr is None:
        # Warm-up: long-term mean recharge, mean GHB head, no pumping
        R     = config.WARMUP_R
        h_ghb = _warmup_ghb
    else:
        R     = config.get_recharge(yr, seas)
        h_ghb = config.get_ghb_head(yr, seas)

    rch_sp[sp] = np.full((nrow, ncol), R, dtype=float)
    wel_sp[sp] = config.make_wel_data(yr)
    ghb_sp[sp] = config.make_ghb_list(h_ghb)
    chd_sp[sp] = config.make_chd_sea(0.0)
    cnc_sp[sp] = config.make_cnc_sea()

print(f"  Stress period data built: {nper} SPs")
print(f"  Warm-up GHB head : {_warmup_ghb:.3f} m")
print(f"  Warm-up recharge : {config.WARMUP_R*1000:.4f} mm/d")

# =============================================================================
# BUILD SIMULATION
# =============================================================================
print("\n  Building MODFLOW 6 simulation ...")

sim = flopy.mf6.MFSimulation(
    sim_name=gwf_name,
    version="mf6",
    exe_name=mf6_exe,
    sim_ws=ws,
)

# Time discretisation
flopy.mf6.ModflowTdis(
    sim,
    time_units="days",
    nper=nper,
    perioddata=list(zip(perlen, nstp, tsmult)),
)

# -------------------------------------------------------------------------
# GWF — groundwater flow model
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
    modelname=gwf_name,
    save_flows=True,
    # newtonoptions="NEWTON UNDER_RELAXATION",
)
sim.register_ims_package(ims_gwf, [gwf_name])

# Discretisation
flopy.mf6.ModflowGwfdis(
    gwf,
    nlay=nlay, nrow=nrow, ncol=ncol,
    delr=delr, delc=delc,
    top=tops[0], botm=bots,
    idomain=1,
    length_units="METERS",
)

# Node property flow — spatially variable K(x,z)
flopy.mf6.ModflowGwfnpf(
    gwf,
    save_specific_discharge=True,
    icelltype=1,
    k=config.K33_3D,
    k33=config.K33_3D,
    wetdry=1.0,
)

# Initial conditions — start at mean GHB head
flopy.mf6.ModflowGwfic(gwf, strt=_warmup_ghb)

# Storage — transient for all SPs
flopy.mf6.ModflowGwfsto(
    gwf,
    sy=config.SY,
    ss=config.SS,
    iconvert=1,
    transient={sp: True for sp in range(nper)},
)

# Constant head — sea boundary (x=0, all layers)
flopy.mf6.ModflowGwfchd(
    gwf,
    auxiliary=["CONCENTRATION"],
    stress_period_data=chd_sp,
    pname="CHD",
)

# General head boundary — inland boundary (x=1500 m, last 4 columns)
flopy.mf6.ModflowGwfghb(
    gwf,
    auxiliary=["CONCENTRATION"],
    stress_period_data=ghb_sp,
    pname="GHB",
)

# Recharge — seasonal, applied to top layer
flopy.mf6.ModflowGwfrcha(
    gwf,
    recharge=rch_sp,
    pname="RCH",
)

# Wells — logistic pumping growth 1996-2025
flopy.mf6.ModflowGwfwel(
    gwf,
    auxiliary=["CONCENTRATION"],
    stress_period_data=wel_sp,
    pname="WEL",
)

# Buoyancy — density coupling with GWT concentration
flopy.mf6.ModflowGwfbuy(
    gwf,
    nrhospecies=1,
    denseref=config.RHO_REF,
    packagedata=[(0, config.DRHODC, config.C_FRESH, gwt_name, "CONCENTRATION")],
)

# Output control
flopy.mf6.ModflowGwfoc(
    gwf,
    head_filerecord  =f"{gwf_name}.hds",
    budget_filerecord=f"{gwf_name}.cbb",
    saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
)

# -------------------------------------------------------------------------
# GWT — groundwater transport model
# -------------------------------------------------------------------------
gwt = flopy.mf6.ModflowGwt(sim, modelname=gwt_name, save_flows=True)

ims_gwt = flopy.mf6.ModflowIms(
    sim,
    pname="ims_gwt",
    filename="gwt.ims",
    print_option="SUMMARY",
    complexity="MODERATE",
    outer_maximum=500,
    inner_maximum=150,
    outer_dvclose=1e-6,
    inner_dvclose=1e-7,
    rcloserecord=1e-5,
    linear_acceleration="BICGSTAB",
    relaxation_factor=0.97,
)
sim.register_ims_package(ims_gwt, [gwt_name])

# Discretisation (identical grid to GWF)
flopy.mf6.ModflowGwtdis(
    gwt,
    nlay=nlay, nrow=nrow, ncol=ncol,
    delr=delr, delc=delc,
    top=tops[0], botm=bots,
    idomain=1,
    length_units="METERS",
)

# Initial condition — fresh start or paleo residual field
if USE_PALEO_IC:
    _paleo_npy = os.path.join(config._HERE, "paleo_final_conc.npy")
    if not os.path.isfile(_paleo_npy):
        raise FileNotFoundError(
            f"\n{'='*60}\n"
            f"  USE_PALEO_IC = True but paleo_final_conc.npy not found.\n"
            f"  Expected: {_paleo_npy}\n"
            f"  Run C1_build_paleo.py first, then re-run this script.\n"
            f"{'='*60}"
        )
    _gwt_ic = np.load(_paleo_npy)          # shape (nlay, nrow, ncol)
    print(f"  GWT IC : PALEO residual field loaded from paleo_final_conc.npy")
    print(f"           Conc range: {_gwt_ic.min():.3f} – {_gwt_ic.max():.3f} ppt")
else:
    _gwt_ic = config.C_FRESH
    print(f"  GWT IC : Fresh start (C = {config.C_FRESH} ppt everywhere)")

flopy.mf6.ModflowGwtic(gwt, strt=_gwt_ic)

# Advection — TVD scheme (best for sharp salt-fresh interface)
flopy.mf6.ModflowGwtadv(gwt, scheme="TVD")

# Dispersion
flopy.mf6.ModflowGwtdsp(
    gwt,
    alh=config.ALPHA_L,
    ath1=config.ALPHA_T,
    atv=config.ALPHA_T,
    diffc=0.0,
)

# Mobile storage and transfer
flopy.mf6.ModflowGwtmst(gwt, porosity=config.POROSITY)

# Concentration boundary — sea (x=0, all layers)
flopy.mf6.ModflowGwtcnc(
    gwt,
    stress_period_data=cnc_sp,
    pname="CNC",
)

# Source-sink mixing — pulls concentration from CHD, GHB, WEL auxiliaries
flopy.mf6.ModflowGwtssm(
    gwt,
    sources=[
        ("CHD", "AUX", "CONCENTRATION"),
        ("GHB", "AUX", "CONCENTRATION"),
        ("WEL", "AUX", "CONCENTRATION"),
    ],
    pname="SSM",
)

# Output control
flopy.mf6.ModflowGwtoc(
    gwt,
    concentration_filerecord=f"{gwt_name}.ucn",
    budget_filerecord        =f"{gwt_name}.cbb",
    saverecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
    printrecord=[("CONCENTRATION", "LAST")],
)

# -------------------------------------------------------------------------
# GWF-GWT flow exchange
# -------------------------------------------------------------------------
flopy.mf6.ModflowGwfgwt(
    sim,
    exgtype="GWF6-GWT6",
    exgmnamea=gwf_name,
    exgmnameb=gwt_name,
    filename=f"{gwf_name}_{gwt_name}.exg",
)

# =============================================================================
# WRITE & RUN
# =============================================================================
print(f"\n  Writing model files to: {ws}")
sim.write_simulation(silent=False)
# Patch NPF file to inject REWET options block
# FloPy 3.10.0 cannot serialise rewet_record correctly (known bug).
# REWET activates cell rewetting so MAM dry cells recover during JJAS.
# WETFCT=0.1: rewetted cell head = bottom + 0.1*wetdry threshold
# IWETIT=5:   check for rewetting every 5 outer iterations
# IHDWET=1:   use head of wetting cell + wetdry to set initial rewet head
_npf_path = os.path.join(ws, f"{gwf_name}.npf")
_npf_text = open(_npf_path).read()
if "REWET" not in _npf_text:
    _npf_text = _npf_text.replace(
        "BEGIN options",
        "BEGIN options\n  REWET\n  WETFCT 0.1\n  IWETIT 5\n  IHDWET 1"
    )
    open(_npf_path, "w").write(_npf_text)
    print("  NPF patched: REWET options injected.")
print(f"\n  Running simulation ({nper} stress periods) ...")
print(f"  Expected runtime: ~10 minutes. Do not close this window.\n")

success, buff = sim.run_simulation(silent=False, report=True)

if not success:
    print("\n" + "="*60)
    print("  SIMULATION FAILED — MODFLOW output:")
    print("="*60)
    for line in buff:
        print(line)
    raise RuntimeError("Base simulation failed. Check output above.")

print("\n" + "="*60)
print("  BASE SIMULATION COMPLETED SUCCESSFULLY")
print("="*60)
print(f"  IC mode          : {'Paleo residual (USE_PALEO_IC=True)' if USE_PALEO_IC else 'Fresh start (USE_PALEO_IC=False)'}")
print(f"  Output directory : {ws}")
print(f"  Head file        : {gwf_name}.hds")
print(f"  Concentration    : {gwt_name}.ucn")
print(f"  GWF budget       : {gwf_name}.cbb")
print(f"  GWT budget       : {gwt_name}.cbb")
print("\n  Next step: run C2_extract.py")
print("="*60)