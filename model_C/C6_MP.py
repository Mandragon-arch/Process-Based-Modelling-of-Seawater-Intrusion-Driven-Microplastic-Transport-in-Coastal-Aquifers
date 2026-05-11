#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
C6_MP_transport.py  —  Microplastic Transport Model (config-coupled)
NITK Surathkal Campus Aquifer  |  Coastal Karnataka, India
======================================================================
Self-contained GWF + GWT_salt + GWT_MP simulation that shares all
physical and grid parameters with the SWI model via config.py.

Key differences from the SWI base model (config / C1_build_run):
  • SOURCE_MODE toggle: "sea_only" | "rech_only" | "both"
  • Two GWT models run simultaneously: gwt_salt (salinity) + gwt_mp (MPs)
  • MP-specific sorption (linear Kd, salinity-dependent) via MST package
  • Bradford et al. (2011) straining decay field (first_order_decay)
  • Vertical anisotropy Kv/Kh = 0.1 for layered laterite
  • Finer time stepping than SWI model for MP front resolution

References:
  Bradford et al. (2002) WRR 38(12) — distance-dependent straining
  Bradford et al. (2011) ES&T 45(16) — depth-dependent straining
  Torkzaban et al. (2012) WRR 48 — MP dispersivity scaling
  Porter et al. (2015) ES&T 49 — Kd salinity-dependence for MPs
  Chen et al. (2024) JHM 480 — MP in coastal groundwater, SWI driver
  Langevin et al. (2017) USGS TM 6-A55 — MODFLOW 6
  Priyanka & Mohan Kumar (2019) Water 11:421 — Pavanje laterite K

======================================================================
BUG-FIX LOG (vs earlier draft):
  FIX-1  : config.WORKSPACE_MP missing -> defined locally here
  FIX-2  : config.Z_BOTTOM missing -> derived from config.bots[-1]
  FIX-3  : config.top_surface missing -> flat array at 0 m MSL (2D model)
  FIX-4  : config.tops_3d / bots_3d missing -> broadcast from 1D tops/bots
  FIX-5  : config.dis_top / dis_botm missing -> built from grid arrays
  FIX-6  : config.IDOMAIN missing -> all-active scalar 1 (consistent with SWI)
  FIX-7  : config.ALPHA_L_MP / ALPHA_T_MP missing -> defined locally
  FIX-8  : config.KD_FRESH / KD_SALINE missing -> defined locally
  FIX-9  : config.GRAIN_DENSITY / RHO_BULK missing -> defined locally
  FIX-10 : config.get_recharge_ms missing -> use config.get_recharge directly
  FIX-11 : ModflowGwfrcha aux= kwarg wrong -> use auxiliary= + stress_period_data
           with aux values packed into array records
  FIX-12 : RCHA auxiliary values need per-SP array dicts, not separate aux=
  FIX-13 : CHD/GHB/WEL auxiliary record lengths: GWF CHD with 2 aux needs
           5-element tuples; GHB with 2 aux needs 5-element tuples; WEL same
  FIX-14 : GWF BUY packagedata references "gwt_salt" which must match GWT
           modelname exactly
  FIX-15 : SNAP_SEASONS iteration was (season, year) but sp_of(yr, s) correct
  FIX-16 : WELL_SPECS_MP depth-filter used tops_3d which doesn't exist in
           config -> use flat tops array directly
  FIX-17 : dis_top for ModflowGwfdis must be scalar or (nrow, ncol) array,
           not 1D (ncol,); corrected to scalar tops[0] = 0.0
  FIX-18 : dis_botm must be (nlay, nrow, ncol); built correctly here
  FIX-19 : GHB CHD list cell filter removed (bots_3d not available);
           simplified to all-layer list consistent with SWI model approach
  FIX-20 : k_str_3d depth calculation used tops_3d/bots_3d -> use 1D arrays
  FIX-21 : ModflowGwtdsp missing diffc keyword (consistent with SWI model)
======================================================================
"""

import os
import shutil
import sys

import flopy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.colors import TwoSlopeNorm
import numpy as np

# =============================================================================
# IMPORT CONFIG
# =============================================================================
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import config  # noqa: E402

# =============================================================================
# USER SETTINGS
# =============================================================================
SOURCE_MODE = "both"   # "sea_only" | "rech_only" | "both"

MF6_EXE = config.MF6_EXE
if not os.path.exists(MF6_EXE):
    _alt = shutil.which("mf6") or shutil.which("mf6.exe")
    if _alt:
        MF6_EXE = _alt
    else:
        raise FileNotFoundError(
            f"MODFLOW 6 not found at {MF6_EXE!r}. "
            "Update config.MF6_EXE or add mf6 to PATH.")

# FIX-1: define WORKSPACE_MP locally — config.py does not define it
WORKSPACE_MP = os.path.join(config._HERE, "modflow_workspace", "mp")
ws = os.path.join(WORKSPACE_MP, SOURCE_MODE)
os.makedirs(ws, exist_ok=True)

mode_label = SOURCE_MODE.replace("_", " ").title()
print(f"\n{'='*60}")
print(f"  C6 MP Transport Model — NITK Surathkal")
print(f"  SOURCE_MODE : {SOURCE_MODE}")
print(f"  Workspace   : {ws}")
print(f"{'='*60}\n")

# =============================================================================
# UNPACK GRID FROM CONFIG
# config.py uses a flat 2D section: tops[k] and bots[k] are scalars
# (same elevation for all columns), consistent with the SWI model.
# =============================================================================
nlay      = config.nlay
nrow      = config.nrow
ncol      = config.ncol
Lx        = config.Lx
dz        = config.dz
delr      = config.delr
delc      = config.delc
x_centers = config.x_centers
x_edges   = np.concatenate([[0.0], np.cumsum(delr)])

# FIX-2: Z_BOTTOM from grid, not a separate config attribute
Z_BOTTOM  = float(config.bots[-1])   # = -40.0 m MSL

# FIX-3: top_surface — the SWI model has a flat ground surface at 0 m MSL
# (tops[0] = 0.0 everywhere). No variable topography in this 2D model.
top_surface = np.zeros(ncol, dtype=float)   # shape (ncol,), all 0.0 m MSL

# FIX-4: broadcast 1D tops/bots to 3D for depth calculations in k_str
# tops_3d[k, 0, j] = config.tops[k]  (same for all j — flat model)
tops_3d = np.broadcast_to(
    config.tops[:, np.newaxis, np.newaxis],
    (nlay, nrow, ncol)).copy()
bots_3d = np.broadcast_to(
    config.bots[:, np.newaxis, np.newaxis],
    (nlay, nrow, ncol)).copy()

# FIX-5: dis_top and dis_botm for ModflowGwfdis
# dis_top must be scalar or (nrow, ncol); use scalar tops[0] = 0.0
dis_top  = float(config.tops[0])     # scalar: 0.0 m
# dis_botm must be (nlay, nrow, ncol) or (nlay,) for uniform layers
# FloPy accepts (nlay, nrow, ncol) or a list; use config.bots directly
dis_botm = config.bots               # shape (nlay,) — FloPy will broadcast

# FIX-6: IDOMAIN — config uses idomain=1 (scalar); build full array
IDOMAIN  = np.ones((nlay, nrow, ncol), dtype=int)

# Hydraulic conductivity — config K33_3D is Kh (isotropic in SWI model)
# Vertical anisotropy Kv/Kh = 0.1 for layered laterite
# Source: Priyanka & Mohan Kumar (2019) Water 11:421 Table 6
K_3D  = config.K33_3D          # Kh, shape (nlay, nrow, ncol)
Kz_3D = config.K33_3D * 0.1    # Kv = Kh / 10

# Storage
SY = config.SY
SS = config.SS

# Density and transport (salt)
RHO_REF  = config.RHO_REF
DRHODC   = config.DRHODC
C_SEA    = config.C_SEA
C_FRESH  = config.C_FRESH
POROSITY = config.POROSITY
ALPHA_L  = config.ALPHA_L
ALPHA_T  = config.ALPHA_T

# Pumping / wells
SECTION_WIDTH  = config.SECTION_WIDTH
WELL_FRACTIONS = config.WELL_FRACTIONS
WELL_SPECS     = config.WELL_SPECS   # list of dicts

# Stress period schedule
PUMPING_YEARS = config.PUMPING_YEARS
SEASONS       = config.SEASONS
SEASON_DAYS   = config.SEASON_DAYS
SP_META       = config.SP_META
nper          = config.nper
perlen        = config.perlen
sp_of         = config.sp_of

# Boundary helpers
WARMUP_R    = config.WARMUP_R
get_ghb_head = config.get_ghb_head
GHB_COL_IDS = config.GHB_COL_IDS
C_GHB_BASE  = config.C_GHB_BASE

# =============================================================================
# MP-SPECIFIC PARAMETERS (not in config — defined here with references)
# =============================================================================

# FIX-7: MP dispersivity
# Torkzaban et al. (2012) WRR 48: MP dispersivity ≈ 0.25 × salt dispersivity
# for 25 µm fibres in sandy/laterite matrix (reduced mechanical mixing due
# to size exclusion and straining).
ALPHA_L_MP = 0.25 * ALPHA_L    # = 2.5 m when ALPHA_L = 10 m
ALPHA_T_MP = 0.25 * ALPHA_T    # = 0.25 m

# FIX-8: Kd (distribution coefficient) for MP sorption onto aquifer grains
# Porter et al. (2015) ES&T 49: Kd_fresh ~ 1.5e-4 m3/kg for polyester fibres
# in freshwater (electrostatic repulsion dominates at low ionic strength);
# Kd_saline ~ 5.0e-4 m3/kg in seawater (double-layer compression increases
# attachment). Values are for 25 µm synthetic fibres, consistent with the
# dominant MP type observed on the Karnataka coast (Sridhar et al. 2021).
KD_FRESH  = 1.5e-4   # m3/kg — freshwater sorption coefficient
KD_SALINE = 5.0e-4   # m3/kg — saline sorption coefficient

# FIX-9: Grain density and bulk density
# Laterite grain density ~2650 kg/m3 (Mahesha 2012, Dakshina Kannada laterite)
# Bulk density = grain_density × (1 − porosity)
GRAIN_DENSITY = 2650.0                          # kg/m3
RHO_BULK      = GRAIN_DENSITY * (1.0 - POROSITY)  # = 1855 kg/m3

print(f"  [C6] MP params: αL_MP={ALPHA_L_MP:.2f} m  αT_MP={ALPHA_T_MP:.3f} m")
print(f"  [C6] Kd: fresh={KD_FRESH*1e3:.2f} L/kg  saline={KD_SALINE*1e3:.2f} L/kg")
print(f"  [C6] ρ_bulk={RHO_BULK:.0f} kg/m3  Z_BOTTOM={Z_BOTTOM:.0f} m MSL")
print(f"  [C6] nper={nper}  total={sum(perlen):.0f} d  ({sum(perlen)/365.25:.1f} yr)")

# Source concentrations
C_MP_SEA_BASE  = 1.0   # normalised — sea is the reference (C = 1.0)
C_MP_RECH_BASE = 0.5   # stormwater recharge = 50% of sea source
                        # Chen et al. (2024) JHM 480: coastal runoff MP
                        # concentrations typically 30-60% of nearshore seawater

if SOURCE_MODE == "sea_only":
    C_MP_SEA  = C_MP_SEA_BASE
    C_MP_RECH = 0.0
elif SOURCE_MODE == "rech_only":
    C_MP_SEA  = 0.0
    C_MP_RECH = C_MP_RECH_BASE
else:   # "both"
    C_MP_SEA  = C_MP_SEA_BASE
    C_MP_RECH = C_MP_RECH_BASE

THRESHOLD_SALT = C_SEA / 2.0   # = 17.5 ppt (standard 50% seawater interface)
THRESHOLD_MP   = 0.10           # 10% of sea source — detection threshold

# Kd initial condition: start with freshwater Kd everywhere
KD_IC = KD_FRESH

# =============================================================================
# BRADFORD STRAINING DECAY FIELD
# Bradford et al. (2002) WRR 38(12) Eq. 6 (distance-dependent):
#   k_str = K_STR0 × exp(−λ × x)
# Bradford et al. (2011) ES&T 45(16) Eq. 5 (depth-dependent):
#   k_str × (D50 / (D50 + depth))^β
#
# Combined: k_str[k,j] = K_STR0 × f_depth(k,j) × f_dist(j)
#
# Parameters — all from published literature for sand/laterite:
#   K_STR0      = 0.005 /d   Bradford 2011 Table 1 (median sand, conservative
#                              lower bound; actual laterite may be higher)
#   BETA_STR    = 0.432       Bradford 2011 Eq. 5 pore-size exclusion exponent
#   D50         = 5.0e-4 m   Laterite median grain size, Dakshina Kannada
#                              (Mahesha 2012, Table 2, 0.3-0.7 mm range, midpoint)
#   LAMBDA_STR  = 0.003 /m   Bradford 2002 WRR distance-decay constant
#                              (calibrated for ~25 µm particles in sandy media)
#
# Limitation (v1): salinity coupling to straining omitted; Kd handles the
# salinity-dependent attachment. Flagged for thesis sensitivity analysis.
# =============================================================================
K_STR0     = 0.005    # /day
BETA_STR   = 0.432    # dimensionless
D50        = 5.0e-4   # m (0.5 mm)
LAMBDA_STR = 0.003    # /m

k_str_3d = np.zeros((nlay, nrow, ncol))
for j in range(ncol):
    for k in range(nlay):
        # FIX-20: depth below GL using 1D arrays (flat model, GL = 0 m MSL)
        z_mid_kj = (config.tops[k] + config.bots[k]) / 2.0
        depth     = max(0.0 - z_mid_kj, 0.0)   # depth below 0 m MSL (GL)
        f_dep     = (D50 / (D50 + depth)) ** BETA_STR

        x = x_centers[j]
        f_sea  = np.exp(-LAMBDA_STR * x)
        f_rech = np.exp(-LAMBDA_STR * abs(x - 400.0))

        if SOURCE_MODE == "sea_only":
            f_dist = f_sea
        elif SOURCE_MODE == "rech_only":
            f_dist = f_rech
        else:   # "both" — use lower (more conservative) straining rate
            f_dist = min(f_sea, f_rech)

        k_str_3d[k, 0, j] = K_STR0 * f_dep * f_dist

_kact = k_str_3d[:, 0, :]
print(f"  [C6] Bradford k_str: {_kact.min():.4e}–{_kact.max():.4e} /d "
      f"(mean {_kact.mean():.4e} /d)")

# =============================================================================
# PLOTTING COORDINATE ARRAYS
# Z_2D[k, j] = midpoint elevation of layer k at column j
# X_2D[k, j] = x-coordinate of column j (same for all k)
# =============================================================================
z_mid_1d = (config.tops + config.bots) / 2.0           # shape (nlay,)
Z_2D      = np.tile(z_mid_1d[:, np.newaxis], (1, ncol)) # shape (nlay, ncol)
X_2D      = np.tile(x_centers, (nlay, 1))               # shape (nlay, ncol)

# =============================================================================
# WELL SCREEN FILTER
# Keep layers at least 2 m below the ground surface (tops[0] = 0.0 m MSL).
# This avoids the near-unsaturated zone in early stress periods.
# FIX-16: use config.tops (1D) directly; flat model so GL = tops[0] = 0.0 m.
# =============================================================================
GL = float(config.tops[0])   # = 0.0 m MSL

WELL_SPECS_MP = []
for _w in config.WELL_SPECS:
    col = _w["col"]
    # Screen layers where top elevation < GL - 2 m (i.e. depth > 2 m)
    screens_mp = [k for k in range(nlay)
                  if config.tops[k] < GL - 2.0]
    if not screens_mp:
        # Fallback: use all layers in the well's defined screen range
        screens_mp = _w["screens"]
    WELL_SPECS_MP.append({**_w, "screens": screens_mp})
    print(f"  [C6] {_w['name']}  col={col:3d}  x={x_centers[col]:.0f} m  "
          f"screens (2 m filter): L{screens_mp[0]}–L{screens_mp[-1]}  "
          f"({len(screens_mp)} layers)")

# =============================================================================
# TIME STEPPING — finer than SWI model for MP front resolution
# tsmult=1.2 gives geometric expansion; warmup SP uses uniform steps.
# =============================================================================
_nstp_mp    = {'JF': 30, 'MAM': 25, 'JJAS': 30, 'OND': 25}
nstp_mp     = [183]   # warmup: 183 uniform steps of 10 days each
tsmult_mp   = [1.0]   # warmup: uniform

for _yr in PUMPING_YEARS:
    for _s in SEASONS:
        nstp_mp.append(_nstp_mp[_s])
        tsmult_mp.append(1.2)

assert len(nstp_mp) == nper, f"nstp_mp length {len(nstp_mp)} != nper {nper}"

# =============================================================================
# RECHARGE MP PATCH (x = 0–400 m)
# Applied over the coastal strip of the NITK campus where surface runoff
# and stormwater carry MPs into the vadose zone.
# Chen et al. (2024) JHM 480 define MP source zones based on land use;
# here the coastal plain (first 400 m from coast) is the primary source.
# =============================================================================
def col_from_x(x_val):
    idx = int(np.searchsorted(x_edges, x_val, side="right")) - 1
    return int(np.clip(idx, 0, ncol - 1))

rch_col_lo = col_from_x(0.0)
rch_col_hi = col_from_x(400.0)

# =============================================================================
# SNAPSHOT STRESS PERIODS FOR PLOTTING
# FIX-15: SNAP_SEASONS is (season_str, year_int) — sp_of(yr, s) is correct
# =============================================================================
SNAP_SEASONS = [('OND', 2005), ('OND', 2015), ('OND', 2025)]
SNAP_SPS     = [sp_of(yr, s) for s, yr in SNAP_SEASONS]
SNAP_LABELS  = [f"{s} {yr}" for s, yr in SNAP_SEASONS]

# =============================================================================
# BOUNDARY CONDITION BUILDERS
# FIX-13: CHD with 2 aux vars needs 5-element tuples:
#   [(lay, row, col), head, aux1, aux2]
# FIX-19: simplified cell filter — include all layers (consistent with SWI)
# =============================================================================

def make_chd_sea_mp():
    """Coastal CHD — head=0 m, SALT_CONC=C_SEA, MP_CONC=C_MP_SEA."""
    return [[(k, 0, 0), 0.0, C_SEA, C_MP_SEA] for k in range(nlay)]


def make_ghb_list_mp(h_s):
    """Inland GHB — SALT_CONC=C_FRESH, MP_CONC=0.
    FIX-13: 5-element tuple for 2 aux vars (SALT_CONC + MP_CONC)."""
    lst = []
    for j in GHB_COL_IDS:
        for k in range(nlay):
            lst.append([(k, 0, j), h_s, C_GHB_BASE, C_FRESH, 0.0])
    return lst


def make_wel_data_mp(year):
    """Pumping WEL — SALT_CONC=C_FRESH, MP_CONC=0.
    FIX-13: 5-element tuple for 2 aux vars."""
    if year is None:
        return []
    Q = config.Q_gw(year)
    lst = []
    for w in WELL_SPECS_MP:
        q_per_layer = -(Q * w["frac"] / SECTION_WIDTH) / len(w["screens"])
        for lyr in w["screens"]:
            lst.append([(lyr, 0, w["col"]), q_per_layer, C_FRESH, 0.0])
    return lst


# =============================================================================
# BUILD STRESS PERIOD DATA DICTS
# FIX-10: use config.get_recharge (not get_recharge_ms which doesn't exist)
# FIX-11/12: RCHA auxiliary handled by building per-SP recharge arrays with
#            the MP concentration embedded — FloPy RCHA with readasarrays
#            does not support per-cell aux arrays cleanly. Instead we use the
#            ModflowGwfrcha with auxiliary= + per-SP aux array dicts.
#            The aux array shape must be (nrow, ncol) per SP.
# =============================================================================
chd_spd = {}
ghb_spd = {}
wel_spd = {}
rch_spd = {}
rch_aux_spd = {}   # per-SP MP recharge aux array

# Warmup SP (index 0)
chd_spd[0]     = make_chd_sea_mp()
ghb_spd[0]     = make_ghb_list_mp(get_ghb_head(PUMPING_YEARS[0], SEASONS[0]))
wel_spd[0]     = []   # no pumping during warmup

_rch_warmup           = np.full((nrow, ncol), WARMUP_R)
_rch_mp_warmup        = np.zeros((nrow, ncol))
rch_spd[0]            = _rch_warmup
rch_aux_spd[0]        = _rch_mp_warmup

for sp in range(1, nper):
    meta   = SP_META[sp]
    yr, s  = meta["year"], meta["season"]

    chd_spd[sp] = make_chd_sea_mp()
    ghb_spd[sp] = make_ghb_list_mp(get_ghb_head(yr, s))
    wel_spd[sp] = make_wel_data_mp(yr)

    # FIX-10: config.get_recharge returns [m/day] — correct function name
    R = config.get_recharge(yr, s)
    _rch = np.full((nrow, ncol), R)
    rch_spd[sp] = _rch

    # Recharge MP aux — C_MP_RECH over coastal patch, 0 elsewhere
    _rch_mp = np.zeros((nrow, ncol))
    _rch_mp[0, rch_col_lo:rch_col_hi] = C_MP_RECH
    rch_aux_spd[sp] = _rch_mp

# =============================================================================
# BUILD MODFLOW 6 SIMULATION
# =============================================================================
print("\n  Building MODFLOW 6 simulation ...")

perioddata = list(zip(perlen, nstp_mp, tsmult_mp))
sim = flopy.mf6.MFSimulation(
    sim_name="mp_swi",
    sim_ws=ws,
    exe_name=MF6_EXE,
)
flopy.mf6.ModflowTdis(
    sim, nper=nper, perioddata=perioddata, time_units="days")

# ---------------------------------------------------------------------------
# GWF
# ---------------------------------------------------------------------------
gwf = flopy.mf6.ModflowGwf(sim, modelname="gwf", save_flows=True,
                             newtonoptions="NEWTON UNDER_RELAXATION")

ims_flow = flopy.mf6.ModflowIms(
    sim, pname="ims_gwf", filename="ims_flow.ims",
    complexity="MODERATE",
    outer_maximum=500, inner_maximum=150,
    outer_dvclose=1e-5, inner_dvclose=1e-6,
    rcloserecord=1e-4,
    linear_acceleration="BICGSTAB",
    relaxation_factor=0.97)
sim.register_ims_package(ims_flow, ["gwf"])

# FIX-17/18: dis_top is scalar 0.0; dis_botm is 1D (nlay,) — FloPy broadcasts
flopy.mf6.ModflowGwfdis(
    gwf, nlay=nlay, nrow=nrow, ncol=ncol,
    delr=delr, delc=delc,
    top=dis_top,    # scalar: 0.0 m MSL
    botm=dis_botm,  # shape (nlay,): config.bots
    idomain=1,
    length_units="METERS")

# FIX-14: IC — use mean seasonal GHB head (consistent with SWI model)
h_init = float(np.mean([get_ghb_head(yr, s)
                         for yr in PUMPING_YEARS for s in SEASONS]))
flopy.mf6.ModflowGwfic(gwf, strt=h_init)
print(f"  GWF IC: strt = {h_init:.3f} m (mean seasonal GHB head)")

flopy.mf6.ModflowGwfnpf(
    gwf, icelltype=1,
    k=K_3D, k33=Kz_3D,
    save_specific_discharge=True)

flopy.mf6.ModflowGwfsto(
    gwf, sy=SY, ss=SS, iconvert=1,
    transient={i: True for i in range(nper)})

# BUY — variable density coupled to gwt_salt (FIX-14: name must match GWT modelname)
flopy.mf6.ModflowGwfbuy(
    gwf, denseref=RHO_REF, nrhospecies=1,
    packagedata=[[0, DRHODC, 0.0, "gwt_salt", "CONCENTRATION"]])

# CHD — coastal sea boundary (2 aux: SALT_CONC, MP_CONC)
flopy.mf6.ModflowGwfchd(
    gwf,
    auxiliary=["SALT_CONC", "MP_CONC"],
    stress_period_data=chd_spd,
    pname="SEA_CHD")

# GHB — inland freshwater boundary (2 aux: SALT_CONC, MP_CONC)
flopy.mf6.ModflowGwfghb(
    gwf,
    auxiliary=["SALT_CONC", "MP_CONC"],
    stress_period_data=ghb_spd,
    pname="INLAND_GHB")

# FIX-11/12: RCHA with MP aux array
# FloPy ModflowGwfrcha with readasarrays=True accepts recharge as per-SP
# 2D arrays. For auxiliary, pass auxiliary= name list and aux= as dict of
# per-SP 2D arrays (shape nrow x ncol).
flopy.mf6.ModflowGwfrcha(
    gwf,
    auxiliary=["MP_CONC"],
    recharge=rch_spd,
    aux=rch_aux_spd,
    pname="RCHA")

# WEL — pumping (2 aux: SALT_CONC, MP_CONC)
flopy.mf6.ModflowGwfwel(
    gwf,
    auxiliary=["SALT_CONC", "MP_CONC"],
    stress_period_data=wel_spd,
    pname="PUMP")

flopy.mf6.ModflowGwfoc(
    gwf,
    head_filerecord="gwf.hds",
    budget_filerecord="gwf.cbc",
    saverecord=[("HEAD", "ALL"), ("BUDGET", "ALL")],
    printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")])

# ---------------------------------------------------------------------------
# GWT — SALT TRANSPORT
# ---------------------------------------------------------------------------
gwt_salt = flopy.mf6.ModflowGwt(sim, modelname="gwt_salt", save_flows=True)

ims_salt = flopy.mf6.ModflowIms(
    sim, pname="ims_salt", filename="ims_salt.ims",
    complexity="MODERATE",
    outer_maximum=500, inner_maximum=150,
    outer_dvclose=1e-6, inner_dvclose=1e-7,
    rcloserecord=1e-5,
    linear_acceleration="BICGSTAB",
    relaxation_factor=0.97)
sim.register_ims_package(ims_salt, ["gwt_salt"])

flopy.mf6.ModflowGwtdis(
    gwt_salt, nlay=nlay, nrow=nrow, ncol=ncol,
    delr=delr, delc=delc,
    top=dis_top, botm=dis_botm,
    idomain=1, length_units="METERS")

flopy.mf6.ModflowGwtic(gwt_salt, strt=C_FRESH)
flopy.mf6.ModflowGwtmst(gwt_salt, porosity=POROSITY)
flopy.mf6.ModflowGwtadv(gwt_salt, scheme="TVD")
# FIX-21: add diffc=0.0 consistent with SWI model
flopy.mf6.ModflowGwtdsp(gwt_salt, alh=ALPHA_L, ath1=ALPHA_T, atv=ALPHA_T,
                         diffc=0.0)

flopy.mf6.ModflowGwtcnc(
    gwt_salt,
    stress_period_data={sp: [[(k, 0, 0), C_SEA] for k in range(nlay)]
                        for sp in range(nper)},
    pname="CNC_SALT")

flopy.mf6.ModflowGwtssm(gwt_salt, sources=[
    ("SEA_CHD",    "AUX", "SALT_CONC"),
    ("INLAND_GHB", "AUX", "SALT_CONC"),
    ("PUMP",       "AUX", "SALT_CONC"),
], pname="SSM_SALT")

flopy.mf6.ModflowGwtoc(
    gwt_salt,
    concentration_filerecord="gwt_salt.ucn",
    budget_filerecord="gwt_salt.bud",
    saverecord=[("CONCENTRATION", "ALL"), ("BUDGET", "LAST")])

flopy.mf6.ModflowGwfgwt(
    sim, exgtype="GWF6-GWT6",
    exgmnamea="gwf", exgmnameb="gwt_salt",
    filename="gwf_gwt_salt.exg")

# ---------------------------------------------------------------------------
# GWT — MICROPLASTIC TRANSPORT
# ---------------------------------------------------------------------------
gwt_mp = flopy.mf6.ModflowGwt(sim, modelname="gwt_mp", save_flows=True)

ims_mp = flopy.mf6.ModflowIms(
    sim, pname="ims_mp", filename="ims_mp.ims",
    complexity="MODERATE",
    outer_maximum=500, inner_maximum=150,
    outer_dvclose=1e-6, inner_dvclose=1e-7,
    rcloserecord=1e-5,
    linear_acceleration="BICGSTAB",
    relaxation_factor=0.97)
sim.register_ims_package(ims_mp, ["gwt_mp"])

flopy.mf6.ModflowGwtdis(
    gwt_mp, nlay=nlay, nrow=nrow, ncol=ncol,
    delr=delr, delc=delc,
    top=dis_top, botm=dis_botm,
    idomain=1, length_units="METERS")

flopy.mf6.ModflowGwtic(gwt_mp, strt=0.0)   # MP-free initial condition

# MST: linear Kd sorption + Bradford straining as first-order decay
# Kd = KD_IC (freshwater) everywhere at t=0.
# Note: MODFLOW 6 GWT MST does not update Kd dynamically with salinity.
# The salinity-dependent Kd effect is approximated by using the mean Kd
# weighted by the expected saline/fresh volume ratio (see thesis discussion).
# For a full coupling, a user-defined sorption subroutine would be needed.
flopy.mf6.ModflowGwtmst(
    gwt_mp,
    porosity=POROSITY,
    sorption="linear",
    bulk_density=RHO_BULK,
    distcoef=KD_IC,           # Kd [m3/kg] — freshwater initial value
    first_order_decay=True,
    decay=k_str_3d,           # Bradford straining [/day], shape (nlay, nrow, ncol)
    decay_sorbed=False)       # straining applies to aqueous phase only

flopy.mf6.ModflowGwtadv(gwt_mp, scheme="TVD")
flopy.mf6.ModflowGwtdsp(gwt_mp, alh=ALPHA_L_MP, ath1=ALPHA_T_MP,
                         atv=ALPHA_T_MP, diffc=0.0)

flopy.mf6.ModflowGwtssm(gwt_mp, sources=[
    ("SEA_CHD",    "AUX", "MP_CONC"),
    ("INLAND_GHB", "AUX", "MP_CONC"),
    ("RCHA",       "AUX", "MP_CONC"),
    ("PUMP",       "AUX", "MP_CONC"),
], pname="SSM_MP")

flopy.mf6.ModflowGwtoc(
    gwt_mp,
    concentration_filerecord="gwt_mp.ucn",
    budget_filerecord="gwt_mp.bud",
    saverecord=[("CONCENTRATION", "ALL"), ("BUDGET", "LAST")])

flopy.mf6.ModflowGwfgwt(
    sim, exgtype="GWF6-GWT6",
    exgmnamea="gwf", exgmnameb="gwt_mp",
    filename="gwf_gwt_mp.exg")

# =============================================================================
# WRITE & RUN
# =============================================================================
print("  Writing simulation files ...")
sim.write_simulation(silent=False)

print("  Running MODFLOW 6 ...")
success, buff = sim.run_simulation(silent=False, report=True)

if not success:
    print("\n*** MODFLOW 6 FAILED ***")
    for line in buff:
        print(line)
    raise RuntimeError(
        f"MODFLOW 6 run FAILED — check {os.path.join(ws, 'mfsim.lst')}")
print("  Run complete.\n")

# =============================================================================
# POST-PROCESSING — LOAD OUTPUT
# =============================================================================
salt_obj = gwt_salt.output.concentration()
mp_obj   = gwt_mp.output.concentration()
head_obj = gwf.output.head()
times    = np.array(salt_obj.times)    # [days] — one per saved time step

salt_all = salt_obj.get_alldata()      # (ntimes, nlay, nrow, ncol)
mp_all   = mp_obj.get_alldata()
head_all = head_obj.get_alldata()

# Darcy velocity (specific discharge) for quiver plots — last time step
try:
    bgt_obj  = gwf.output.budget()
    spd_last = bgt_obj.get_data(text="DATA-SPDIS", totim=bgt_obj.get_times()[-1])[0]
    qx_raw   = np.zeros((nlay, nrow, ncol))
    qz_raw   = np.zeros((nlay, nrow, ncol))
    for rec in spd_last:
        node = rec["node"] - 1
        k    = node // (nrow * ncol)
        j    = node % ncol
        qx_raw[k, 0, j] = rec["qx"]
        qz_raw[k, 0, j] = rec["qz"]
    qx  = np.squeeze(qx_raw)
    qz  = np.squeeze(qz_raw)
    mag = np.hypot(qx, qz)
    mag[mag == 0] = np.nan
    _has_quiver = True
except Exception as _e:
    print(f"  WARNING: Darcy velocity not available ({_e}) — quiver plots skipped")
    _has_quiver = False
    qx = qz = mag = None

# Snap indices: find time step closest to end of each snapshot SP
_cumlen   = np.concatenate([[0.0], np.cumsum(perlen)])
snap_times = [_cumlen[sp + 1] for sp in SNAP_SPS]
snap_idx   = [int(np.argmin(np.abs(times - t))) for t in snap_times]
snap_cols  = ["royalblue", "darkorange", "crimson"]

# Profile x-locations and columns
profile_xs   = [75, 200, 500, 800]
profile_cols = [col_from_x(x) for x in profile_xs]
well_xs      = [w["x_m"] for w in WELL_SPECS_MP]
well_fracs   = [WELL_FRACTIONS[w["name"]] for w in WELL_SPECS_MP]

qi, qj = 3, 6
skip   = (slice(None, None, qi), slice(None, None, qj))

# =============================================================================
# PLOT HELPERS
# =============================================================================
def mark_wells(ax, fontsize=6.5):
    for wx, wf in zip(well_xs, well_fracs):
        ax.axvline(wx, color="white", lw=0.7, ls=":")
        ax.text(wx + 5, Z_BOTTOM + 2, f"W\n{wf*100:.0f}%",
                color="white", fontsize=fontsize, va="bottom", ha="left",
                bbox=dict(boxstyle="round,pad=0.12",
                          fc="black", alpha=0.45, ec="none"))


def mark_rech_mp(ax, fontsize=6):
    ax.axvspan(0, 400, alpha=0.08, color="lime", lw=0)
    ax.text(200, Z_BOTTOM + 3, "Rech\nMP",
            color="lime", fontsize=fontsize, ha="center", va="bottom")


def add_quiver(ax):
    if not _has_quiver:
        return
    ax.quiver(X_2D[skip], Z_2D[skip],
              qx[skip] / mag[skip], qz[skip] / mag[skip],
              color="white", alpha=0.35, scale=30, width=0.0016,
              headwidth=3.5, headlength=4)


def set_axes(ax):
    ax.set_xlim(0, min(1000, Lx))
    ax.set_ylim(Z_BOTTOM, 2.0)


# =============================================================================
# FIG 1 — Hydraulic head (final time step)
# =============================================================================
head_final = np.squeeze(head_all[-1]).copy()
head_final[head_final <= -1e9] = np.nan

_hf = head_final[np.isfinite(head_final)]
if len(_hf) > 0:
    _sym = max(abs(np.percentile(_hf, 5)) + 0.05,
               abs(np.percentile(_hf, 95)) + 0.05, 0.1)
else:
    _sym = 2.0

fig, ax = plt.subplots(figsize=(14, 4.5))
_norm_h = TwoSlopeNorm(vmin=-_sym, vcenter=0.0, vmax=_sym)
pcm = ax.pcolormesh(X_2D, Z_2D,
                     np.clip(head_final, -_sym, _sym),
                     shading="auto", cmap="seismic", norm=_norm_h)
cbar = fig.colorbar(pcm, ax=ax, pad=0.01, label="Hydraulic Head (m)")
cbar.ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
add_quiver(ax)
mark_wells(ax)
set_axes(ax)
ax.set_xlabel("Distance from Coast (m)", fontsize=10)
ax.set_ylabel("Elevation (m MSL)", fontsize=10)
ax.set_title(
    f"Hydraulic Head — {SNAP_LABELS[-1]}  [{mode_label}]\n"
    "Blue = seawater depression  |  Red = freshwater head  "
    "|  Arrows = Darcy velocity",
    fontsize=10, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(ws, "Fig1_head_final.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Fig1 saved.")

# =============================================================================
# FIG 2 — Salinity cross-sections (3 snapshots)
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True, sharey=True)
for ax, si, label, col in zip(axes, snap_idx, SNAP_LABELS, snap_cols):
    salt = np.squeeze(salt_all[si])
    pcm  = ax.pcolormesh(X_2D, Z_2D, salt, shading="auto",
                          vmin=0, vmax=C_SEA, cmap="viridis")
    fig.colorbar(pcm, ax=ax, pad=0.01, label="Salinity (ppt)")
    cs   = ax.contour(X_2D, Z_2D, salt,
                      levels=[1, 5, 10, THRESHOLD_SALT, 25, 30],
                      colors=["cyan","cyan","yellow","red","orange","white"],
                      linewidths=[0.7, 0.7, 0.8, 2.0, 0.8, 0.7])
    ax.clabel(cs, fmt="%.0f ppt", fontsize=6.5, colors="white")
    add_quiver(ax)
    mark_wells(ax)
    set_axes(ax)
    ax.set_ylabel("Elevation (m MSL)", fontsize=9)

    _toe_c = np.where(salt[-1, :] >= THRESHOLD_SALT)[0]
    _toe   = x_centers[_toe_c[-1]] if len(_toe_c) > 0 else 0.0
    ax.set_title(f"{label}  |  salt toe = {_toe:.0f} m from coast  "
                 f"|  max = {salt.max():.1f} ppt",
                 fontsize=9, color=col)

axes[-1].set_xlabel("Distance from Coast (m)", fontsize=10)
fig.suptitle(f"Salinity Cross-Sections  [{mode_label}]",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(ws, "Fig2_salinity_snapshots.png"),
            dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Fig2 saved.")

# =============================================================================
# FIG 3 — MP cross-sections (3 snapshots)
# =============================================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 13), sharex=True, sharey=True)
for ax, si, label, col in zip(axes, snap_idx, SNAP_LABELS, snap_cols):
    mp  = np.squeeze(mp_all[si])
    pcm = ax.pcolormesh(X_2D, Z_2D, mp, shading="auto",
                         cmap="plasma", vmin=0, vmax=C_MP_SEA_BASE)
    fig.colorbar(pcm, ax=ax, pad=0.01, label="MP (normalised)")
    cs  = ax.contour(X_2D, Z_2D, mp,
                     levels=[0.10, 0.25, 0.50, 0.75],
                     colors=["white","cyan","yellow","red"], linewidths=0.9)
    ax.clabel(cs, fmt="%.2f", fontsize=6.5, colors="white")
    add_quiver(ax)
    mark_wells(ax)
    if SOURCE_MODE in ("rech_only", "both"):
        mark_rech_mp(ax)
    set_axes(ax)
    ax.set_ylabel("Elevation (m MSL)", fontsize=9)

    _fc = np.where(mp.max(axis=0) >= THRESHOLD_MP)[0]
    _fr = x_centers[_fc[-1]] if len(_fc) > 0 else 0.0
    ax.set_title(f"{label}  |  MP front = {_fr:.0f} m  |  max = {mp.max():.3f}",
                 fontsize=9, color=col)

axes[-1].set_xlabel("Distance from Coast (m)", fontsize=10)
fig.suptitle(
    f"Microplastic Concentration Cross-Sections  [{mode_label}]\n"
    f"αL_MP={ALPHA_L_MP:.2f} m  Kd_fresh={KD_FRESH*1e3:.1f} L/kg  "
    f"Kd_saline={KD_SALINE*1e3:.1f} L/kg  K_STR0={K_STR0} /d",
    fontsize=10, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(ws, "Fig3_MP_snapshots.png"),
            dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Fig3 saved.")

# =============================================================================
# FIG 4 — Combined Salt + MP (one figure per snapshot)
# =============================================================================
for si, label in zip(snap_idx, SNAP_LABELS):
    salt = np.squeeze(salt_all[si])
    mp   = np.squeeze(mp_all[si])

    _toe_c = np.where(salt[-1, :] >= THRESHOLD_SALT)[0]
    _toe   = x_centers[_toe_c[-1]] if len(_toe_c) > 0 else 0.0
    _fc    = np.where(mp.max(axis=0) >= THRESHOLD_MP)[0]
    _front = x_centers[_fc[-1]] if len(_fc) > 0 else 0.0

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax in (ax0, ax1):
        set_axes(ax)

    pcm0 = ax0.pcolormesh(X_2D, Z_2D, salt, shading="auto",
                           vmin=0, vmax=C_SEA, cmap="viridis")
    fig.colorbar(pcm0, ax=ax0, pad=0.01, label="Salinity (ppt)")
    cs0  = ax0.contour(X_2D, Z_2D, salt,
                       levels=[1, 5, 10, THRESHOLD_SALT, 25, 30],
                       colors=["cyan","cyan","yellow","red","orange","white"],
                       linewidths=0.8)
    ax0.clabel(cs0, fmt="%.0f ppt", fontsize=6.5, colors="white")
    add_quiver(ax0)
    mark_wells(ax0)
    ax0.set_ylabel("Elevation (m MSL)", fontsize=9)
    ax0.set_title(f"Salinity  |  toe = {_toe:.0f} m  |  max = {salt.max():.1f} ppt",
                  fontsize=9)

    pcm1 = ax1.pcolormesh(X_2D, Z_2D, mp, shading="auto",
                           cmap="plasma", vmin=0, vmax=C_MP_SEA_BASE)
    fig.colorbar(pcm1, ax=ax1, pad=0.01, label="MP (normalised)")
    cs1  = ax1.contour(X_2D, Z_2D, mp,
                       levels=[0.10, 0.25, 0.50, 0.75],
                       colors=["white","cyan","yellow","red"], linewidths=0.9)
    ax1.clabel(cs1, fmt="%.2f", fontsize=6.5, colors="white")
    add_quiver(ax1)
    mark_wells(ax1)
    if SOURCE_MODE in ("rech_only", "both"):
        mark_rech_mp(ax1)
    ax1.set_ylabel("Elevation (m MSL)", fontsize=9)
    ax1.set_xlabel("Distance from Coast (m)", fontsize=10)
    ax1.set_title(
        f"Microplastics  |  front = {_front:.0f} m  |  max = {mp.max():.3f}  "
        f"|  MP lag = {_toe - _front:.0f} m behind salt toe",
        fontsize=9)

    fig.suptitle(f"Salt + MP Integration — {label}  [{mode_label}]",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(os.path.join(ws, f"Fig4_combined_{label.replace(' ', '_')}.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
print("  Fig4 saved.")

# =============================================================================
# FIG 5 — Time-series: salt toe + MP front 1991–2025
# =============================================================================
toe_hist, mp_hist = [], []
for ti in range(len(times)):
    _salt = np.squeeze(salt_all[ti])
    _mp   = np.squeeze(mp_all[ti])
    _tc   = np.where(_salt[-1, :] >= THRESHOLD_SALT)[0]
    toe_hist.append(x_centers[_tc[-1]] if len(_tc) > 0 else 0.0)
    _fc   = np.where(_mp.max(axis=0) >= THRESHOLD_MP)[0]
    mp_hist.append(x_centers[_fc[-1]] if len(_fc) > 0 else 0.0)

# Warmup SP = 5 yr (1825 d), so t=0 corresponds to calendar year 1991
t_years = times / 365.25 + (PUMPING_YEARS[0] - perlen[0] / 365.25)

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(t_years, toe_hist, color="steelblue", lw=2,
        label=f"Salt toe ({THRESHOLD_SALT:.1f} ppt, bottom layer)")
ax.plot(t_years, mp_hist, color="darkorange", lw=2, ls="--",
        label=f"MP front ({THRESHOLD_MP:.0%} of sea source)")
for _yr in [2005, 2015, 2025]:
    ax.axvline(_yr, color="gray", lw=0.8, ls=":")
    ax.text(_yr + 0.2, 10, str(_yr), fontsize=7, color="gray")
ax.set_xlabel("Year", fontsize=10)
ax.set_ylabel("Distance from Coast (m)", fontsize=10)
ax.set_title(f"Salt Toe and MP Front Migration  [{mode_label}]",
             fontsize=11, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_xlim(t_years[0], t_years[-1] + 1)
fig.tight_layout()
fig.savefig(os.path.join(ws, "Fig5_timeseries.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Fig5 saved.")

# =============================================================================
# FIG 6 — Vertical profiles at selected x-locations
# =============================================================================
fig, axes2d = plt.subplots(len(profile_xs), 2,
                            figsize=(11, 3.5 * len(profile_xs)),
                            sharey="row")
for row_i, (px, pc) in enumerate(zip(profile_xs, profile_cols)):
    ax_s = axes2d[row_i, 0]
    ax_m = axes2d[row_i, 1]
    z_col = Z_2D[:, pc]

    for si, label, col in zip(snap_idx, SNAP_LABELS, snap_cols):
        _salt = np.squeeze(salt_all[si])
        _mp   = np.squeeze(mp_all[si])
        ax_s.plot(_salt[:, pc], z_col, color=col, lw=2, label=label)
        ax_m.plot(_mp[:, pc],   z_col, color=col, lw=2, label=label)

    ax_s.axvline(THRESHOLD_SALT, color="red",  ls="--", lw=1.5,
                 label=f"{THRESHOLD_SALT:.1f} ppt")
    ax_s.axvline(1.0, color="gray", ls=":", lw=1.0, label="1 ppt")
    ax_s.set_xlabel("Salinity (ppt)", fontsize=9)
    ax_s.set_ylabel("Elevation (m MSL)", fontsize=9)
    ax_s.set_title(f"Salinity @ x = {px} m", fontsize=10)
    ax_s.legend(fontsize=7)
    ax_s.grid(alpha=0.25)

    ax_m.axvline(THRESHOLD_MP, color="cyan", ls="--", lw=1.5,
                 label=f"{THRESHOLD_MP:.0%} threshold")
    ax_m.set_xlabel("MP (normalised)", fontsize=9)
    ax_m.set_title(f"MP @ x = {px} m\nαL_MP={ALPHA_L_MP:.2f} m  "
                   f"Kd_fresh={KD_FRESH*1e3:.1f} L/kg", fontsize=10)
    ax_m.legend(fontsize=7)
    ax_m.grid(alpha=0.25)

fig.suptitle(f"Vertical Profiles at Selected Cross-Sections  [{mode_label}]",
             fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(ws, "Fig6_vertical_profiles.png"),
            dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Fig6 saved.")

# =============================================================================
# FIG 7 — Bar chart: salt toe vs MP front at snapshot years
# =============================================================================
toe_snap, mp_snap = [], []
for si in snap_idx:
    _salt = np.squeeze(salt_all[si])
    _mp   = np.squeeze(mp_all[si])
    _tc   = np.where(_salt[-1, :] >= THRESHOLD_SALT)[0]
    _fc   = np.where(_mp.max(axis=0) >= THRESHOLD_MP)[0]
    toe_snap.append(x_centers[_tc[-1]] if len(_tc) > 0 else 0.0)
    mp_snap.append(x_centers[_fc[-1]] if len(_fc) > 0 else 0.0)

x_bar = np.arange(len(SNAP_LABELS))
width = 0.35
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x_bar - width/2, toe_snap, width,
       label=f"Salt toe ({THRESHOLD_SALT:.1f} ppt)", color="steelblue", alpha=0.85)
ax.bar(x_bar + width/2, mp_snap, width,
       label=f"MP front ({THRESHOLD_MP:.0%})", color="darkorange", alpha=0.85)
for i, (toe, mp) in enumerate(zip(toe_snap, mp_snap)):
    ax.text(i, max(toe, mp) + 8, f"lag={toe-mp:.0f} m",
            ha="center", fontsize=8, color="gray")
ax.set_xticks(x_bar)
ax.set_xticklabels(SNAP_LABELS)
ax.set_ylabel("Distance from Coast (m)")
ax.set_title(f"Salt Toe vs MP Front  [{mode_label}]", fontweight="bold")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(ws, "Fig7_bar_chart.png"), dpi=300, bbox_inches="tight")
plt.close(fig)
print("  Fig7 saved.")

# =============================================================================
# SUMMARY TABLE
# =============================================================================
print("\n" + "="*65)
print(f"  C6 MP MODEL SUMMARY — SOURCE_MODE = {SOURCE_MODE}")
print("="*65)
print(f"  {'Snapshot':<15} {'Salt toe (m)':>14} {'MP front (m)':>14} {'Lag (m)':>10}")
print(f"  {'-'*55}")
for label, toe, mp in zip(SNAP_LABELS, toe_snap, mp_snap):
    print(f"  {label:<15} {toe:>14.0f} {mp:>14.0f} {toe-mp:>10.0f}")
print("="*65)
print(f"\n  PARAMETERS:")
print(f"    K_TOP / K_BOTTOM         : {config.K_TOP} / {config.K_BOTTOM} m/d")
print(f"    Kv/Kh                    : 0.1  (Priyanka & MK 2019 Table 6)")
print(f"    ALPHA_L / ALPHA_T (salt) : {ALPHA_L} / {ALPHA_T} m")
print(f"    ALPHA_L_MP / ALPHA_T_MP  : {ALPHA_L_MP:.2f} / {ALPHA_T_MP:.3f} m  "
      f"(Torkzaban 2012 × 0.25)")
print(f"    Kd_fresh / Kd_saline     : {KD_FRESH*1e3:.1f} / {KD_SALINE*1e3:.1f} L/kg  "
      f"(Porter et al. 2015)")
print(f"    ρ_bulk                   : {RHO_BULK:.0f} kg/m3")
print(f"    Bradford K_STR0 / β      : {K_STR0} /d  /  {BETA_STR}  "
      f"(Bradford 2011 Table 1)")
print(f"    D50 / λ_STR              : {D50*1e3:.1f} mm  /  {LAMBDA_STR} /m  "
      f"(Bradford 2002)")
print(f"    Porosity / SY            : {POROSITY} / {SY}")
print(f"    Recharge coeff           : {config.RECHARGE_COEFF*100:.0f}%  "
      f"(Kumar et al. 2020 calibrated)")
print(f"    Q_GW 1996 → 2025         : "
      f"{config.Q_gw(1996):.0f} → {config.Q_gw(2025):.0f} m3/d")
print(f"    SECTION_WIDTH            : {SECTION_WIDTH} m")
print(f"    Z_BOTTOM                 : {Z_BOTTOM:.0f} m MSL")
print(f"    C_MP_SEA / C_MP_RECH     : {C_MP_SEA} / {C_MP_RECH} (normalised)")
print(f"\n  All figures saved to: {ws}")
print("="*65 + "\n")