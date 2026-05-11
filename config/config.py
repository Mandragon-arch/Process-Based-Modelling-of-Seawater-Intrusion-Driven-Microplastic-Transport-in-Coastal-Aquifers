# =============================================================================
# config.py  —  SWI PROJECT  |  NITK / Surathkal Campus Aquifer
# =============================================================================
# Single source of truth for ALL parameters shared across C1–C5b.
# Every other script does:  import config
#
# This file:
#   1. Reads rainfall CSV (1960–2025, all years included)
#   2. Reads input_ghb.csv (1996–2025, all years included)
#      HARD STOP if file missing or any year/season entry absent
#
# Aquifer: Laterite/coastal sandy unit, 0 to -40 m MSL.
# Base at -40 m MSL: below lies fractured Deccan basalt / granitic basement
# — not modelled. The laterite aquifer (0 to -40 m) is the sole modelled unit.
#
# x = 0 at COAST (sea side)   |   x = 1500 m at INLAND boundary
# =============================================================================

import os
import csv
import numpy as np

# =============================================================================
# PATHS  —  all relative to SWI_project/
# Set PyCharm working directory to SWI_project/ for all run configurations.
# =============================================================================
_HERE            = os.path.dirname(os.path.abspath(__file__))
GHB_CSV          = os.path.join(_HERE, "input ghb.csv")
RAINFALL_CSV     = os.path.join(_HERE, "rainfall COASTAL KARNATAKA 1960-2017.csv")
RESULTS_DIR      = os.path.join(_HERE, "results")
WORKSPACE_BASE   = os.path.join(_HERE, "modflow_workspace", "base")
WORKSPACE_SENS   = os.path.join(_HERE, "modflow_workspace", "ghb_sens")
WORKSPACE_SLR    = os.path.join(_HERE, "modflow_workspace", "slr")


os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================================================================
# MODFLOW EXECUTABLE & MODEL NAMES
# =============================================================================
MF6_EXE  = r"C:\WRDAPP\mf6.6.2_win64\bin\mf6.exe"
GWF_NAME = "gwf_swi"
GWT_NAME = "gwt_swi"

# =============================================================================
# GRID
# 120 columns (fine at coast, coarsening inland) x 40 layers x 1 row
# Aquifer: 0 to -40 m MSL
# =============================================================================
Lx, Lz          = 1500.0, 40.0
nlay, nrow, ncol = 40, 1, 120

dx_raw  = np.geomspace(5.0, 20.0, ncol)
dx_raw *= Lx / dx_raw.sum()
delr    = dx_raw
delc    = np.array([1.0])

dz      = Lz / nlay          # 1 m per layer
tops    = np.zeros(nlay)
bots    = np.zeros(nlay)
tops[0] =  0.0
bots[0] = -dz
for k in range(1, nlay):
    tops[k] = bots[k - 1]
    bots[k] = tops[k] - dz

x_centers    = np.cumsum(delr) - delr / 2.0
z_mid_layers = (tops + bots) / 2.0

# =============================================================================
# HYDRAULIC CONDUCTIVITY  K(x, z)
# Sigmoid in depth: K DECREASES with depth — surface laterite is most permeable,
# compact deep zone least permeable.
#
# Source: Priyanka B.N. & Mohan Kumar M.S. (2019), Water 11:421,
#         Three-Dimensional Modelling of Heterogeneous Coastal Aquifer,
#         Pavanje basin, Surathkal (this exact aquifer).
#         Layer 1 (surface) lognormal mean K = exp(3.473) ≈ 32 m/d
#         Layer 3 (deep)    lognormal mean K = exp(2.024) ≈  8 m/d
#         Full range: 5.4–84.5 m/d (Layer 1) to 5.4–9.9 m/d (Layer 3)
#
# K_TOP    = 20 m/d  — weathered surface laterite (0 to ~-10 m MSL).
#                       Priyanka & Mohan Kumar (2019) Layer 1 lognormal mean ~32 m/d
#                       but calibrated pumping test zone values (Kumar et al. 2020,
#                       Table 4) range 10-28 m/d; 20 m/d is the zone 3 calibrated
#                       value and represents the effective bulk hydraulic conductivity
#                       for the laterite surface layer in this 2D model.
# K_BOTTOM =  7 m/d  — compact deep laterite transitioning to gneiss bedrock
#                       (~-30 to -40 m MSL); fractured bedrock below -40 m not modelled
# Z0_SIG   = -20 m   — inflection point of sigmoid (mid-depth transition zone)
# B_SIG    =  0.3    — sigmoid steepness (gentle transition over ~15 m)
# Mild tanh horizontal variation (factor 0.8-1.2) retained from original.
# =============================================================================
K_TOP, K_BOTTOM, Z0_SIG, B_SIG = 20.0, 7.0, -20.0, 0.3

# Sigmoid: K_BOTTOM + (K_TOP - K_BOTTOM) / (1 + exp(-B*(z-Z0)))
# At z=0 (surface, z_mid=-0.5): K -> K_TOP (20 m/d)
# At z=-40 (deep, z_mid=-39.5): K -> K_BOTTOM (7 m/d)
# This correctly represents decreasing-with-depth K for a laterite aquifer.
Kz     = K_BOTTOM + (K_TOP - K_BOTTOM) / (1.0 + np.exp(-B_SIG * (z_mid_layers - Z0_SIG)))
Kx     = 1.0 + 0.2 * np.tanh((x_centers - Lx / 2.0) / (Lx / 4.0))
K33_3D = Kz[:, np.newaxis, np.newaxis] * Kx[np.newaxis, np.newaxis, :]

assert K33_3D.min() > 0, "K field contains non-positive values."
print(f"  [config] K(x,z): Min={K33_3D.min():.2f}  Max={K33_3D.max():.2f} m/d")

SY = 0.20    # specific yield
SS = 1e-5    # specific storage [1/m]

# =============================================================================
# TRANSPORT & DENSITY PARAMETERS
# =============================================================================
POROSITY = 0.30
# Longitudinal dispersivity — Xu & Eckstein (1995) formula applied to Pavanje basin:
#   alphaL = 0.83 x (log10 L)^2.414   (L = field length from source, valid to 1000 m)
# Applied across our 0-1500 m domain:
#   L=75 m -> 4.1 m  |  L=500 m -> 9.6 m  |  L=1000 m -> 11.77 m (formula cap)
#   Domain-weighted mean: (1000 x 8.7 + 500 x 11.77) / 1500 = 9.7 m -> rounded to 10.0 m
# Source: Priyanka B.N. & Mohan Kumar M.S. (2019), Water 11:421, Table 4.
#         Kriged alphaL field for this exact Pavanje basin aquifer ranged 4.96-11.9 m.
# alphaT = 0.1 x alphaL per Priyanka & Mohan Kumar (2019) Table 2 standard ratio.
ALPHA_L  = 10.0   # longitudinal dispersivity [m]
ALPHA_T  =  1.0   # transverse dispersivity [m]  (= 0.1 x ALPHA_L)
RHO_REF  = 1000.0 # reference (fresh) water density [kg/m3]
DRHODC   = 0.7    # density slope [kg/m3 per ppt]
C_SEA    = 35.0   # seawater salinity [ppt] -- field-measured OND 2024-25
C_FRESH  = 0.0    # freshwater salinity [ppt]

RHO_SEA  = RHO_REF + DRHODC * C_SEA
GH_RATIO = RHO_REF / (RHO_SEA - RHO_REF)
print(f"  [config] rho_sea={RHO_SEA:.2f} kg/m3  |  GH ratio={GH_RATIO:.2f}")

# =============================================================================
# SEASONS
# JF=59 d  MAM=92 d  JJAS=122 d  OND=92 d  (total = 365 d)
# =============================================================================
SEASONS       = ['JF', 'MAM', 'JJAS', 'OND']
SEASON_DAYS   = {'JF': 59, 'MAM': 92, 'JJAS': 122, 'OND': 92}
SEASON_OFFSET = {'JF': 0.08, 'MAM': 0.30, 'JJAS': 0.58, 'OND': 0.83}

# =============================================================================
# RECHARGE
# 15% of seasonal rainfall -> recharge.
# Source: GEC (1997) Groundwater Resource Estimation Methodology, Ministry of
#         Water Resources, GoI -- west coast India laterite norm = 8-15%.
#         Kumar S.S., Deb Barma S., Mahesha A. (2020), J. Earth Syst. Sci.
#         129:66 -- calibrated value of 10% for this exact Pavanje basin aquifer.
# Previous value of 15% was the upper end of the broader Karnataka laterite
# range; replaced by site-specific calibrated literature value.
# =============================================================================
RECHARGE_COEFF = 0.15

# =============================================================================
# SEASONAL RAINFALL  --  read from CSV (1960-2025, all rows in one file)
# CSV columns: SUBDIVISION, YEAR, ANNUAL, JF, MAM, JJAS, OND
# Rows for 2018-2025 are appended directly in the CSV.
# =============================================================================
_rain_csv = {}

def _parse_float(val):
    try:
        v = float(val)
        return v if not np.isnan(v) else None
    except (ValueError, TypeError):
        return None

with open(RAINFALL_CSV, newline='', encoding='utf-8') as _f:
    _reader = csv.DictReader(_f)
    for _row in _reader:
        try:
            _yr = int(float(_row['YEAR']))
        except (ValueError, KeyError):
            continue
        _jf   = _parse_float(_row.get('JF'))
        _mam  = _parse_float(_row.get('MAM'))
        _jjas = _parse_float(_row.get('JJAS'))
        _ond  = _parse_float(_row.get('OND'))
        _ann  = _parse_float(_row.get('ANNUAL'))
        if any(v is not None for v in [_jf, _mam, _jjas, _ond]):
            _rain_csv[_yr] = {
                'JF':    _jf   if _jf   is not None else 0.0,
                'MAM':   _mam  if _mam  is not None else 0.0,
                'JJAS':  _jjas if _jjas is not None else 0.0,
                'OND':   _ond  if _ond  is not None else 0.0,
                'ANNUAL': _ann,
            }

_rain_years = sorted(_rain_csv.keys())
print(f"  [config] Rainfall loaded: {len(_rain_years)} years "
      f"({_rain_years[0]}-{_rain_years[-1]})")

def get_rainfall(year, season):
    """Return seasonal rainfall [mm] for given year and season."""
    return _rain_csv[year][season]

def get_recharge(year, season):
    """Return recharge rate [m/day] for MODFLOW RCH package."""
    return _rain_csv[year][season] * 0.001 * RECHARGE_COEFF / SEASON_DAYS[season]

# Long-term mean recharge (1960-2017) used for warm-up stress period
_yrs_lt   = [y for y in range(1960, 2018) if y in _rain_csv]
LT_MEAN_R = {s: float(np.mean([get_recharge(y, s) for y in _yrs_lt]))
             for s in SEASONS}
WARMUP_R  = (sum(LT_MEAN_R[s] * SEASON_DAYS[s] for s in SEASONS)
             / sum(SEASON_DAYS.values()))

# =============================================================================
# GHB BOUNDARY -- READ CSV  (HARD STOP IF MISSING OR INCOMPLETE)
# CSV format: wide -- Year | JF | MAM | JJAS | OND
# Row 1 is a title row (,GHB Final,,,) -- skipped explicitly.
# Row 2 is the real header: Year,JF,MAM,JJAS,OND
# All years 1996-2025 and all 4 seasons must be present.
# =============================================================================
if not os.path.isfile(GHB_CSV):
    raise FileNotFoundError(
        f"\n{'='*70}\n"
        f"  HARD STOP -- input_ghb.csv not found.\n"
        f"  Expected location: {GHB_CSV}\n"
        f"  Place input_ghb.csv in SWI_project/ and re-run.\n"
        f"{'='*70}"
    )

_ghb_raw = {}
with open(GHB_CSV, newline='', encoding='utf-8') as _f:
    next(_f)                      # skip title row: ",GHB Final,,,"
    _reader = csv.DictReader(_f)  # real header: Year,JF,MAM,JJAS,OND
    for _row in _reader:
        try:
            _yr = int(float(_row['Year']))
        except (ValueError, KeyError):
            continue
        _heads = {}
        for _s in SEASONS:
            _v = _parse_float(_row.get(_s))
            if _v is not None:
                _heads[_s] = _v
        if _heads:
            _ghb_raw[_yr] = _heads

_csv_years = sorted(_ghb_raw.keys())
print(f"  [config] GHB CSV loaded: {len(_csv_years)} years "
      f"({_csv_years[0]}-{_csv_years[-1]})")

# Build lookup table and verify all 1996-2025 entries are present
_ghb_table = {}
for _yr, _heads in _ghb_raw.items():
    for _s, _h in _heads.items():
        _ghb_table[(_yr, _s)] = _h

_missing = [(y, s) for y in range(1996, 2026) for s in SEASONS
            if (y, s) not in _ghb_table]
if _missing:
    raise ValueError(
        f"\n{'='*70}\n"
        f"  HARD STOP -- GHB table is missing {len(_missing)} entries.\n"
        f"  First missing: {_missing[:6]}\n"
        f"  Add these rows to input_ghb.csv and re-run.\n"
        f"{'='*70}"
    )
print(f"  [config] GHB table complete: 1996-2025, all 4 seasons. OK")

def get_ghb_head(year, season):
    """Return GHB head [m above MSL] for given year and season."""
    return _ghb_table[(year, season)]
# =============================================================================
# GHB CONDUCTANCE
# C_per_layer = K_regional * dz * delc / L_regional
# Base      : K=3.5 m/d, L=1200 m
# Sensitivity: K=7.0 m/d, L=1200 m
# =============================================================================
K_REGIONAL_BASE = 3.5
K_REGIONAL_SENS = 7.0
L_REGIONAL      = 1200.0

C_GHB_BASE = K_REGIONAL_BASE * dz * float(delc[0]) / L_REGIONAL
C_GHB_SENS = K_REGIONAL_SENS * dz * float(delc[0]) / L_REGIONAL

N_GHB_COLS  = 4
GHB_COL_IDS = list(range(ncol - N_GHB_COLS, ncol))

def make_ghb_list(h_s, c_per_layer=None):
    """Return GHB stress period list for given head and conductance."""
    if c_per_layer is None:
        c_per_layer = C_GHB_BASE
    return [[(k, 0, j), h_s, c_per_layer, C_FRESH]
            for j in GHB_COL_IDS for k in range(nlay)]

# =============================================================================
# COASTAL CHD  (x=0, all layers -- sea boundary)
# =============================================================================
def make_chd_sea(h_coast=0.0):
    """Return CHD stress period list for sea boundary."""
    return [[(k, 0, 0), h_coast, C_SEA] for k in range(nlay)]

def make_cnc_sea():
    """Return CNC stress period list for sea concentration boundary."""
    return [[(k, 0, 0), C_SEA] for k in range(nlay)]

# =============================================================================
# WELL SYSTEM  --  LOGISTIC DEMAND GROWTH 1996-2025
# Total pumping Q_GW(year) grows logistically from 270 to 720 m3/d.
# SECTION_WIDTH = 200 m (numerical 2D->3D conversion, not reported in thesis).
# =============================================================================
Q_GW_MIN      = 270.0
Q_GW_MAX      = 480.0
T_MID_GROWTH  = 2006.0
K_GROWTH      = 0.3
SECTION_WIDTH = 200.0   # numerical parameter -- not physical, not in thesis

def Q_gw(year):
    """Total pumping [m3/d] for given year (logistic growth)."""
    return Q_GW_MIN + (Q_GW_MAX - Q_GW_MIN) / (
        1.0 + np.exp(-K_GROWTH * (year - T_MID_GROWTH))
    )

WELL_FRACTIONS = {'W1': 0.06, 'W2': 0.12, 'W3': 0.26, 'W4': 0.32, 'W5': 0.24}
assert abs(sum(WELL_FRACTIONS.values()) - 1.0) < 1e-9

WELL_SCREENS = {
    'W1': list(range(3,  8)),    # layers 3-7  (~3-7 m depth) — raised to avoid saline zone
    'W2': list(range(5,  12)),   # layers 5-11 (~5-11 m depth) — raised to avoid saline zone
    'W3': list(range(12, 29)),   # layers 12-28
    'W4': list(range(14, 33)),   # layers 14-32
    'W5': list(range(12, 29)),   # layers 12-28
}

WELL_SPECS = [
    {"name": "W1", "x_m":  150.0},
    {"name": "W2", "x_m":  250.0},
    {"name": "W3", "x_m":  500.0},
    {"name": "W4", "x_m":  750.0},
    {"name": "W5", "x_m": 1000.0},
]
for _w in WELL_SPECS:
    _w["col"]     = int(np.argmin(np.abs(x_centers - _w["x_m"])))
    _w["screens"] = WELL_SCREENS[_w["name"]]
    _w["frac"]    = WELL_FRACTIONS[_w["name"]]

def make_wel_data(year):
    """Return WEL stress period list for given year. Returns [] for warm-up."""
    if year is None:
        return []
    Q = Q_gw(year)
    lst = []
    for w in WELL_SPECS:
        q_per_layer = -(Q * w["frac"] / SECTION_WIDTH) / len(w["screens"])
        for lyr in w["screens"]:
            lst.append([(lyr, 0, w["col"]), q_per_layer, C_FRESH])
    return lst

# =============================================================================
# STRESS PERIOD SCHEDULE
# SP 0     : 1825 d warm-up (5 yr, long-term mean recharge, no pumping)
# SP 1-120 : 1996-2025 x 4 seasons = 120 SPs
# Total    : 121 SPs
# =============================================================================
PUMPING_YEARS = list(range(1996, 2026))
N_PUMP_YEARS  = len(PUMPING_YEARS)

perlen = [1825.0]
for _yr in PUMPING_YEARS:
    for _s in SEASONS:
        perlen.append(float(SEASON_DAYS[_s]))

_nstp_season = {'JF': 12, 'MAM': 18, 'JJAS': 20, 'OND': 18}
nstp = [183]
for _yr in PUMPING_YEARS:
    for _s in SEASONS:
        nstp.append(_nstp_season[_s])

tsmult = [1.0] * len(perlen)
nper   = len(perlen)
assert nper == 1 + 4 * N_PUMP_YEARS, f"nper={nper} unexpected"

SP_META = {0: {"year": None, "season": None}}
_sp_idx = 1
for _yr in PUMPING_YEARS:
    for _s in SEASONS:
        SP_META[_sp_idx] = {"year": _yr, "season": _s}
        _sp_idx += 1

def sp_of(year, season):
    """Return stress period index for given year and season."""
    return 1 + PUMPING_YEARS.index(year) * 4 + SEASONS.index(season)

print(f"  [config] Schedule: {nper} SPs  |  "
      f"{sum(perlen):.0f} d  ({sum(perlen)/365.25:.1f} yr)")

# =============================================================================
# FIELD DATA  --  Post-monsoon OND 2024-25
# Source : In-situ measurements, Thermo Scientific Orion multiparameter meter
# sal    : EC-derived salinity [ppt], instrument-measured
# ec     : electrical conductivity [uS/cm], direct field measurement
# dist   : GPS-surveyed distance from shoreline [m]
# type   : DW = open dug well  |  TW = tubewell
#
# NOTE: TW8 (~150 m, sal=0.2269 ppt) excluded -- full dataset not available.
#
# Mismatch note codes (comparison table):
#   T = Tidal/beach-face processes not in steady 2D model
#   H = Lateral heterogeneity / preferential flow paths
#   D = Depth mismatch (shallow open well vs model layer average)
#   S = 2D section cannot capture 3D lateral mixing signal
#   P = Paleo-intrusion residual -- geochemical fingerprint of historical
#       seawater contact not reproduced in the 30-year model window
# =============================================================================
FIELD_WELLS = {
    'TW6':  {'dist':   75, 'sal': 0.140, 'ec':  337.2, 'type': 'TW'},
    'GW7':  {'dist':   80, 'sal': 0.134, 'ec':  327.6, 'type': 'DW'},
    'GW9':  {'dist':  130, 'sal': 0.060, 'ec':  201.0, 'type': 'DW'},
    'GW5':  {'dist':  180, 'sal': 0.150, 'ec':  166.5, 'type': 'DW'},
    'GW10': {'dist':  210, 'sal': 0.081, 'ec':  228.5, 'type': 'DW'},
    'TW12': {'dist':  310, 'sal': 0.088, 'ec':  240.6, 'type': 'TW'},
    'GW11': {'dist':  405, 'sal': 0.050, 'ec':   80.0, 'type': 'DW'},
    'GW13': {'dist':  520, 'sal': 0.050, 'ec':   70.6, 'type': 'DW'},
    'GW2':  {'dist':  745, 'sal': 0.030, 'ec':  178.0, 'type': 'DW'},
    'TW3':  {'dist':  790, 'sal': 0.030, 'ec':  195.0, 'type': 'TW'},
    'TW4':  {'dist':  908, 'sal': 0.043, 'ec':  225.1, 'type': 'TW'},
    'GW1':  {'dist': 1100, 'sal': 0.050, 'ec':  103.0, 'type': 'DW'},
}

WELL_MISMATCH_NOTES = {
    'TW6':  'T,H',
    'GW7':  'T,H',
    'GW9':  'T,D',
    'GW5':  'T,D',
    'GW10': 'D,S',
    'TW12': 'D,S',
    'GW11': 'S,P',
    'GW13': 'S,P',
    'GW2':  'S,P',
    'TW3':  'S,P',
    'TW4':  'S,P',
    'GW1':  'S,P',
}

# =============================================================================
# PLOTTING CONSTANTS  (shared by C3, C4, C5a, C5b)
# =============================================================================
SEASON_LABELS = {
    'JF':   'Jan-Feb (JF) -- Dry winter',
    'MAM':  'Mar-May (MAM) -- Pre-monsoon',
    'JJAS': 'Jun-Sep (JJAS) -- SW Monsoon',
    'OND':  'Oct-Dec (OND) -- Post-monsoon',
}
SEASON_COLORS  = {'JF': '#8c564b', 'MAM': '#d62728', 'JJAS': '#1f77b4', 'OND': '#ff7f0e'}
SEASON_MARKERS = {'JF': 'D', 'MAM': 'v', 'JJAS': 'o', 'OND': 's'}
WELL_COLORS    = ["#e41a1c", "#ff7f00", "#4daf4a", "#984ea3", "#377eb8"]

# =============================================================================
# SELF-CHECK SUMMARY
# =============================================================================
print(f"  [config] Recharge coeff : {RECHARGE_COEFF*100:.0f}%  "
      f"(GEC 1997 west coast norm 8-12%; Kumar et al. 2020 calibrated 10%)")
print(f"  [config] K(x,z)         : surface={K_TOP:.0f} m/d -> deep={K_BOTTOM:.0f} m/d  "
      f"(Priyanka & Mohan Kumar 2019; Kumar et al. 2020 zone value)")
print(f"  [config] ALPHA_L        : {ALPHA_L:.1f} m  |  ALPHA_T={ALPHA_T:.1f} m  "
      f"(Xu & Eckstein 1995 applied to Pavanje basin)")
print(f"  [config] GHB L_regional : {L_REGIONAL:.0f} m  |  "
      f"C_base={C_GHB_BASE:.5f} m2/d/layer")
print(f"  [config] Section width  : {SECTION_WIDTH:.0f} m")
print(f"  [config] Q_gw range     : {Q_gw(1996):.0f}-{Q_gw(2025):.0f} m3/d")
print(f"  [config] Ready.\n")
# =============================================================================
# DERIVED GRID ARRAYS  —  required by C4, C5
# The model domain top is flat at 0.0 m MSL (uniform laterite surface).
# top_surface is a per-column array used for cross-section plots.
# dis_top / dis_botm / tops_3d / bots_3d are the 3D arrays FloPy DIS expects.
# IDOMAIN = 1 everywhere (no inactive cells in this model).
# =============================================================================
top_surface = np.zeros(ncol, dtype=float)          # flat at 0.0 m MSL, shape (ncol,)

dis_top  = tops[0]                                  # scalar: 0.0 m MSL (FloPy DIS top=)
dis_botm = bots                                     # shape (nlay,)  — FloPy DIS botm=

tops_3d  = np.full((nlay, nrow, ncol), 0.0)
for k in range(nlay):
    tops_3d[k, :, :] = tops[k]

bots_3d  = np.full((nlay, nrow, ncol), 0.0)
for k in range(nlay):
    bots_3d[k, :, :] = bots[k]

IDOMAIN  = np.ones((nlay, nrow, ncol), dtype=int)  # all cells active