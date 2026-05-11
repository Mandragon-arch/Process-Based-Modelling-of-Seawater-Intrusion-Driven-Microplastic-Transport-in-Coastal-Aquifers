# =============================================================================
# C2_extract.py  —  SWI PROJECT  |  NITK / Surathkal Campus Aquifer
# =============================================================================
# Reads MODFLOW 6 binary output from C1_build_run.py and extracts:
#   1. Concentration field (nlay x ncol) for every pumping stress period
#   2. Hydraulic head   field (nlay x ncol) for every pumping stress period
#   3. Wedge toe position (m from coast) for every pumping stress period
#   4. Model salinity at each field well column (layer 1, OND 2025)
#
# Output structure confirmed from C1_build_run.py run log (2026/05/06):
#   121 stress periods — SP 0 warm-up (183 tsteps) + SP 1-120 pumping seasons
#   OC saverecord = LAST  -> one head/conc record per SP in binary file
#   Binary kstpkper: (kstp_0based, kper_0based)
#     warm-up   SP 0  -> (182, 0)
#     1996 JF   SP 1  -> (11,  1)
#     2025 OND  SP 120 -> (17, 120)
#
# kper map is read DIRECTLY from the binary file — robust against any
# nstp or tsmult changes.
#
# Toe definition:
#   Most inland column where mean concentration of bottom 5 layers
#   (indices 35-39, elevations -35 to -40 m MSL) >= 0.5 ppt.
#   Returns 0.0 if no intrusion detected.
#
# Dry-cell masking:
#   Cells where head <= layer bottom set to NaN before averaging.
#   Prevents false-zero salinity in dry cells during MAM/JF seasons.
#   OND 2025 (validation season) is fully saturated — masking has no effect.
#
# Saves: results/C2_extracted.npz
# Consumed by: C3_plots.py, C4_validate.py
# Runtime: ~15-30 s
# =============================================================================

import numpy as np
import flopy
import os
import config

print("=" * 65)
print("  C2_extract.py  —  reading MODFLOW 6 binary output")
print("=" * 65)

# =============================================================================
# PATHS
# =============================================================================
ws       = config.WORKSPACE_BASE
gwf_name = config.GWF_NAME
gwt_name = config.GWT_NAME

hds_path = os.path.join(ws, f"{gwf_name}.hds")
ucn_path = os.path.join(ws, f"{gwt_name}.ucn")

for p, label in [(hds_path, "gwf_swi.hds"), (ucn_path, "gwt_swi.ucn")]:
    if not os.path.isfile(p):
        raise FileNotFoundError(
            f"\n{'='*65}\n"
            f"  HARD STOP — {label} not found:\n"
            f"    {p}\n"
            f"  Run C1_build_run.py first.\n"
            f"{'='*65}"
        )

print(f"  HDS : {hds_path}")
print(f"  UCN : {ucn_path}")

# =============================================================================
# GRID AND SCHEDULE
# =============================================================================
nlay, nrow, ncol = config.nlay, config.nrow, config.ncol
x_centers        = config.x_centers
z_mid_layers     = config.z_mid_layers
tops             = np.array(config.tops)
bots             = np.array(config.bots)
nper             = config.nper              # 121
SP_META          = config.SP_META
PUMPING_YEARS    = config.PUMPING_YEARS
SEASONS          = config.SEASONS

# Pumping SP indices in config: 1 to 120
pumping_sp_indices = [sp for sp in range(1, nper)
                      if SP_META[sp]["year"] is not None]
n_pump = len(pumping_sp_indices)   # 120

print(f"\n  Grid       : {nlay} layers x {ncol} cols x {nrow} row")
print(f"  Total SPs  : {nper}  (1 warm-up + {n_pump} pumping)")
print(f"  Years      : {PUMPING_YEARS[0]}–{PUMPING_YEARS[-1]}")

# =============================================================================
# OPEN BINARY FILES
# Note: flopy.utils.UcnFile (not UCNFile) — case-sensitive in flopy 3.x
# =============================================================================
hds_obj = flopy.utils.HeadFile(hds_path)
ucn_obj = flopy.utils.HeadFile(ucn_path, text='concentration')

hds_all = hds_obj.get_kstpkper()
ucn_all = ucn_obj.get_kstpkper()

print(f"\n  HDS records found : {len(hds_all)}  (expected {nper})")
print(f"  UCN records found : {len(ucn_all)}  (expected {nper})")

# Build kper -> kstpkper maps (take highest kstp if multiple per kper)
# kper = second element of tuple, 0-based stress period index
hds_kper_map = {}
for kstp, kper in hds_all:
    if kper not in hds_kper_map or kstp > hds_kper_map[kper][0]:
        hds_kper_map[kper] = (kstp, kper)

ucn_kper_map = {}
for kstp, kper in ucn_all:
    if kper not in ucn_kper_map or kstp > ucn_kper_map[kper][0]:
        ucn_kper_map[kper] = (kstp, kper)

# Spot-check three key SPs to verify mapping before full extraction
print("\n  Spot-check kstpkper mapping:")
for sp, yr, seas in [(1, 1996, 'JF'), (60, 2010, 'OND'), (120, 2025, 'OND')]:
    u = ucn_kper_map.get(sp, 'MISSING')
    h = hds_kper_map.get(sp, 'MISSING')
    print(f"    SP {sp:3d} ({yr} {seas:4s}) -> UCN {u}  HDS {h}")
# Expected output:
#   SP   1 (1996 JF  ) -> UCN (11, 1)    HDS (11, 1)
#   SP  60 (2010 OND ) -> UCN (17, 60)   HDS (17, 60)
#   SP 120 (2025 OND ) -> UCN (17, 120)  HDS (17, 120)

# =============================================================================
# HELPER: dry-cell mask
# =============================================================================
def apply_dry_mask(c2d, h2d):
    """Set concentration to NaN where head <= layer bottom elevation."""
    out = c2d.astype(np.float64).copy()
    for k in range(nlay):
        out[k, h2d[k, :] <= bots[k]] = np.nan
    return out

# =============================================================================
# HELPER: wedge toe
# =============================================================================
def compute_toe(c2d_masked):
    """
    Most inland column where mean of bottom 5 layers (indices 35-39) >= 0.5 ppt.
    Returns x in metres from coast; 0.0 if no intrusion detected.
    """
    base = np.nanmean(c2d_masked[35:, :], axis=0)
    cols = np.where(base >= 0.5)[0]
    return float(x_centers[cols[-1]]) if len(cols) > 0 else 0.0

# =============================================================================
# MAIN EXTRACTION LOOP — all 120 pumping stress periods
# =============================================================================
print(f"\n  Extracting {n_pump} pumping stress periods ...")

conc_all   = np.full((n_pump, nlay, ncol), np.nan, dtype=np.float32)
head_all   = np.full((n_pump, nlay, ncol), np.nan, dtype=np.float32)
sp_years   = np.zeros(n_pump, dtype=np.int32)
sp_seasons = np.empty(n_pump, dtype='U4')
toe_arr    = np.zeros(n_pump, dtype=np.float32)

n_missing = 0

for idx, sp in enumerate(pumping_sp_indices):
    yr   = SP_META[sp]["year"]
    seas = SP_META[sp]["season"]
    sp_years[idx]   = yr
    sp_seasons[idx] = seas

    kper = sp   # config SP index == binary kper (both 0-based)

    if kper not in ucn_kper_map or kper not in hds_kper_map:
        n_missing += 1
        print(f"  WARNING: SP {sp} ({yr} {seas}) missing from binary — skipped.")
        continue

    h3d = hds_obj.get_data(kstpkper=hds_kper_map[kper])   # (nlay, 1, ncol)
    c3d = ucn_obj.get_data(kstpkper=ucn_kper_map[kper])   # (nlay, 1, ncol)

    h2d = h3d[:, 0, :]
    c2d = c3d[:, 0, :]
    c2d_masked = apply_dry_mask(c2d, h2d)

    conc_all[idx] = c2d_masked.astype(np.float32)
    head_all[idx] = h2d.astype(np.float32)
    toe_arr[idx]  = compute_toe(c2d_masked)

extracted = n_pump - n_missing
print(f"  Extracted : {extracted} / {n_pump}  |  Missing : {n_missing}")

# =============================================================================
# TOE TIME SERIES BY SEASON
# =============================================================================
toe_years = np.array(sorted({int(y) for y in sp_years if y > 0}), dtype=np.int32)
n_yr      = len(toe_years)

toe_JF   = np.zeros(n_yr, dtype=np.float32)
toe_MAM  = np.zeros(n_yr, dtype=np.float32)
toe_JJAS = np.zeros(n_yr, dtype=np.float32)
toe_OND  = np.zeros(n_yr, dtype=np.float32)
_toe_map = {'JF': toe_JF, 'MAM': toe_MAM, 'JJAS': toe_JJAS, 'OND': toe_OND}

for idx in range(n_pump):
    yr   = int(sp_years[idx])
    seas = str(sp_seasons[idx])
    if yr == 0:
        continue
    yi = int(np.searchsorted(toe_years, yr))
    _toe_map[seas][yi] = float(toe_arr[idx])

print(f"\n  Wedge toe range (m from coast):")
print(f"    JF   : {toe_JF.min():.0f} – {toe_JF.max():.0f}")
print(f"    MAM  : {toe_MAM.min():.0f} – {toe_MAM.max():.0f}")
print(f"    JJAS : {toe_JJAS.min():.0f} – {toe_JJAS.max():.0f}")
print(f"    OND  : {toe_OND.min():.0f} – {toe_OND.max():.0f}")

# =============================================================================
# FIELD WELL EXTRACTION — OND 2025 (SP index 120)
# Extraction: min salinity of top 5 layers (indices 0-4, 0 to -5 m MSL).
# This represents the shallowest freshwater value the model can produce,
# consistent with shallow dug wells sampling near the water table.
# Thesis point 19: well comparison uses min of top 5 model layers.
# =============================================================================
print("\n  Field well extraction (OND 2025, min of top 5 layers) ...")

sp_ond2025  = config.sp_of(2025, 'OND')          # = 120
idx_ond2025 = pumping_sp_indices.index(sp_ond2025)
c_ond       = conc_all[idx_ond2025].astype(np.float64)

_fw_sorted = sorted(config.FIELD_WELLS.items(), key=lambda x: x[1]['dist'])

fw_names_list = []
fw_model_list = []

for wname, wdata in _fw_sorted:
    cidx  = int(np.argmin(np.abs(x_centers - wdata['dist'])))
    vals  = c_ond[:5, cidx]          # top 5 layers
    valid = vals[~np.isnan(vals)]
    sal   = float(np.min(valid)) if len(valid) > 0 else 0.0

    fw_names_list.append(wname)
    fw_model_list.append(sal)

print(f"\n  {'Well':6s}  {'Dist (m)':8s}  {'Field (ppt)':11s}  {'Model min5 (ppt)':16s}")
print(f"  {'-'*6}  {'-'*8}  {'-'*11}  {'-'*16}")
for (wname, wdata), msal in zip(_fw_sorted, fw_model_list):
    print(f"  {wname:6s}  {wdata['dist']:8d}  {wdata['sal']:11.4f}  {msal:16.4f}")

# =============================================================================
# BUILD SAVE ARRAYS
# =============================================================================
fw_names_arr = np.array([w[0] for w in _fw_sorted])
fw_dist_arr  = np.array([w[1]['dist'] for w in _fw_sorted], dtype=np.float32)
fw_sal_arr   = np.array([w[1]['sal']  for w in _fw_sorted], dtype=np.float32)
fw_ec_arr    = np.array([w[1]['ec']   for w in _fw_sorted], dtype=np.float32)
fw_type_arr  = np.array([w[1]['type'] for w in _fw_sorted])
fw_model_arr = np.array(fw_model_list, dtype=np.float32)
fw_notes_arr = np.array([config.WELL_MISMATCH_NOTES.get(w[0], '')
                         for w in _fw_sorted])

# Ion chemistry — field data (OND 2024-25, Ion Chromatography)
# Source: IC Results, field study Surathkal campus (OND 2024-25)
# Cl and HCO3 in mg/L; Cl/HCO3 > 2 = severe seawater intrusion signature
IC_DATA = {
    'TW6':  {'Cl': 2170.0, 'HCO3':  12.20},
    'GW7':  {'Cl': 2074.0, 'HCO3': 140.30},
    'GW9':  {'Cl': 2050.0, 'HCO3':  84.18},
    'GW5':  {'Cl': 2002.0, 'HCO3':  36.60},
    'GW10': {'Cl': 2000.0, 'HCO3': 103.70},
    'TW12': {'Cl': 2025.0, 'HCO3': 122.00},
    'GW11': {'Cl': 2037.0, 'HCO3':  34.16},
    'GW13': {'Cl': 2053.0, 'HCO3':  18.30},
    'GW2':  {'Cl': 1954.0, 'HCO3':  48.80},
    'TW3':  {'Cl': 1957.0, 'HCO3':  67.10},
    'TW4':  {'Cl': 1953.0, 'HCO3':  80.52},
    'GW1':  {'Cl': 1931.0, 'HCO3':  18.30},
}

fw_Cl    = np.array([IC_DATA.get(w[0], {}).get('Cl',   np.nan) for w in _fw_sorted],
                    dtype=np.float32)
fw_HCO3  = np.array([IC_DATA.get(w[0], {}).get('HCO3', np.nan) for w in _fw_sorted],
                    dtype=np.float32)
fw_ratio = np.where((fw_HCO3 > 0) & ~np.isnan(fw_HCO3),
                    fw_Cl / fw_HCO3, np.nan).astype(np.float32)

# =============================================================================
# SAVE
# =============================================================================
os.makedirs(config.RESULTS_DIR, exist_ok=True)
out_path = os.path.join(config.RESULTS_DIR, "C2_extracted.npz")

np.savez_compressed(
    out_path,
    # Grid
    x_centers    = x_centers.astype(np.float32),
    z_mid_layers = z_mid_layers.astype(np.float32),
    tops         = tops.astype(np.float32),
    bots         = bots.astype(np.float32),
    # 3D fields: shape (120, 40, 120)
    conc_all     = conc_all,
    head_all     = head_all,
    sp_years     = sp_years,
    sp_seasons   = sp_seasons,
    # Toe time series: shape (30,) per season
    toe_years    = toe_years,
    toe_JF       = toe_JF,
    toe_MAM      = toe_MAM,
    toe_JJAS     = toe_JJAS,
    toe_OND      = toe_OND,
    # Field wells: shape (12,)
    fw_names     = fw_names_arr,
    fw_dist      = fw_dist_arr,
    fw_sal       = fw_sal_arr,
    fw_ec        = fw_ec_arr,
    fw_type      = fw_type_arr,
    fw_model     = fw_model_arr,
    fw_notes     = fw_notes_arr,
    fw_Cl        = fw_Cl,
    fw_HCO3      = fw_HCO3,
    fw_ratio     = fw_ratio,
    top_surface  = np.zeros(ncol, dtype=np.float32),          # flat 0 m MSL, shape (ncol,)
    fw_gl        = np.zeros(len(_fw_sorted), dtype=np.float32), # ground level = 0 m MSL
)

mb = os.path.getsize(out_path) / 1e6
print(f"\n  Saved -> {out_path}  ({mb:.1f} MB)")
print(f"  conc_all shape : {conc_all.shape}  (n_sp, nlay, ncol)")
print(f"  toe_years      : {toe_years[0]}–{toe_years[-1]}")
print(f"  n_wells        : {len(fw_names_arr)}")
print("\n" + "=" * 65)
print("  C2 COMPLETE — next step: run C3_plots.py")
print("=" * 65)