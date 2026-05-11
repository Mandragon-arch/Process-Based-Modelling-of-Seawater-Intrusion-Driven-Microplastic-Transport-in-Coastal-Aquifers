# =============================================================================
# SALTWATER INTRUSION MODEL — PART A + HEAD SWEEP
# x = 0 at COAST (sea), x = 1500 m at INLAND
# MODFLOW 6 + GWF + GWT + BUY (density coupling)
# dispersivity coefficients for model aL = 7.5, aT =0.75
# =============================================================================

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — no GUI, saves only
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import flopy
import os

# -----------------------------------------------------------------------------
# 0. EXECUTABLE
# -----------------------------------------------------------------------------
mf6_exe = r"C:\WRDAPP\mf6.6.2_win64\bin\mf6.exe"

# -----------------------------------------------------------------------------
# 1. WORKSPACE & NAMES
# -----------------------------------------------------------------------------
ws       = "./swi_partA"
gwf_name = "gwf_swi"
gwt_name = "gwt_swi"
os.makedirs(ws, exist_ok=True)

# -----------------------------------------------------------------------------
# 2. GRID
# -----------------------------------------------------------------------------
Lx   = 1500.0
Lz   = 40.0
nlay = 40
nrow = 1
ncol = 120

# Fine grid at coast (left/x=0), coarse inland (right/x=1500)
dx_raw  = np.geomspace(5.0, 20.0, ncol)
dx_raw *= Lx / dx_raw.sum()
delr    = dx_raw
delc    = np.array([1.0])

# Uniform layer thicknesses
dz   = Lz / nlay
tops = np.zeros(nlay)
bots = np.zeros(nlay)
tops[0] =  0.0
bots[0] = -dz
for k in range(1, nlay):
    tops[k] = bots[k - 1]
    bots[k] = tops[k] - dz

# Column centre x-coordinates (0 = coast, 1500 = inland)
x_centers = np.cumsum(delr) - delr / 2.0

# -----------------------------------------------------------------------------
# 3. HYDRAULIC CONDUCTIVITY — K(x, z) spatially variable
#    K(x,z) = Kz(z) * Kx(x)
#    Kz: sigmoid in depth (lateritic near surface, sandy at depth)
#    Kx: mild tanh horizontal variation
# -----------------------------------------------------------------------------

# --- Vertical variation Kz(z) ---
K_top    =  7.0    # m/day  — near-surface lateritic zone
K_bottom = 30.0    # m/day  — deep sandy aquifer
z0       = -10.0   # m      — transition depth below MSL
b        =  0.5    # sigmoid steepness (0.4–0.8)

z_mid_layers = (tops + bots) / 2.0   # shape: (nlay,)  layer centre elevations

# Kz shape: (nlay,)
Kz = K_top + (K_bottom - K_top) / (1.0 + np.exp(-b * (z_mid_layers - z0)))

# --- Horizontal variation Kx(x) ---
# Range ≈ 0.8 to 1.2, smooth tanh
# x_centers shape: (ncol,)
Kx = 1.0 + 0.2 * np.tanh((x_centers - Lx / 2.0) / (Lx / 4.0))

# --- Build full 3D K array [nlay, nrow, ncol] ---
# Kz[:, np.newaxis, np.newaxis] broadcasts over row and col
# Kx[np.newaxis, np.newaxis, :] broadcasts over lay and row
K33_3d = Kz[:, np.newaxis, np.newaxis] * Kx[np.newaxis, np.newaxis, :]
# shape after broadcast: (nlay, 1, ncol) — correct for nrow=1

# --- Sanity check ---
print(f"\n  K(x,z) field summary:")
print(f"  Min K = {K33_3d.min():.3f} m/day  (expected ≈ 5–7)")
print(f"  Max K = {K33_3d.max():.3f} m/day  (expected ≈ 30–36)")
print(f"  Shape = {K33_3d.shape}  ✓" if K33_3d.shape == (nlay, nrow, ncol) else
      f"  Shape = {K33_3d.shape}  ✗ MISMATCH — expected ({nlay}, {nrow}, {ncol})")

# Confirm no invalid values
assert K33_3d.min() > 0, "ERROR: Zero or negative K values found!"
# -----------------------------------------------------------------------------
# 4. TIME
# -----------------------------------------------------------------------------
nper   = 1
perlen = [4000.0]
nstp   = [400]
tsmult = [1.0]

# -----------------------------------------------------------------------------
# 5. PARAMETERS
# -----------------------------------------------------------------------------
porosity  = 0.30
alphaL    = 7.5
alphaT    = 0.75
rho_ref   = 1000.0
drhodc    = 0.7
C_sea     = 35.0
C_fresh   = 0.0

# -----------------------------------------------------------------------------
# 6. BOUNDARY DATA — base case (inland_head = 1.0 m)
#    LEFT  (j=0,      x=0 m)    → SEA    : head=0.0,        C=35 ppt
#    RIGHT (j=ncol-1, x=1500 m) → INLAND : head=inland_head, C=0 ppt
# -----------------------------------------------------------------------------
inland_head = 1.0

def make_chd(h):
    lst = []
    for k in range(nlay):
        lst.append([(k, 0, 0),        0.0, C_sea  ])   # coast
        lst.append([(k, 0, ncol - 1), h,   C_fresh])   # inland
    return lst

def make_cnc():
    lst = []
    for k in range(nlay):
        lst.append([(k, 0, 0),        C_sea  ])
        lst.append([(k, 0, ncol - 1), C_fresh])
    return lst

chd_list_gwf = make_chd(inland_head)
cnc_list     = make_cnc()

# =============================================================================
# 7. SIMULATION
# =============================================================================
sim = flopy.mf6.MFSimulation(
    sim_name=gwf_name,
    version="mf6",
    exe_name=mf6_exe,
    sim_ws=ws,
)

flopy.mf6.ModflowTdis(
    sim,
    time_units="days",
    nper=nper,
    perioddata=list(zip(perlen, nstp, tsmult)),
)

# =============================================================================
# 8. GWF — IMS first, then model, then register
# =============================================================================
ims_gwf = flopy.mf6.ModflowIms(
    sim,
    pname="ims_gwf",
    filename="gwf.ims",
    print_option="SUMMARY",
    complexity="MODERATE",
    outer_maximum=200,
    inner_maximum=100,
    outer_dvclose=1e-5,
    inner_dvclose=1e-6,
    rcloserecord=1e-4,
    linear_acceleration="BICGSTAB",
    relaxation_factor=0.97,
)

gwf = flopy.mf6.ModflowGwf(
    sim,
    modelname=gwf_name,
    save_flows=True,
    newtonoptions="NEWTON UNDER_RELAXATION",
)
sim.register_ims_package(ims_gwf, [gwf_name])

# DIS
flopy.mf6.ModflowGwfdis(
    gwf,
    nlay=nlay, nrow=nrow, ncol=ncol,
    delr=delr, delc=delc,
    top=tops[0], botm=bots,
    idomain=1,
    length_units="METERS",
)

# NPF
flopy.mf6.ModflowGwfnpf(
    gwf,
    save_specific_discharge=True,
    icelltype=1,
    k=K33_3d,
    k33=K33_3d,
)

# IC
flopy.mf6.ModflowGwfic(gwf, strt=0.0)

# STO
flopy.mf6.ModflowGwfsto(
    gwf,
    sy=0.2, ss=1e-5,
    iconvert=1,
    steady_state={0: True},
)

# CHD with auxiliary concentration
flopy.mf6.ModflowGwfchd(
    gwf,
    auxiliary=["CONCENTRATION"],
    stress_period_data={0: chd_list_gwf},
    pname="CHD",
)

# BUY
flopy.mf6.ModflowGwfbuy(
    gwf,
    nrhospecies=1,
    denseref=rho_ref,
    packagedata=[(0, drhodc, C_fresh, gwt_name, "CONCENTRATION")],
)

# OC
flopy.mf6.ModflowGwfoc(
    gwf,
    head_filerecord  =f"{gwf_name}.hds",
    budget_filerecord=f"{gwf_name}.cbb",
    saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
)

# =============================================================================
# 9. GWT — model first, then IMS, then register
# =============================================================================
gwt = flopy.mf6.ModflowGwt(
    sim,
    modelname=gwt_name,
    save_flows=True,
)

ims_gwt = flopy.mf6.ModflowIms(
    sim,
    pname="ims_gwt",
    filename="gwt.ims",
    print_option="SUMMARY",
    complexity="SIMPLE",
    outer_maximum=200,
    inner_maximum=100,
    outer_dvclose=1e-6,
    inner_dvclose=1e-7,
    rcloserecord=1e-5,
    linear_acceleration="BICGSTAB",
    relaxation_factor=0.97,
)
sim.register_ims_package(ims_gwt, [gwt_name])

# DIS
flopy.mf6.ModflowGwtdis(
    gwt,
    nlay=nlay, nrow=nrow, ncol=ncol,
    delr=delr, delc=delc,
    top=tops[0], botm=bots,
    idomain=1,
    length_units="METERS",
)

# IC
flopy.mf6.ModflowGwtic(gwt, strt=C_fresh)

# ADV
flopy.mf6.ModflowGwtadv(gwt, scheme="TVD")

# DSP
flopy.mf6.ModflowGwtdsp(
    gwt,
    alh=alphaL, ath1=alphaT, atv=alphaT,
    diffc=0.0,
)

# MST
flopy.mf6.ModflowGwtmst(gwt, porosity=porosity)

# CNC
flopy.mf6.ModflowGwtcnc(
    gwt,
    stress_period_data={0: cnc_list},
    pname="CNC",
)

# SSM
flopy.mf6.ModflowGwtssm(
    gwt,
    sources=[("CHD", "AUX", "CONCENTRATION")],
    pname="SSM",
)

# OC
flopy.mf6.ModflowGwtoc(
    gwt,
    concentration_filerecord =f"{gwt_name}.ucn",
    budget_filerecord         =f"{gwt_name}.cbb",
    saverecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
    printrecord=[("CONCENTRATION", "LAST")],
)

# =============================================================================
# 10. GWF-GWT EXCHANGE
# =============================================================================
flopy.mf6.ModflowGwfgwt(
    sim,
    exgtype="GWF6-GWT6",
    exgmnamea=gwf_name,
    exgmnameb=gwt_name,
    filename=f"{gwf_name}_{gwt_name}.exg",
)

# =============================================================================
# 11. WRITE & RUN — BASE CASE (inland head = 1.0 m)
# =============================================================================
sim.write_simulation(silent=False)
success, buff = sim.run_simulation(silent=False, report=True)

if not success:
    print("\n*** MODFLOW 6 DID NOT CONVERGE ***")
    for line in buff:
        print(line)
    raise RuntimeError("Simulation failed — see output above.")
else:
    print("\n✓ Base case simulation completed successfully.")

# =============================================================================
# 12. POST-PROCESSING UTILITIES
# =============================================================================

def load_concentration(ws, gwt_name, kstpkper=None):
    ucn_file = os.path.join(ws, f"{gwt_name}.ucn")
    conc_obj = flopy.utils.HeadFile(ucn_file, text="CONCENTRATION")
    ksp = kstpkper if kstpkper else (nstp[0] - 1, 0)
    return conc_obj.get_data(kstpkper=ksp)

def load_head(ws, gwf_name, kstpkper=None):
    hds_file = os.path.join(ws, f"{gwf_name}.hds")
    hds_obj  = flopy.utils.HeadFile(hds_file)
    ksp = kstpkper if kstpkper else (nstp[0] - 1, 0)
    return hds_obj.get_data(kstpkper=ksp)

def get_toe_position(conc, x_centers, threshold=17.5):
    """
    Toe = furthest INLAND column in bottom layer where C > threshold.
    Coast at x=0, inland at x=1500 m.
    Returns x in metres from coast.
    """
    bottom_layer   = conc[-1, 0, :]
    inland_indices = np.where(bottom_layer > threshold)[0]
    if len(inland_indices) == 0:
        return 0.0
    return x_centers[inland_indices.max()]

def save_concentration_plot(conc, x_centers, toe_x, h_val, filepath):
    C2d     = conc[:, 0, :]
    x_edges = np.concatenate([[0], np.cumsum(delr)])
    z_edges = np.concatenate([[tops[0]], bots])
    z_mid   = (tops + bots) / 2.0

    fig, ax = plt.subplots(figsize=(14, 5))
    cmap = plt.cm.RdYlBu_r
    norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=17.5, vmax=35)
    pcm  = ax.pcolormesh(x_edges, z_edges, C2d, cmap=cmap, norm=norm, shading="flat")
    ax.contour(x_centers, z_mid, C2d,
               levels=[17.5], colors="black", linewidths=1.5, linestyles="--")
    ax.axvline(toe_x, color="red", lw=1.5, ls=":",
               label=f"Toe @ {toe_x:.0f} m from coast")
    fig.colorbar(pcm, ax=ax, label="Concentration (ppt)")
    ax.set_xlabel("Distance from coast [m]  (0 = sea, 1500 = inland)", fontsize=12)
    ax.set_ylabel("Elevation [m]", fontsize=12)
    ax.set_title(f"Saltwater Wedge — Steady State  |  Inland head = {h_val} m",
                 fontsize=13)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(filepath, dpi=150)
    plt.close()
    print(f"  ✓ Plot saved → {filepath}")

# =============================================================================
# 13. BASE CASE RESULTS (inland head = 1.0 m)
# =============================================================================
conc_base = load_concentration(ws, gwt_name)
toe_base  = get_toe_position(conc_base, x_centers)

print(f"\n→ Base case toe : {toe_base:.1f} m from coast")
print(f"  (= {Lx - toe_base:.1f} m from inland boundary)")

save_concentration_plot(
    conc_base, x_centers, toe_base,
    h_val=inland_head,
    filepath=os.path.join(ws, "swi_base_case.png"),
)

# =============================================================================
# 14. HEAD SWEEP LOOP — 0.5, 1.0, 2.0, 3.0 m
# =============================================================================
inland_heads_list = [0.5, 1.0, 2.0, 3.0]
toe_results       = {}
conc_results      = {}

for h in inland_heads_list:
    print(f"\n{'='*55}")
    print(f"  Running: inland head = {h} m")
    print(f"{'='*55}")

    # Update CHD only
    gwf.chd.stress_period_data.set_data({0: make_chd(h)})

    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True, report=True)

    if not success:
        print(f"  ✗ FAILED for h = {h} m")
        for line in buff:
            print(line)
        continue

    conc_h = load_concentration(ws, gwt_name)
    toe_h  = get_toe_position(conc_h, x_centers)

    conc_results[h] = conc_h
    toe_results[h]  = toe_h

    print(f"  ✓ Toe @ {toe_h:.1f} m from coast")

    # Save individual cross-section plot
    save_concentration_plot(
        conc_h, x_centers, toe_h,
        h_val=h,
        filepath=os.path.join(ws, f"swi_head_{str(h).replace('.','p')}m.png"),
    )

# =============================================================================
# 15. PRINT TOE SUMMARY TABLE
# =============================================================================
print("\n" + "="*50)
print(f"  {'Inland Head (m)':<22} {'Toe Position (m)'}")
print("="*50)
for h, toe in toe_results.items():
    print(f"  {h:<22.1f} {toe:.1f}")
print("="*50)

# =============================================================================
# 16. PLOT — All 4 cross-sections stacked
# =============================================================================
fig, axes = plt.subplots(len(inland_heads_list), 1,
                          figsize=(14, 4 * len(inland_heads_list)),
                          sharex=True, sharey=True)

cmap    = plt.cm.RdYlBu_r
norm    = mcolors.TwoSlopeNorm(vmin=0, vcenter=17.5, vmax=35)
x_edges = np.concatenate([[0], np.cumsum(delr)])
z_edges = np.concatenate([[tops[0]], bots])
z_mid   = (tops + bots) / 2.0

for ax, h in zip(axes, inland_heads_list):
    if h not in conc_results:
        continue
    C2d = conc_results[h][:, 0, :]
    pcm = ax.pcolormesh(x_edges, z_edges, C2d,
                        cmap=cmap, norm=norm, shading="flat")
    ax.contour(x_centers, z_mid, C2d,
               levels=[17.5], colors="black", linewidths=1.2, linestyles="--")
    ax.axvline(toe_results[h], color="red", lw=1.5, ls=":",
               label=f"Toe @ {toe_results[h]:.0f} m")
    ax.set_ylabel("Elevation [m]", fontsize=10)
    ax.set_title(f"Inland head = {h} m", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    fig.colorbar(pcm, ax=ax, label="Conc (ppt)")

axes[-1].set_xlabel("Distance from coast [m]  (0 = sea, 1500 = inland)", fontsize=12)
plt.suptitle("Saltwater Wedge — Head Sensitivity Study", fontsize=13, y=1.005)
plt.tight_layout()
sweep_plot1 = os.path.join(ws, "swi_head_sweep_sections.png")
plt.savefig(sweep_plot1, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✓ Stacked cross-section plot saved → {sweep_plot1}")

# =============================================================================
# 17. PLOT — Toe position vs inland head
# =============================================================================
h_vals   = list(toe_results.keys())
toe_vals = list(toe_results.values())

fig2, ax2 = plt.subplots(figsize=(7, 5))
ax2.plot(h_vals, toe_vals, "o-", color="royalblue",
         lw=2, ms=8, markerfacecolor="red")
for h, toe in zip(h_vals, toe_vals):
    ax2.annotate(f"{toe:.0f} m",
                 xy=(h, toe), xytext=(8, 4),
                 textcoords="offset points", fontsize=9)
ax2.set_xlabel("Inland Head [m]", fontsize=12)
ax2.set_ylabel("Toe Position [m from coast]", fontsize=12)
ax2.set_title("Wedge Toe vs Inland Head", fontsize=13)
ax2.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
sweep_plot2 = os.path.join(ws, "swi_toe_vs_head.png")
plt.savefig(sweep_plot2, dpi=150)
plt.close()
print(f"✓ Toe vs head plot saved → {sweep_plot2}")

# =============================================================================
# ALL DONE
# Output files
#   swi_base_case.png
#   swi_head_0p5m.png
#   swi_head_1p0m.png
#   swi_head_2p0m.png
#   swi_head_3p0m.png
#   swi_head_sweep_sections.png
#   swi_toe_vs_head.png
# =============================================================================
# TASK 1: GHYBEN-HERZBERG VALIDATION  +  TASK 2: DISPERSION SENSITIVITY

# =============================================================================

# =============================================================================
# TASK 1 — GHYBEN–HERZBERG VALIDATION (inland head = 1.0 m case)
# =============================================================================

print("\n" + "="*60)
print("  TASK 1: GHYBEN–HERZBERG VALIDATION")
print("="*60)

# --- GH prediction ---
# z_GH = (rho_f / (rho_s - rho_f)) * h_inland
# With rho_f=1000, rho_s=1025 → ratio = 40
# Here we use density slope: rho_s = rho_ref + drhodc * C_sea = 1000 + 0.7*35 = 1024.5
rho_s       = rho_ref + drhodc * C_sea          # 1024.5 kg/m³
rho_f       = rho_ref                            # 1000.0 kg/m³
GH_ratio    = rho_f / (rho_s - rho_f)           # ~40.2
h_for_GH    = 1.0                               # inland head [m]
z_GH        = GH_ratio * h_for_GH               # predicted depth below MSL [m]

print(f"\n  Density of seawater (rho_s) : {rho_s:.2f} kg/m³")
print(f"  Density of freshwater (rho_f): {rho_f:.2f} kg/m³")
print(f"  Ghyben–Herzberg ratio        : {GH_ratio:.4f}")
print(f"  Inland head                  : {h_for_GH} m")
print(f"  GH predicted depth below MSL : {z_GH:.2f} m")

# --- Model simulated interface depth at inland boundary (x ~ 1500 m) ---
# Use concentration from h=1.0 m case (already loaded as conc_base or conc_results[1.0])
conc_1m = conc_results.get(1.0, conc_base)  # use sweep result if available

# Find the 17.5 ppt interface depth at the RIGHTMOST column (inland boundary)
inland_col   = ncol - 1
conc_profile = conc_1m[:, 0, inland_col]    # vertical profile at inland col [nlay]
z_mids       = (tops + bots) / 2.0          # layer midpoint elevations [nlay]

# Interpolate: find z where concentration crosses 17.5 ppt
z_model = None
for k in range(nlay - 1):
    if conc_profile[k] <= 17.5 <= conc_profile[k + 1] or \
       conc_profile[k] >= 17.5 >= conc_profile[k + 1]:
        # linear interpolation
        c0, c1 = conc_profile[k], conc_profile[k + 1]
        z0, z1 = z_mids[k], z_mids[k + 1]
        z_model = z0 + (17.5 - c0) / (c1 - c0) * (z1 - z0)
        break

if z_model is None:
    # Interface not found at inland col — try toe column instead
    toe_col = np.argmin(np.abs(x_centers - toe_results.get(1.0, toe_base)))
    conc_profile = conc_1m[:, 0, toe_col]
    for k in range(nlay - 1):
        if conc_profile[k] <= 17.5 <= conc_profile[k + 1] or \
           conc_profile[k] >= 17.5 >= conc_profile[k + 1]:
            c0, c1 = conc_profile[k], conc_profile[k + 1]
            z0, z1 = z_mids[k], z_mids[k + 1]
            z_model = z0 + (17.5 - c0) / (c1 - c0) * (z1 - z0)
            break
    print(f"\n  (Interface not at inland col — evaluated at toe column instead)")

if z_model is not None:
    depth_model = abs(z_model)             # convert elevation to depth [m]
    pct_diff    = abs(z_GH - depth_model) / z_GH * 100
    print(f"\n  Model interface elevation    : {z_model:.2f} m  (depth = {depth_model:.2f} m)")
    print(f"  GH predicted depth           : {z_GH:.2f} m")
    print(f"  Percentage difference        : {pct_diff:.1f} %")
else:
    print("\n  WARNING: 17.5 ppt interface not found in vertical profile.")
    depth_model = z_GH   # fallback

# --- Plot: h=1.0 m concentration + GH line ---
C2d_1m  = conc_1m[:, 0, :]
x_edges = np.concatenate([[0], np.cumsum(delr)])
z_edges = np.concatenate([[tops[0]], bots])
z_mid   = (tops + bots) / 2.0
toe_1m  = toe_results.get(1.0, toe_base)

fig, ax = plt.subplots(figsize=(14, 5))
cmap = plt.cm.RdYlBu_r
norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=17.5, vmax=35)
pcm  = ax.pcolormesh(x_edges, z_edges, C2d_1m, cmap=cmap, norm=norm, shading="flat")
ax.contour(x_centers, z_mid, C2d_1m,
           levels=[17.5], colors="black", linewidths=1.5, linestyles="--",
           label="17.5 ppt interface")
ax.axvline(toe_1m, color="red", lw=1.5, ls=":",
           label=f"Model toe @ {toe_1m:.0f} m")

# GH depth line (horizontal, across full domain)
gh_elev = -z_GH   # depth below MSL → negative elevation
ax.axhline(gh_elev, color="green", lw=2.0, ls="-.",
           label=f"GH depth = {z_GH:.1f} m ({gh_elev:.1f} m elev)")

fig.colorbar(pcm, ax=ax, label="Concentration (ppt)")
ax.set_xlabel("Distance from coast [m]  (0 = sea, 1500 = inland)", fontsize=12)
ax.set_ylabel("Elevation [m]", fontsize=12)
ax.set_title("Saltwater Wedge — h=1.0 m  |  Ghyben–Herzberg Validation", fontsize=13)
ax.legend(loc="lower right", fontsize=9)
plt.tight_layout()
gh_plot = os.path.join(ws, "swi_GH_validation.png")
plt.savefig(gh_plot, dpi=150)
plt.close()
print(f"\n✓ GH validation plot saved → {gh_plot}")

# =============================================================================
# TASK 2 — DISPERSION SENSITIVITY ANALYSIS
# Case A: alphaL = 10 m  |  Case B: alphaL = 30 m
# =============================================================================

print("\n" + "="*60)
print("  TASK 2: DISPERSION SENSITIVITY ANALYSIS")
print("="*60)

disp_cases = {
    "alphaL_10": {"alphaL": 10.0, "alphaT": 1.0},
    "alphaL_30": {"alphaL": 30.0, "alphaT": 3.0},
}

disp_results = {}   # stores {case_name: {"conc": ..., "toe": ...}}

for case_name, params in disp_cases.items():
    aL = params["alphaL"]
    aT = params["alphaT"]
    print(f"\n  Running: {case_name}  (αL={aL} m, αT={aT} m)")

    # Reset CHD to inland head = 1.0 m
    gwf.chd.stress_period_data.set_data({0: make_chd(1.0)})

    # Update DSP package dispersivity
    gwt.dsp.alh  = aL
    gwt.dsp.ath1 = aT
    gwt.dsp.atv  = aT

    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True, report=True)

    if not success:
        print(f"  ✗ FAILED for {case_name}")
        for line in buff:
            print(line)
        continue

    conc_d = load_concentration(ws, gwt_name)
    toe_d  = get_toe_position(conc_d, x_centers)
    disp_results[case_name] = {"conc": conc_d, "toe": toe_d, "aL": aL, "aT": aT}
    print(f"  ✓ Toe @ {toe_d:.1f} m from coast  (αL={aL} m)")

# --- Print dispersion toe summary ---
print("\n" + "="*50)
print(f"  {'Case':<18} {'αL (m)':<10} {'Toe (m from coast)'}")
print("="*50)
for cname, res in disp_results.items():
    print(f"  {cname:<18} {res['aL']:<10.0f} {res['toe']:.1f}")
print("="*50)

# --- Comparison plot: side-by-side subplots ---
fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharey=True)

cmap    = plt.cm.RdYlBu_r
norm    = mcolors.TwoSlopeNorm(vmin=0, vcenter=17.5, vmax=35)
x_edges = np.concatenate([[0], np.cumsum(delr)])
z_edges = np.concatenate([[tops[0]], bots])
z_mid   = (tops + bots) / 2.0

case_labels = list(disp_results.keys())

for ax, cname in zip(axes, case_labels):
    res  = disp_results[cname]
    C2d  = res["conc"][:, 0, :]
    toe  = res["toe"]
    aL   = res["aL"]

    pcm = ax.pcolormesh(x_edges, z_edges, C2d, cmap=cmap, norm=norm, shading="flat")
    ax.contour(x_centers, z_mid, C2d,
               levels=[17.5], colors="black", linewidths=1.5, linestyles="--")
    ax.axvline(toe, color="red", lw=1.5, ls=":",
               label=f"Toe @ {toe:.0f} m")
    ax.set_xlabel("Distance from coast [m]", fontsize=11)
    ax.set_title(f"αL = {aL} m  |  Toe = {toe:.0f} m from coast", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    fig.colorbar(pcm, ax=ax, label="Concentration (ppt)")

axes[0].set_ylabel("Elevation [m]", fontsize=11)
plt.suptitle("Dispersion Sensitivity — αL = 10 m vs αL = 30 m  (h_inland = 1.0 m)",
             fontsize=13)
plt.tight_layout()
disp_plot = os.path.join(ws, "swi_dispersion_comparison.png")
plt.savefig(disp_plot, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n✓ Dispersion comparison plot saved → {disp_plot}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*60)
print("  FINAL SUMMARY")
print("="*60)
print(f"  GH predicted depth (h=1m)     : {z_GH:.2f} m below MSL")
if z_model is not None:
    print(f"  Model interface depth (h=1m)  : {depth_model:.2f} m below MSL")
    print(f"  GH vs Model difference        : {pct_diff:.1f} %")
print()
for cname, res in disp_results.items():
    print(f"  {cname:<20} toe = {res['toe']:.1f} m from coast")
print("="*60)
print("\n  Output files in ./swi_partA/:")
print("    swi_GH_validation.png")
print("    swi_dispersion_comparison.png")

# =============================================================================
# END OF TASK 1 + TASK 2
# =============================================================================