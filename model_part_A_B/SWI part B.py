# =============================================================================
# SALTWATER INTRUSION MODEL — PART B: RECHARGE-DRIVEN SYSTEM
# Modification of Part A:
#   - REMOVE inland CHD boundary (right side, j = ncol-1)
#   - KEEP sea boundary (left side, j = 0) unchanged
#   - ADD uniform RCH recharge R = 0.0005 m/day to top layer
# All other settings (grid, K, dispersivity, BUY, GWT) remain IDENTICAL to Part A
# =============================================================================

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import flopy
import os

# ── Reuse identical settings from Part A ──────────────────────────────────────
mf6_exe  = r"C:\WRDAPP\mf6.6.2_win64\bin\mf6.exe"
ws       = "./swi_partB"           # NEW workspace so Part A is untouched
gwf_name = "gwf_swi"
gwt_name = "gwt_swi"
os.makedirs(ws, exist_ok=True)

# Grid (identical to Part A)
Lx, Lz    = 1500.0, 40.0
nlay, nrow, ncol = 40, 1, 120
dx_raw    = np.geomspace(5.0, 20.0, ncol)
dx_raw   *= Lx / dx_raw.sum()
delr      = dx_raw
delc      = np.array([1.0])
dz        = Lz / nlay
tops      = np.zeros(nlay);  bots = np.zeros(nlay)
tops[0]   = 0.0;             bots[0] = -dz
for k in range(1, nlay):
    tops[k] = bots[k-1];    bots[k] = tops[k] - dz
x_centers = np.cumsum(delr) - delr / 2.0

# K(x,z) (identical to Part A)
K_top, K_bottom, z0, b = 7.0, 30.0, -10.0, 0.5
z_mid_layers = (tops + bots) / 2.0
Kz           = K_top + (K_bottom - K_top) / (1.0 + np.exp(-b * (z_mid_layers - z0)))
Kx           = 1.0 + 0.2 * np.tanh((x_centers - Lx/2.0) / (Lx/4.0))
K33_3d       = Kz[:, np.newaxis, np.newaxis] * Kx[np.newaxis, np.newaxis, :]

# Parameters (identical to Part A)
porosity = 0.30
alphaL   = 7.5;    alphaT = 0.75
rho_ref  = 1000.0; drhodc = 0.7
C_sea    = 35.0;   C_fresh = 0.0
nper     = 1;      perlen = [4000.0]; nstp = [400]; tsmult = [1.0]

# =============================================================================
# SECTION B-1 ▸ SEA BOUNDARY ONLY (inland CHD REMOVED)
# Only j=0 (coast) cells are included — j=ncol-1 (inland) is dropped entirely
# =============================================================================
def make_chd_sea_only():
    """CHD at coast (x=0) only — no inland boundary."""
    lst = []
    for k in range(nlay):
        lst.append([(k, 0, 0), 0.0, C_sea])   # head=0 m, C=35 ppt
    return lst

def make_cnc_sea_only():
    """CNC at coast (x=0) only — no inland concentration constraint."""
    lst = []
    for k in range(nlay):
        lst.append([(k, 0, 0), C_sea])
    return lst

chd_list_b = make_chd_sea_only()
cnc_list_b = make_cnc_sea_only()

# =============================================================================
# SECTION B-2 ▸ RECHARGE ARRAY
# R = 0.0005 m/day, applied to top layer (layer 0), all columns
# Shape: (nrow, ncol) — FloPy RCH with readasarrays
# =============================================================================
R_val  = 0.0005   # m/day
rch_array = np.full((nrow, ncol), R_val, dtype=float)

# =============================================================================
# BUILD SIMULATION
# =============================================================================
sim = flopy.mf6.MFSimulation(
    sim_name=gwf_name, version="mf6",
    exe_name=mf6_exe, sim_ws=ws,
)

flopy.mf6.ModflowTdis(
    sim, time_units="days", nper=nper,
    perioddata=list(zip(perlen, nstp, tsmult)),
)

# ── GWF ───────────────────────────────────────────────────────────────────────
ims_gwf = flopy.mf6.ModflowIms(
    sim, pname="ims_gwf", filename="gwf.ims",
    print_option="SUMMARY", complexity="MODERATE",
    outer_maximum=200, inner_maximum=100,
    outer_dvclose=1e-5, inner_dvclose=1e-6,
    rcloserecord=1e-4, linear_acceleration="BICGSTAB",
    relaxation_factor=0.97,
)

gwf = flopy.mf6.ModflowGwf(
    sim, modelname=gwf_name,
    save_flows=True, newtonoptions="NEWTON UNDER_RELAXATION",
)
sim.register_ims_package(ims_gwf, [gwf_name])

flopy.mf6.ModflowGwfdis(
    gwf, nlay=nlay, nrow=nrow, ncol=ncol,
    delr=delr, delc=delc,
    top=tops[0], botm=bots,
    idomain=1, length_units="METERS",
)

flopy.mf6.ModflowGwfnpf(
    gwf, save_specific_discharge=True,
    icelltype=1, k=K33_3d, k33=K33_3d,
)

flopy.mf6.ModflowGwfic(gwf, strt=0.0)

flopy.mf6.ModflowGwfsto(
    gwf, sy=0.2, ss=1e-5, iconvert=1,
    steady_state={0: True},
)

# ── CHD: SEA ONLY ─────────────────────────────────────────────────────────────
flopy.mf6.ModflowGwfchd(
    gwf,
    auxiliary=["CONCENTRATION"],
    stress_period_data={0: chd_list_b},
    pname="CHD",
)

# ── RCH: UNIFORM RECHARGE (top layer, all columns) ───────────────────────────
flopy.mf6.ModflowGwfrcha(
    gwf,
    recharge={0: rch_array},    # readasarrays format
    pname="RCH",
)

# ── BUY: density coupling (identical to Part A) ──────────────────────────────
flopy.mf6.ModflowGwfbuy(
    gwf, nrhospecies=1, denseref=rho_ref,
    packagedata=[(0, drhodc, C_fresh, gwt_name, "CONCENTRATION")],
)

# ── OC ────────────────────────────────────────────────────────────────────────
flopy.mf6.ModflowGwfoc(
    gwf,
    head_filerecord  =f"{gwf_name}.hds",
    budget_filerecord=f"{gwf_name}.cbb",
    saverecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
    printrecord=[("HEAD", "LAST"), ("BUDGET", "LAST")],
)

# ── GWT ───────────────────────────────────────────────────────────────────────
gwt = flopy.mf6.ModflowGwt(sim, modelname=gwt_name, save_flows=True)

ims_gwt = flopy.mf6.ModflowIms(
    sim, pname="ims_gwt", filename="gwt.ims",
    print_option="SUMMARY", complexity="SIMPLE",
    outer_maximum=200, inner_maximum=100,
    outer_dvclose=1e-6, inner_dvclose=1e-7,
    rcloserecord=1e-5, linear_acceleration="BICGSTAB",
    relaxation_factor=0.97,
)
sim.register_ims_package(ims_gwt, [gwt_name])

flopy.mf6.ModflowGwtdis(
    gwt, nlay=nlay, nrow=nrow, ncol=ncol,
    delr=delr, delc=delc,
    top=tops[0], botm=bots,
    idomain=1, length_units="METERS",
)

flopy.mf6.ModflowGwtic(gwt, strt=C_fresh)
flopy.mf6.ModflowGwtadv(gwt, scheme="TVD")

flopy.mf6.ModflowGwtdsp(
    gwt, alh=alphaL, ath1=alphaT, atv=alphaT, diffc=0.0,
)

flopy.mf6.ModflowGwtmst(gwt, porosity=porosity)

# ── CNC: sea only ─────────────────────────────────────────────────────────────
flopy.mf6.ModflowGwtcnc(
    gwt,
    stress_period_data={0: cnc_list_b},
    pname="CNC",
)

# ── SSM ───────────────────────────────────────────────────────────────────────
flopy.mf6.ModflowGwtssm(
    gwt,
    sources=[("CHD", "AUX", "CONCENTRATION")],
    pname="SSM",
)

# ── GWT OC ────────────────────────────────────────────────────────────────────
flopy.mf6.ModflowGwtoc(
    gwt,
    concentration_filerecord =f"{gwt_name}.ucn",
    budget_filerecord         =f"{gwt_name}.cbb",
    saverecord=[("CONCENTRATION", "LAST"), ("BUDGET", "LAST")],
    printrecord=[("CONCENTRATION", "LAST")],
)

# ── GWF-GWT Exchange ──────────────────────────────────────────────────────────
flopy.mf6.ModflowGwfgwt(
    sim, exgtype="GWF6-GWT6",
    exgmnamea=gwf_name, exgmnameb=gwt_name,
    filename=f"{gwf_name}_{gwt_name}.exg",
)

# =============================================================================
# RUN SIMULATION
# =============================================================================
sim.write_simulation(silent=False)
success, buff = sim.run_simulation(silent=False, report=True)

if not success:
    print("\n*** MODFLOW 6 DID NOT CONVERGE ***")
    for line in buff:
        print(line)
    raise RuntimeError("Simulation failed — see output above.")
else:
    print("\n✓ Part B simulation completed successfully.")

# =============================================================================
# POST-PROCESSING UTILITIES (same helpers as Part A)
# =============================================================================
def load_concentration(ws, gwt_name, kstpkper=None):
    ucn_file  = os.path.join(ws, f"{gwt_name}.ucn")
    conc_obj  = flopy.utils.HeadFile(ucn_file, text="CONCENTRATION")
    ksp       = kstpkper if kstpkper else (nstp[0]-1, 0)
    return conc_obj.get_data(kstpkper=ksp)

def load_head(ws, gwf_name, kstpkper=None):
    hds_file = os.path.join(ws, f"{gwf_name}.hds")
    hds_obj  = flopy.utils.HeadFile(hds_file)
    ksp      = kstpkper if kstpkper else (nstp[0]-1, 0)
    return hds_obj.get_data(kstpkper=ksp)

def get_toe_position(conc, x_centers, threshold=17.5):
    bottom_layer   = conc[-1, 0, :]
    inland_indices = np.where(bottom_layer > threshold)[0]
    if len(inland_indices) == 0:
        return 0.0
    return x_centers[inland_indices.max()]

# =============================================================================
# OUTPUT 1 — HEAD DISTRIBUTION PLOT
# =============================================================================
head_b    = load_head(ws, gwf_name)
head_top  = head_b[0, 0, :]      # top layer head along all columns

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(x_centers, head_top, color="steelblue", lw=2, label="Head (top layer)")
ax.set_xlabel("Distance from coast [m]  (0 = sea, 1500 = inland)", fontsize=12)
ax.set_ylabel("Hydraulic head [m]", fontsize=12)
ax.set_title("Part B — Head Distribution (Recharge-Driven)", fontsize=13)
ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(ws, "partB_head_distribution.png"), dpi=150)
plt.close()
print("✓ Head distribution plot saved → partB_head_distribution.png")

# =============================================================================
# OUTPUT 2 — SALINITY DISTRIBUTION (same style as Part A)
# =============================================================================
conc_b = load_concentration(ws, gwt_name)
toe_b  = get_toe_position(conc_b, x_centers)
print(f"\n→ Part B toe position : {toe_b:.1f} m from coast")
print(f"  (= {Lx - toe_b:.1f} m from inland boundary)")

C2d     = conc_b[:, 0, :]
x_edges = np.concatenate([[0], np.cumsum(delr)])
z_edges = np.concatenate([[tops[0]], bots])
z_mid   = (tops + bots) / 2.0

fig, ax = plt.subplots(figsize=(14, 5))
cmap = plt.cm.RdYlBu_r
norm = mcolors.TwoSlopeNorm(vmin=0, vcenter=17.5, vmax=35)
pcm  = ax.pcolormesh(x_edges, z_edges, C2d, cmap=cmap, norm=norm, shading="flat")
ax.contour(x_centers, z_mid, C2d,
           levels=[17.5], colors="black", linewidths=1.5, linestyles="--")
ax.axvline(toe_b, color="red", lw=1.5, ls=":",
           label=f"Toe @ {toe_b:.0f} m from coast")
fig.colorbar(pcm, ax=ax, label="Concentration (ppt)")
ax.set_xlabel("Distance from coast [m]  (0 = sea, 1500 = inland)", fontsize=12)
ax.set_ylabel("Elevation [m]", fontsize=12)
ax.set_title(f"Part B — Salinity Distribution (Recharge R={R_val} m/day)  |  Toe = {toe_b:.0f} m",
             fontsize=13)
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(os.path.join(ws, "partB_salinity.png"), dpi=150)
plt.close()
print("✓ Salinity plot saved → partB_salinity.png")

# =============================================================================
# SUMMARY PRINT
# =============================================================================
print("\n" + "="*50)
print("  PART B — RECHARGE-DRIVEN MODEL SUMMARY")
print("="*50)
print(f"  Recharge rate       : {R_val} m/day")
print(f"  Sea head (CHD)      : 0.0 m")
print(f"  Inland boundary     : NONE (CHD removed)")
print(f"  Toe position        : {toe_b:.1f} m from coast")
print(f"  Max head (top layer): {head_top.max():.4f} m")
print("="*50)

# =============================================================================
# PART B — EXTENSION: RECHARGE SENSITIVITY STUDY
# Appended after the base Part B simulation (R = 0.002 m/day) completes.
# ONLY the sweep loop + plotting are added here.
# =============================================================================

# =============================================================================
# R-SWEEP SECTION 1 — RECHARGE VALUES & STORAGE DICTS
# =============================================================================
R_values = [0.0005, 0.0010, 0.0015]   # m/day

toe_results_R  = {}   # R → toe position (m from coast)
head_results_R = {}   # R → head profile along top layer (array, ncol)
maxh_results_R = {}   # R → max head in top layer (scalar)
conc_results_R = {}   # R → full 2D concentration array (nlay, ncol)

# =============================================================================
# R-SWEEP SECTION 2 — LOOP OVER RECHARGE VALUES
# =============================================================================
for R in R_values:
    print(f"\n{'='*55}")
    print(f"  Running: R = {R} m/day")
    print(f"{'='*55}")

    # ── Update RCH array ────────────────────────────────────────────────────
    rch_new = np.full((nrow, ncol), R, dtype=float)
    gwf.rcha.recharge.set_data({0: rch_new})

    # ── Re-write and run ────────────────────────────────────────────────────
    sim.write_simulation(silent=True)
    success, buff = sim.run_simulation(silent=True, report=True)

    if not success:
        print(f"  ✗ FAILED for R = {R} m/day")
        for line in buff:
            print(line)
        continue

    # ── Extract results ─────────────────────────────────────────────────────
    conc_R = load_concentration(ws, gwt_name)
    head_R = load_head(ws, gwf_name)

    toe_R  = get_toe_position(conc_R, x_centers)
    prof_R = head_R[0, 0, :]          # top-layer head profile, shape (ncol,)
    maxh_R = prof_R.max()

    toe_results_R[R]  = toe_R
    head_results_R[R] = prof_R
    maxh_results_R[R] = maxh_R
    conc_results_R[R] = conc_R

    print(f"  ✓ Toe = {toe_R:.1f} m | Max head = {maxh_R:.4f} m")

# =============================================================================
# R-SWEEP SECTION 3 — SUMMARY TABLE
# =============================================================================
print("\n" + "="*52)
print(f"  {'Recharge (m/day)':<22} {'Toe (m)':<15} {'Max Head (m)'}")
print("="*52)
for R in R_values:
    if R in toe_results_R:
        print(f"  {R:<22.4f} {toe_results_R[R]:<15.1f} {maxh_results_R[R]:.4f}")
print("="*52)

# =============================================================================
# R-SWEEP SECTION 4 — PLOT 1: TOE POSITION vs RECHARGE
# =============================================================================
R_plot   = [R for R in R_values if R in toe_results_R]
toe_plot = [toe_results_R[R] for R in R_plot]

fig1, ax1 = plt.subplots(figsize=(7, 5))
ax1.plot(R_plot, toe_plot, "o-",
         color="royalblue", lw=2, ms=9, markerfacecolor="red",
         zorder=3)
for R, toe in zip(R_plot, toe_plot):
    ax1.annotate(f"{toe:.0f} m",
                 xy=(R, toe), xytext=(8, 4),
                 textcoords="offset points", fontsize=10,
                 color="black")
ax1.set_xlabel("Recharge [m/day]", fontsize=12)
ax1.set_ylabel("Toe Position [m from coast]", fontsize=12)
ax1.set_title("Wedge Toe vs Recharge Rate", fontsize=13)
ax1.grid(True, linestyle="--", alpha=0.5)
# Annotate expected trend
ax1.annotate("↑ R → toe retreats seaward",
             xy=(R_plot[1], toe_plot[1]),
             xytext=(0.55, 0.65), textcoords="axes fraction",
             fontsize=9, color="gray",
             arrowprops=dict(arrowstyle="->", color="gray", lw=0.8))
plt.tight_layout()
plt.savefig(os.path.join(ws, "partB_toe_vs_recharge.png"), dpi=150)
plt.close()
print("\n✓ Plot 1 saved → partB_toe_vs_recharge.png")

# =============================================================================
# R-SWEEP SECTION 5 — PLOT 2: HEAD PROFILES (overlaid)
# =============================================================================
colors_head = {0.0015: "steelblue", 0.0020: "seagreen", 0.0027: "darkorange"}

fig2, ax2 = plt.subplots(figsize=(11, 5))
for R in R_plot:
    ax2.plot(x_centers, head_results_R[R],
             color=colors_head.get(R, "gray"), lw=2,
             label=f"R = {R} m/day  (max h = {maxh_results_R[R]:.3f} m)")
ax2.set_xlabel("Distance from coast [m]  (0 = sea, 1500 = inland)", fontsize=12)
ax2.set_ylabel("Hydraulic head [m]", fontsize=12)
ax2.set_title("Head Profiles — Recharge Sensitivity Study", fontsize=13)
ax2.legend(fontsize=10)
ax2.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(os.path.join(ws, "partB_head_profiles.png"), dpi=150)
plt.close()
print("✓ Plot 2 saved → partB_head_profiles.png")

# =============================================================================
# R-SWEEP SECTION 6 — PLOT 3: SALINITY DISTRIBUTIONS (3 subplots)
# =============================================================================
n_R     = len(R_plot)
x_edges = np.concatenate([[0], np.cumsum(delr)])
z_edges = np.concatenate([[tops[0]], bots])
z_mid   = (tops + bots) / 2.0
cmap    = plt.cm.RdYlBu_r
norm    = mcolors.TwoSlopeNorm(vmin=0, vcenter=17.5, vmax=35)

fig3, axes3 = plt.subplots(n_R, 1,
                            figsize=(14, 4 * n_R),
                            sharex=True, sharey=True)
if n_R == 1:
    axes3 = [axes3]   # ensure iterable for single-case edge

for ax, R in zip(axes3, R_plot):
    C2d = conc_results_R[R][:, 0, :]
    pcm = ax.pcolormesh(x_edges, z_edges, C2d,
                        cmap=cmap, norm=norm, shading="flat")
    # 17.5 ppt interface — white line
    ax.contour(x_centers, z_mid, C2d,
               levels=[17.5], colors="white", linewidths=1.8,
               linestyles="-")
    # Toe — vertical dashed line
    ax.axvline(toe_results_R[R], color="red", lw=1.5, ls="--",
               label=f"Toe @ {toe_results_R[R]:.0f} m")
    fig3.colorbar(pcm, ax=ax, label="Conc (ppt)")
    ax.set_ylabel("Elevation [m]", fontsize=10)
    ax.set_title(f"R = {R} m/day  |  Toe = {toe_results_R[R]:.0f} m from coast",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9)

axes3[-1].set_xlabel("Distance from coast [m]  (0 = sea, 1500 = inland)", fontsize=12)
plt.suptitle("Salinity Distribution — Recharge Sensitivity Study",
             fontsize=13, y=1.005)
plt.tight_layout()
plt.savefig(os.path.join(ws, "partB_salinity_sweep.png"), dpi=150, bbox_inches="tight")
plt.close()
print("✓ Plot 3 saved → partB_salinity_sweep.png")

# =============================================================================
# FINAL SUMMARY
# =============================================================================
print("\n" + "="*52)
print("  PART B — RECHARGE SENSITIVITY SUMMARY")
print("="*52)
print(f"  {'Recharge (m/day)':<22} {'Toe (m)':<15} {'Max Head (m)'}")
print("-"*52)
for R in R_plot:
    print(f"  {R:<22.4f} {toe_results_R[R]:<15.1f} {maxh_results_R[R]:.4f}")
print("="*52)
print("\nOutput files (all in ./swi_partB/):")
print("  partB_toe_vs_recharge.png   — Plot 1: Toe vs R")
print("  partB_head_profiles.png     — Plot 2: Head profiles")
print("  partB_salinity_sweep.png    — Plot 3: Salinity subplots")