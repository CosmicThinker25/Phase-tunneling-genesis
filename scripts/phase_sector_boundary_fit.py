"""
phase_sector_boundary_fit.py
----------------------------

Extracts and plots the A/C boundary ("Martin curve") in the
(m_phi, k_rot) plane for the physical branch delta_phi_ini = 0.01.

It:
  - reads results_phase_sectors/phase_sectors_summary.csv
  - selects rows with q=1.0 and delta_phi_ini=0.01
  - for each m_phi, finds the transition between sectors A and C
  - defines k_crit(m_phi) as the midpoint between the last A and first C
  - saves the boundary points to CSV
  - plots the boundary over the sector map

Outputs:
  results_phase_sectors/phase_sector_boundary.csv
  results_phase_sectors/phase_sector_boundary.png
"""

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

SUMMARY_FILE = "results_phase_sectors/phase_sectors_summary.csv"
BASE_DIR = "results_phase_sectors"
OUT_CSV = os.path.join(BASE_DIR, "phase_sector_boundary.csv")
OUT_FIG = os.path.join(BASE_DIR, "phase_sector_boundary.png")

if not os.path.isfile(SUMMARY_FILE):
    raise RuntimeError("Summary file not found. Run phase_sector_scan.py first.")

# ---------------------------------------------------------------
# 1) Load rows
# ---------------------------------------------------------------
rows = []
with open(SUMMARY_FILE, "r") as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

# Convert basic types
for r in rows:
    r["m_phi"] = float(r["m_phi"])
    r["k_rot"] = float(r["k_rot"])
    r["q"] = float(r["q"])
    r["delta_phi_ini"] = float(r["delta_phi_ini"])
    r["sector"] = r["sector"]

# Filter q and delta_phi_ini
q_target = 1.0
dphi_ini_target = 0.01

rows_phy = [
    r for r in rows
    if abs(r["q"] - q_target) < 1e-9
    and abs(r["delta_phi_ini"] - dphi_ini_target) < 1e-6
]

if not rows_phy:
    raise RuntimeError("No rows with q={} and delta_phi_ini={}."
                       .format(q_target, dphi_ini_target))

# ---------------------------------------------------------------
# 2) Build lists for the sector map (for plotting background)
# ---------------------------------------------------------------
m_list = sorted(set(r["m_phi"] for r in rows_phy))
k_list = sorted(set(r["k_rot"] for r in rows_phy))

code = {"A": 0, "B": 1, "C": 2}
grid = np.zeros((len(m_list), len(k_list)), dtype=int)

for r in rows_phy:
    i = m_list.index(r["m_phi"])
    j = k_list.index(r["k_rot"])
    grid[i, j] = code.get(r["sector"], 2)

# ---------------------------------------------------------------
# 3) For each m_phi, find A/C boundary
# ---------------------------------------------------------------
boundary_points = []

for i, m in enumerate(m_list):
    # Sort all entries for this m by k_rot
    rows_m = [r for r in rows_phy if abs(r["m_phi"] - m) < 1e-9]
    rows_m.sort(key=lambda r: r["k_rot"])

    ks = np.array([r["k_rot"] for r in rows_m])
    sectors = [r["sector"] for r in rows_m]

    # Find indices where sector == A and sector == C
    idx_A = [j for j, s in enumerate(sectors) if s == "A"]
    idx_C = [j for j, s in enumerate(sectors) if s == "C"]

    if not idx_A or not idx_C:
        # For this m, boundary not well-defined (all A or all C)
        continue

    # "Last A" and "first C" in k-order
    k_last_A = ks[max(idx_A)]
    k_first_C = ks[min(idx_C)]

    if k_first_C <= k_last_A:
        # Degenerate / overlapping, skip
        continue

    k_crit = 0.5 * (k_last_A + k_first_C)
    boundary_points.append((m, k_crit))

# Save boundary to CSV
with open(OUT_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["m_phi", "k_crit"])
    for m, kc in boundary_points:
        w.writerow([m, kc])

print("Boundary points saved to:", OUT_CSV)

# ---------------------------------------------------------------
# 4) Plot sector map + boundary
# ---------------------------------------------------------------
plt.figure(figsize=(8, 6))

cmap = plt.colormaps.get_cmap("viridis").resampled(3)
im = plt.imshow(
    grid,
    origin="lower",
    cmap=cmap,
    extent=[min(k_list), max(k_list), min(m_list), max(m_list)],
    aspect="auto",
)
plt.colorbar(im, ticks=[0, 1, 2], label="Sector")
plt.clim(-0.5, 2.5)

plt.xlabel(r"$k_{\rm rot}$")
plt.ylabel(r"$m_\phi$")
plt.title(
    rf"Asymptotic Sectors and A/C Boundary for $q={q_target}$, "
    rf"$\Delta\phi_{{\rm ini}}={dphi_ini_target}$"
)

# Overlay boundary points
if boundary_points:
    m_b, k_b = zip(*boundary_points)
    plt.plot(k_b, m_b, "wo-", linewidth=2.0, markersize=6,
             label="A/C boundary (Martin curve)")
    plt.legend(frameon=False)
else:
    print("WARNING: No boundary points detected. Check sector pattern.")

plt.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(OUT_FIG, dpi=300)
plt.show()

print("Boundary figure saved to:", OUT_FIG)
