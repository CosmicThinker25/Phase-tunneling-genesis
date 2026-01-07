"""
analyse_mphi_krot_critical.py
-----------------------------

Post-procesado del escaneo de sectores de fase para obtener

    m_phi,crit(k_rot)

a partir de los resultados ya existentes en:

    results_phase_sectors/phase_sector_probabilities.csv

El script:
  - Reconstruye la malla (m_phi, k_rot) -> (P_A, P_B, P_C)
  - Calcula m_phi,crit(k_rot) como el punto donde P_A cruza 0.5
    (interpolación lineal entre puntos vecinos en m_phi).
  - Genera:
      * JSON   -> results_phase_sectors/mphi_crit_vs_krot.json
      * Figura -> results_phase_sectors/mphi_crit_vs_krot.png
      * Figura -> results_phase_sectors/PA_map_mphi_krot_with_mcrit.png
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# 1. Rutas y lectura del CSV
# ---------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results_phase_sectors")

csv_path = os.path.join(RESULTS_DIR, "phase_sector_probabilities.csv")
if not os.path.isfile(csv_path):
    raise FileNotFoundError(
        f"No se encuentra el CSV esperado:\n  {csv_path}\n"
        "Asegúrate de ejecutar este script desde 'limite de martin' y de que "
        "el escaneo de sectores ya se ha realizado."
    )

print("📂 Leyendo datos desde:", csv_path)
df = pd.read_csv(csv_path)

# Se asume que el CSV tiene columnas: m_phi, k_rot, P_A, P_B, P_C
required_cols = {"m_phi", "k_rot", "P_A"}
if not required_cols.issubset(df.columns):
    raise ValueError(
        f"El CSV no contiene las columnas necesarias {required_cols}. "
        f"Columnas presentes: {list(df.columns)}"
    )

# ---------------------------------------------------------------------
# 2. Reconstruir malla y ordenar
# ---------------------------------------------------------------------

m_values = np.sort(df["m_phi"].unique())
k_values = np.sort(df["k_rot"].unique())

print(f"🔢 Valores únicos de m_phi: {m_values}")
print(f"🔢 Valores únicos de k_rot: {k_values}")

# Crear matrices PA(m,k) para referencia (no estrictamente necesario pero útil)
PA_grid = np.zeros((len(k_values), len(m_values)))

for i_k, k in enumerate(k_values):
    sub = df[df["k_rot"] == k].sort_values("m_phi")
    # Por si acaso: asegurar alineación con m_values
    sub = sub.set_index("m_phi").reindex(m_values).reset_index()
    PA_grid[i_k, :] = sub["P_A"].values

# ---------------------------------------------------------------------
# 3. Calcular m_phi,crit(k_rot) a partir de PA=0.5
# ---------------------------------------------------------------------

def find_mphi_crit(m_arr, PA_arr, target=0.5):
    """
    Dado un array ordenado m_arr y PA_arr, busca el intervalo donde
    PA cruza 'target' y devuelve el m_phi interpolado.
    Si no encuentra cruce, devuelve None.
    """
    for i in range(len(m_arr) - 1):
        PA1, PA2 = PA_arr[i], PA_arr[i + 1]
        if (PA1 - target) * (PA2 - target) <= 0:
            # Cruce o toque del nivel target
            if PA2 == PA1:
                return float(m_arr[i])
            frac = (target - PA1) / (PA2 - PA1)
            return float(m_arr[i] + frac * (m_arr[i + 1] - m_arr[i]))
    return None

mphi_crit_list = []
for i_k, k in enumerate(k_values):
    PA_row = PA_grid[i_k, :]
    mcrit = find_mphi_crit(m_values, PA_row, target=0.5)
    mphi_crit_list.append(mcrit)

# ---------------------------------------------------------------------
# 4. Mostrar resultados en consola y guardar JSON
# ---------------------------------------------------------------------

print("\n===============================================")
print(" m_phi,crit(k_rot) a partir de P_A(m_phi, k_rot)")
print("  (interpolación lineal para P_A = 0.5)")
print("===============================================")

results = []
for k, mcrit in zip(k_values, mphi_crit_list):
    if mcrit is None:
        print(f"k_rot = {k:5.3f} -> SIN cruce claro P_A=0.5")
    else:
        print(f"k_rot = {k:5.3f} -> m_phi,crit ≈ {mcrit:6.3f}")
    results.append({"k_rot": float(k), "m_phi_crit": None if mcrit is None else float(mcrit)})

json_path = os.path.join(RESULTS_DIR, "mphi_crit_vs_krot.json")
with open(json_path, "w") as f:
    json.dump({"data": results}, f, indent=2)

print("\n📝 Resumen guardado en:", json_path)

# ---------------------------------------------------------------------
# 5. Figura: mapa P_A(m_phi, k_rot) con m_phi,crit(k_rot)
# ---------------------------------------------------------------------

# (a) Mapa 2D de P_A
fig, ax = plt.subplots(figsize=(8, 5))
im = ax.imshow(
    PA_grid,
    origin="lower",
    aspect="auto",
    extent=[m_values[0], m_values[-1], k_values[0], k_values[-1]],
    vmin=0.0,
    vmax=1.0,
)

cbar = fig.colorbar(im, ax=ax)
cbar.set_label(r"$P_A$ (sincronía)", fontsize=12)

ax.set_xlabel(r"$m_\phi$", fontsize=12)
ax.set_ylabel(r"$k_{\mathrm{rot}}$", fontsize=12)
ax.set_title(r"Mapa $P_A(m_\phi, k_{\mathrm{rot}})$", fontsize=14)

# Dibujar la curva m_phi,crit(k_rot) encima si existe
valid_k = [k for k, m in zip(k_values, mphi_crit_list) if m is not None]
valid_m = [m for m in mphi_crit_list if m is not None]
if len(valid_k) > 0:
    ax.plot(valid_m, valid_k, "w--", linewidth=2, label=r"$m_{\phi,\mathrm{crit}}(k_{\mathrm{rot}})$")
    ax.legend(loc="lower right")

fig.tight_layout()
map_fig_path = os.path.join(RESULTS_DIR, "PA_map_mphi_krot_with_mcrit.png")
fig.savefig(map_fig_path, dpi=150)
plt.close(fig)

print("🖼  Figura de mapa guardada en:", map_fig_path)

# (b) m_phi,crit vs k_rot (si hay al menos un valor válido)
if len(valid_k) > 0:
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(valid_k, valid_m, "o-", label=r"$m_{\phi,\mathrm{crit}}$")
    ax2.set_xlabel(r"$k_{\mathrm{rot}}$")
    ax2.set_ylabel(r"$m_{\phi,\mathrm{crit}}$")
    ax2.set_title(r"$m_{\phi,\mathrm{crit}}$ vs $k_{\mathrm{rot}}$")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig2.tight_layout()
    curve_fig_path = os.path.join(RESULTS_DIR, "mphi_crit_vs_krot.png")
    fig2.savefig(curve_fig_path, dpi=150)
    plt.close(fig2)
    print("🖼  Figura m_phi,crit vs k_rot guardada en:", curve_fig_path)

print("\n✅ Análisis completado.")
