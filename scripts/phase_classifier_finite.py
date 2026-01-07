"""
phase_classifier.py
-------------------

Clasifica una trayectoria Δφ(a) en uno de los sectores:
    A = Sincronía (Δφ → 0)
    B = Pico (Δφ tiene un máximo > π/2 pero < π)
    C = Escape / Divergencia (Δφ cruza π o se aleja sin volver)

Se importa desde:
- phase_sector_scan.py
- phase_sector_map.py
- phase_sector_map_delta.py
- phase_sector_boundary_fine_scan.py
- phase_sector_zoom_fractal.py
etc.

=====================================================
CRITERIOS FORMALES
=====================================================

Definición operativa:

Sector A:
    Δφ(a_final) < eps   (≈ converge a 0)

Sector C:
    max(Δφ(a)) > π      (cruza el valor crítico)  O
    Δφ(a_final) > 0.8 * max(Δφ)  (trayectoria que no decae)

Sector B:
    En cualquier otro caso → tiene un máximo intermedio
"""

import numpy as np


# ============================================================
# Clasificador principal
# ============================================================

def classify_sector(dphi_arr, eps=0.05):
    """
    Clasifica una trayectoria Δφ(a) en sectores A, B o C.

    Parámetros
    ----------
    dphi_arr : array-like
        Serie temporal de Δφ(a)
    eps : float
        Umbral para decidir si Δφ → 0

    Devuelve
    --------
    str : 'A', 'B' o 'C'
    """

    dphi = np.array(dphi_arr)
    final = dphi[-1]
    maxval = np.max(dphi)

    # --- Sector A: sincronía total ---
    if final < eps:
        return "A"

    # --- Sector C: escape / divergencia ---
    if maxval > np.pi:           # supera barrera crítica
        return "C"

    # si no cae tras el máximo → divergencia suave
    if final > 0.8 * maxval:
        return "C"

    # --- Sector B: pico intermedio ---
    return "B"
