# money_scan.py
# Script principal (ejecutable) para el "Money Plot":
#   P_surv(m_phi, k_rot) bajo prior uniforme vs prior WKB+Stark (Kernel 2)
#
# Ejecutar desde la raíz:
#   python src/money_scan.py
#
# Outputs:
#   data/grid_results.csv
#   data/grid_results.npz
#   figs/money_plot_uniform.png
#   figs/money_plot_wkb.png
#   figs/money_plot_wkb_minus_uniform.png

from __future__ import annotations

import os
import sys
import time
import csv
import argparse
from dataclasses import asdict
import numpy as np
import matplotlib.pyplot as plt

# Asegurar imports desde src/
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from tunneling_sampler import TunnelingSamplerWKBStark, TunnelingConfig
from phase_dynamics import integrate, DynamicsConfig


# -------------------------
# Utils: progreso y rutas
# -------------------------

def ensure_dirs(root: str) -> tuple[str, str]:
    data_dir = os.path.join(root, "data")
    figs_dir = os.path.join(root, "figs")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)
    return data_dir, figs_dir


def progress_bar(i: int, n: int, t0: float, prefix: str = "") -> str:
    # Barra de 30 chars
    width = 30
    frac = (i + 1) / n
    filled = int(width * frac)
    bar = "█" * filled + "·" * (width - filled)
    dt = time.time() - t0
    rate = (i + 1) / dt if dt > 0 else 0.0
    eta = (n - (i + 1)) / rate if rate > 0 else float("inf")
    eta_str = f"{eta:6.1f}s" if np.isfinite(eta) else "  inf s"
    return f"{prefix}[{bar}] {i+1}/{n}  {100*frac:5.1f}%  ETA {eta_str}"


def make_grid(center: float, half_width: float, n: int) -> np.ndarray:
    if n <= 1:
        return np.array([center], dtype=float)
    return np.linspace(center - half_width, center + half_width, n, dtype=float)


# -------------------------
# Core: supervivencia
# -------------------------

def survival_probability_uniform(
    rng: np.random.Generator,
    m_phi: float,
    k_rot: float,
    n_shots: int,
    dyn_cfg: DynamicsConfig
) -> float:
    # φ_ini uniforme en [0, π]
    phis = rng.uniform(0.0, np.pi, size=int(n_shots))
    surv = 0
    for phi_ini in phis:
        out = integrate(phi_ini=float(phi_ini), m_phi=float(m_phi), k_rot=float(k_rot), cfg=dyn_cfg)
        if out["sector"] == "C":
            surv += 1
    return surv / float(n_shots)


def survival_probability_wkb(
    rng_seed: int,
    m_phi: float,
    k_rot: float,
    n_shots: int,
    tun_cfg: TunnelingConfig,
    dyn_cfg: DynamicsConfig
) -> tuple[float, dict]:
    sampler = TunnelingSamplerWKBStark(m_phi=float(m_phi), k_rot=float(k_rot), cfg=tun_cfg, seed=int(rng_seed))
    phis = sampler.sample(int(n_shots))

    surv = 0
    for phi_ini in phis:
        out = integrate(phi_ini=float(phi_ini), m_phi=float(m_phi), k_rot=float(k_rot), cfg=dyn_cfg)
        if out["sector"] == "C":
            surv += 1

    return surv / float(n_shots), sampler.info()


# -------------------------
# Plotting
# -------------------------

def plot_heatmap(Z: np.ndarray, x: np.ndarray, y: np.ndarray, title: str, outpath: str):
    plt.figure(figsize=(8, 6))
    # imshow con extent para que los ejes sean m_phi (x) y k_rot (y)
    plt.imshow(
        Z,
        origin="lower",
        aspect="auto",
        extent=[x[0], x[-1], y[0], y[-1]]
    )
    plt.colorbar(label="P_surv (Sector C)")
    plt.xlabel("m_phi")
    plt.ylabel("k_rot")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


# -------------------------
# Main
# -------------------------

def main():
    ap = argparse.ArgumentParser(description="Money Plot scan: uniform vs WKB+Stark prior")
    ap.add_argument("--m-center", type=float, default=1.965, help="Centro de m_phi")
    ap.add_argument("--m-half", type=float, default=0.25, help="Semi-ancho del barrido en m_phi")
    ap.add_argument("--m-n", type=int, default=21, help="Puntos en m_phi")

    ap.add_argument("--k-center", type=float, default=0.33, help="Centro de k_rot")
    ap.add_argument("--k-half", type=float, default=0.33, help="Semi-ancho del barrido en k_rot")
    ap.add_argument("--k-n", type=int, default=21, help="Puntos en k_rot")

    ap.add_argument("--shots", type=int, default=40, help="Tiros por punto (phi_ini samples)")
    ap.add_argument("--seed", type=int, default=12345, help="Seed global reproducible")

    # Túnel (Kernel 2)
    ap.add_argument("--alpha", type=float, default=1.0, help="Alpha WKB")
    ap.add_argument("--gamma", type=float, default=0.5, help="Gamma Stark (epsilon=gamma*k_rot)")
    ap.add_argument("--ngrid", type=int, default=4000, help="Resolución de la CDF del sampler")
    ap.add_argument("--measure-sinphi", action="store_true", help="Multiplica prior por sin(phi)")

    # Dinámica: reducir trazas para acelerar
    ap.add_argument("--max-trace", type=int, default=200, help="Máx puntos guardados en traza (fase)")
    args = ap.parse_args()

    # Raíz del proyecto (PhaseTunneling/)
    root = os.path.abspath(os.path.join(THIS_DIR, os.pardir))
    data_dir, figs_dir = ensure_dirs(root)

    # Grids
    m_grid = make_grid(args.m_center, args.m_half, args.m_n)
    k_grid = make_grid(args.k_center, args.k_half, args.k_n)

    # Configs
    tun_cfg = TunnelingConfig(
        alpha=float(args.alpha),
        gamma=float(args.gamma),
        n_grid=int(args.ngrid),
        use_measure_sinphi=bool(args.measure_sinphi),
    )

    # Dinámica: usamos tu DynamicsConfig pero reduciendo max_trace_points para velocidad
    dyn_cfg = DynamicsConfig(max_trace_points=int(args.max_trace))

    # RNG global (para uniforme). Para WKB usamos seed derivada por punto.
    rng = np.random.default_rng(int(args.seed))

    # Matrices resultado (k x m)
    P_uni = np.zeros((k_grid.size, m_grid.size), dtype=float)
    P_wkb = np.zeros((k_grid.size, m_grid.size), dtype=float)
    turning = np.full((k_grid.size, m_grid.size), np.nan, dtype=float)
    fallback = np.zeros((k_grid.size, m_grid.size), dtype=int)

    # CSV output
    csv_path = os.path.join(data_dir, "grid_results.csv")
    npz_path = os.path.join(data_dir, "grid_results.npz")

    total = int(k_grid.size * m_grid.size)
    t0 = time.time()
    idx = 0

    print("\n=== PhaseTunneling: Money Plot scan ===")
    print(f"Root: {root}")
    print(f"Output CSV: {csv_path}")
    print(f"m_phi grid: {m_grid[0]:.4f} .. {m_grid[-1]:.4f}  (n={m_grid.size})")
    print(f"k_rot grid: {k_grid[0]:.4f} .. {k_grid[-1]:.4f}  (n={k_grid.size})")
    print(f"shots per point: {args.shots}")
    print(f"WKB: alpha={args.alpha}, gamma={args.gamma}, ngrid={args.ngrid}, measure_sinphi={args.measure_sinphi}")
    print(f"Dynamics: max_trace_points={args.max_trace}")
    print("======================================\n")

    # Abrir CSV y escribir cabecera
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "m_phi", "k_rot",
            "P_surv_uniform", "P_surv_wkb",
            "turning_point", "fallback",
            "alpha", "gamma", "ngrid", "shots", "seed"
        ])

        # Scan
        for iy, k in enumerate(k_grid):
            for ix, m in enumerate(m_grid):
                # Progreso
                print("\r" + progress_bar(idx, total, t0, prefix="Scanning "), end="")
                idx += 1

                # Uniform
                p_u = survival_probability_uniform(rng, m, k, args.shots, dyn_cfg)

                # WKB: seed por punto (determinista)
                # Mezcla simple reproducible
                point_seed = int(args.seed + 100000 * iy + ix)
                p_w, info = survival_probability_wkb(point_seed, m, k, args.shots, tun_cfg, dyn_cfg)

                P_uni[iy, ix] = p_u
                P_wkb[iy, ix] = p_w

                tp = info.get("turning_point", None)
                fb = int(info.get("fallback", False))
                turning[iy, ix] = float(tp) if tp is not None else np.nan
                fallback[iy, ix] = fb

                # CSV row
                w.writerow([
                    float(m), float(k),
                    float(p_u), float(p_w),
                    "" if tp is None else float(tp),
                    fb,
                    float(args.alpha), float(args.gamma), int(args.ngrid),
                    int(args.shots), int(args.seed)
                ])

    print("\n\nScan terminado.")

    # Guardar NPZ para replot rápido
    np.savez(
        npz_path,
        m_grid=m_grid,
        k_grid=k_grid,
        P_uni=P_uni,
        P_wkb=P_wkb,
        turning=turning,
        fallback=fallback,
        tunneling_cfg=asdict(tun_cfg),
        dynamics_cfg=asdict(dyn_cfg),
        meta=dict(seed=int(args.seed), shots=int(args.shots))
    )
    print(f"Guardado: {npz_path}")

    # Figuras
    out_uni = os.path.join(figs_dir, "money_plot_uniform.png")
    out_wkb = os.path.join(figs_dir, "money_plot_wkb.png")
    out_diff = os.path.join(figs_dir, "money_plot_wkb_minus_uniform.png")

    plot_heatmap(P_uni, m_grid, k_grid, "P_surv (Uniform prior)", out_uni)
    plot_heatmap(P_wkb, m_grid, k_grid, "P_surv (WKB+Stark prior)", out_wkb)

    diff = P_wkb - P_uni
    plt.figure(figsize=(8, 6))
    plt.imshow(
        diff,
        origin="lower",
        aspect="auto",
        extent=[m_grid[0], m_grid[-1], k_grid[0], k_grid[-1]]
    )
    plt.colorbar(label="ΔP_surv = P_wkb - P_uniform")
    plt.xlabel("m_phi")
    plt.ylabel("k_rot")
    plt.title("Money Plot difference (WKB - Uniform)")
    plt.tight_layout()
    plt.savefig(out_diff, dpi=200)
    plt.close()

    print(f"Figuras guardadas en: {figs_dir}")
    print("Listo.\n")


if __name__ == "__main__":
    main()
