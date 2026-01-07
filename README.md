# Phase-Tunneling Cosmogenesis: Structural Origin of a Critical Phase Mass in a Finite-Window Regime

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18168587.svg)](https://zenodo.org/records/18168587)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Repository for the study of emergent time, phase bifurcation, and quantum priors in CPT-symmetric cosmogenesis.**

---

## 🔬 Scientific Description (Technical)

This repository contains the reproduction package (manuscript and Python source code) for the study of **Phase-Tunneling Cosmogenesis**. This work investigates the emergence of non-trivial dynamical histories from a pre-dynamical regime within the framework of CPT-symmetric ("Twin") cosmology.

### Key Scientific Contributions
The central finding of this work is the identification of a robust critical phase mass, **$m_{\phi,\mathrm{crit}} \simeq 1.965$**, which marks a sharp structural bifurcation in the phase configuration space. By analyzing the competition between a periodic effective potential, Hubble friction, and a decaying rotational source term, we demonstrate that:

1.  **Finite-Window Criticality:** The "escape sector" (Sector C)—characterized by trajectories that avoid immediate relaxation—is intrinsically a **finite-window phenomenon** (occurring within $\Delta N \simeq 9.2$ e-folds). It is defined by transient non-relaxation rather than asymptotic divergence.
2.  **Asymptotic Null Result:** Extending the integration to arbitrarily late times ($N \to \infty$) yields a universal synchronization ($\Delta \phi \to 0$). This null result confirms that the critical threshold is not an eternal attractor but a geometric feature governing the emergence of transient complexity (interpreted as Barbour-style "Time Capsules").
3.  **Quantum Tunneling Prior:** We introduce a semiclassical WKB tunneling mechanism as a prior for initial conditions. The analysis reveals that the classical threshold ($m_{\phi,\mathrm{crit}}$) is structurally robust and insensitive to the specific choice of the quantum prior (WKB vs. Uniform), as demonstrated by differential survival maps included in the analysis. A primordial rotational coupling is shown to act as a catalyst by reducing the Euclidean action via a Stark-like effect.

---

## 🔭 Plain English Summary (Divulgative)

### How Time Escapes the Trap

How does a universe "start" if there is no time before it? This project explores a model of **Cosmogenesis** (the origin of the cosmos) where time is not a background stage, but a temporary "escape" from a static state.

In this model, the universe—and its "twin" partner—start locked in synchronization. To create a history, they must break apart. We found a specific "magic number" or critical mass (**1.965**) that determines whether a universe can break free and develop a history or if it immediately collapses back into silence.

**Three Big Ideas in this Work:**
* **The Window of Opportunity:** We show that "time" as we know it is a temporary bubble. If you run the clock forever, everything eventually syncs back up. Real physical history happens only in a brief, finite window of expansion.
* **The Critical Threshold:** Just like a rocket needs escape velocity to leave Earth, the phase of the universe needs a specific mass to escape immediate collapse. We calculated this value precisely at 1.965.
* **The Quantum Kick:** Before the clock starts, quantum mechanics rolls the dice. We modeled how the universe "tunnels" through a barrier to get started. Crucially, we found that the rules of the game (the 1.965 threshold) don't change regardless of how the dice are rolled—the structure of the universe is robust.

---

## 📂 Repository Structure

The repository is organized as follows:

* **`src/`**: Python source code.
    * `phase_classifier_finite.py`: Operational classifier for trajectory sectors (A, B, C).
    * `compute_mphi_crit.py`: Utilities to calculate the critical mass threshold.
    * `tunneling_sampler.py`: Implementation of WKB and Uniform priors.
* **`figs/`**: High-resolution figures generated for the manuscript (including the "Money Plot" on robustness).
* **`archive_asymptotic_tests/`**: Scripts verifying the asymptotic null result (synchronization at late times).
* **`PhaseTunneling_paper_vFinal_Submission.pdf`**: The full scientific manuscript.

## 📝 Citation

If you use this code in your research, please cite the associated software via Zenodo:

```bibtex
@misc{phase_tunneling_2026,
  author       = {CosmicThinker},
  title        = {Phase-Tunneling Cosmogenesis: Structural Origin of a Critical Phase Mass in a Finite-Window Regime},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18168587},
  url          = {[https://zenodo.org/records/18168587](https://zenodo.org/records/18168587)}
}
