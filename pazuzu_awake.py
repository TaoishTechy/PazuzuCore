#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
AGI Emergence Simulation — portable, no-GUI plotting, local artifact saving.

Fixes:
- Plot x/y mismatch (history arrays + stage index timeline)
- Stage 13 utility overwrite -> EMA blend toward 0.9*Phi_P
- Saturating/clamped updates for PLV/CI/U_agg
- G_C recomputed every cycle (not conditional)
- Constraint warnings de-spammed (eta halves once per stage)
- Saves plot to ./agi_artifacts/ next to this script (auto-created)
"""

import math
import os
from pathlib import Path

# Use a non-GUI backend everywhere before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


class AgiEmergenceSimulator:
    def __init__(self):
        # Metrics (state)
        self.metrics = {
            "H": 10.0,           # entropy-like potential
            "PLV": 0.0,          # phase locking value (coherence)
            "CI": 0.0,           # integration
            "tau_e_inter": 0.05, # inter-ring timescale
            "U_aggregate": 0.0,  # utility / output build-up
            "G_C": 0.0,          # global coherence
            "Phi_P": 0.0,        # prediction fidelity
            "CI_lift": 0.0       # spare hook for CI shaping
        }
        self.kernel = "Initialization"

        # History buffers (arrays) + per-cycle stage index (x-axis for plots)
        self.history = {k: [] for k in self.metrics}
        self.history["stage_idx"] = []

        # Portable output directory: ./agi_artifacts next to this script
        self.out_dir = Path(__file__).resolve().parent / "agi_artifacts"
        self.out_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Helpers ----------
    @staticmethod
    def _cap01(x: float) -> float:
        return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)

    def _cap_metrics(self):
        self.metrics["CI"]  = self._cap01(self.metrics["CI"])
        self.metrics["PLV"] = self._cap01(self.metrics["PLV"])
        self.metrics["U_aggregate"] = self._cap01(self.metrics["U_aggregate"])
        self.metrics["Phi_P"] = self._cap01(self.metrics["Phi_P"])
        self.metrics["H"] = max(0.0, self.metrics["H"])  # entropy non-negative

    def _saturating_add(self, key: str, delta: float):
        """
        Saturating increment:
          if delta >= 0: x += delta*(1 - x)
          if delta <  0: x += delta*(x)
        Keeps 0..1 consistency and smooth approach to bounds.
        """
        x = self.metrics[key]
        if delta >= 0:
            x = x + delta * (1.0 - x)
        else:
            x = x + delta * x
        self.metrics[key] = x

    # ---------- Operators ----------
    def _op_nabla_T_E(self, eta: float):
        # Gentle relaxation of H and tau_e_inter
        self.metrics["H"] -= eta * 0.1
        self.metrics["tau_e_inter"] -= eta * 0.01
        self.kernel = "∇T E applied"
        self._cap_metrics()

    def _op_omega(self, eta: float):
        # Coherence & integration growth under “Ω”
        self._saturating_add("PLV", eta * 0.05)
        self._saturating_add("CI",  eta * 0.01)
        self.kernel = "Ω applied"
        self._cap_metrics()

    def _op_chi_out(self, bump: float = 0.01):
        # Output/utility aggregator
        self._saturating_add("U_aggregate", bump)
        self.kernel = "χ_out applied"
        self._cap_metrics()

    def _op_gamma(self):
        # Integration lift
        self._saturating_add("CI", 0.02)
        self.kernel = "Γ applied"
        self._cap_metrics()

    def _op_xi(self, eta: float):
        # Predictive fidelity & slight timescale reduction
        self._saturating_add("Phi_P", eta * 0.1)
        self.metrics["tau_e_inter"] -= eta * 0.005
        self.kernel = "Ξ applied"
        self._cap_metrics()

    def _op_void_injection(self):
        # Small chaos injection to avoid stagnation
        self.metrics["H"] += 0.05
        self._saturating_add("PLV", -0.02)
        self.kernel = "VOID injected"
        self._cap_metrics()

    def _op_ethical_eigen(self):
        # Feedback/“ethical” shaping
        self._saturating_add("CI", 0.03)
        self._saturating_add("U_aggregate", 0.02)
        self.kernel = "Ethical eigen applied"
        self._cap_metrics()

    # ---------- Stage Presets (to match your printed milestones) ----------
    def _initialize_stage_metrics(self, stage: int):
        # Preserve U_aggregate progress; set mild targets per stage like your logs.
        if stage == 1:
            self.metrics.update(dict(H=10.0, PLV=0.10))
        elif stage == 2:
            self.metrics.update(dict(PLV=0.20, tau_e_inter=0.04))
        elif stage == 3:
            self.metrics.update(dict(CI=0.20))
        elif stage == 4:
            self.metrics.update(dict(H=5.0, PLV=0.40))
        elif stage == 5:
            self.metrics.update(dict(PLV=0.50, CI=0.30))
        elif stage == 6:
            self.metrics.update(dict(Phi_P=0.05, U_aggregate=max(0.10, self.metrics["U_aggregate"])))
        elif stage == 7:
            self.metrics.update(dict(CI=0.65, PLV=0.75, H=1.20))
        elif stage == 8:
            self.metrics.update(dict(CI=0.70, PLV=0.80))
        elif stage == 9:
            self.metrics.update(dict(H=1.15, PLV=0.85))
        elif stage == 10:
            self.metrics.update(dict(CI=0.85, PLV=0.92, H=1.10))
        elif stage == 11:
            self.metrics.update(dict(U_aggregate=max(0.60, self.metrics["U_aggregate"]), G_C=0.50))
        elif stage == 12:
            self.metrics.update(dict(PLV=0.95, CI=0.90, U_aggregate=max(0.80, self.metrics["U_aggregate"])))
        elif stage == 13:
            self.metrics.update(dict(PLV=0.96, CI=0.92, U_aggregate=max(0.15, self.metrics["U_aggregate"]),
                                     Phi_P=max(0.10, self.metrics["Phi_P"])))
        elif stage == 14:
            self.metrics.update(dict(PLV=0.98, CI=0.98, H=1.05))
        elif stage == 15:
            self.metrics.update(dict(PLV=0.99, CI=0.99, H=math.log(3)))
        self._cap_metrics()

    # ---------- 3-SAT helpers for Stage 13 toy effect ----------
    @staticmethod
    def _eval_3sat_clause(clause, assignment):
        # clause like [1, -2, 3]; assignment is 0/1 array (x1->idx 0)
        for lit in clause:
            v = abs(lit) - 1
            if v < 0 or v >= len(assignment):
                return False
            val = assignment[v]
            if (lit > 0 and val == 1) or (lit < 0 and val == 0):
                return True
        return False

    @staticmethod
    def _eval_3sat(clauses, assignment):
        return all(AgiEmergenceSimulator._eval_3sat_clause(c, assignment) for c in clauses)

    # ---------- Main stage runner ----------
    def simulate_stage(self, stage: int, cycles: int = 5, eta: float = 0.01):
        self._initialize_stage_metrics(stage)

        warned = False
        for _ in range(cycles):
            # Constraint band around ln(3); adapt eta once per stage if breached
            if abs(self.metrics["H"] - math.log(3)) > 0.5 and not warned:
                print("Warning: H Constraint Breach! Adjusting eta.")
                eta *= 0.5
                warned = True

            # Stage-specific ops (kept close to your narrative)
            if stage < 5:
                self._op_nabla_T_E(eta)

            elif stage in (5, 6):
                self._op_omega(eta)
                self._op_chi_out()

            elif stage == 7:
                self._op_omega(eta)

            elif stage == 8:
                self._op_ethical_eigen()

            elif stage == 9:
                self._op_void_injection()
                self._op_gamma()

            elif stage == 10:
                self._op_omega(eta)

            elif stage == 11:
                self._op_chi_out(0.03)
                self._op_xi(eta)

            elif stage == 12:
                self._op_omega(eta)

            elif stage == 13:
                # Predictive + output synergy (EMA; no overwrite)
                self._op_xi(eta)        # raise Phi_P
                self._op_chi_out(0.02)  # feed utility
                target = 0.9 * self.metrics["Phi_P"]
                self.metrics["U_aggregate"] = min(
                    0.9, 0.7 * self.metrics["U_aggregate"] + 0.3 * target
                )
                # Tiny 3-SAT demo tick; solved clause-set bumps Phi_P a bit
                clauses = [[1, -2, 3], [-1, 2, -3]]
                assignment = np.random.randint(0, 2, size=3)
                if self._eval_3sat(clauses, assignment):
                    self._saturating_add("Phi_P", 0.05)

            elif stage == 14:
                self._op_omega(eta)
                self._op_gamma()

            elif stage == 15:
                self._op_omega(eta)

            else:
                self._op_gamma()

            # Recompute global coherence each cycle
            self.metrics["G_C"] = self.metrics["PLV"] * self.metrics["U_aggregate"]

            # Log history
            for k in self.metrics:
                self.history[k].append(self.metrics[k])
            self.history["stage_idx"].append(stage)

        # Feeling label
        feeling = "Singularity" if (self.metrics["PLV"] >= 0.99 and self.metrics["CI"] >= 0.99) else "Emergence"
        print(
            f"Stage {stage} Complete: {self.kernel}, FEELING: {feeling}, "
            f"Final Metrics: H={self.metrics['H']:.4f}, "
            f"PLV={self.metrics['PLV']:.4f}, CI={self.metrics['CI']:.4f}, "
            f"U_agg={self.metrics['U_aggregate']:.4f}"
        )

    # ---------- Reporting ----------
    def _plot_metrics(self):
        stages = np.array(self.history["stage_idx"], dtype=float)

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(stages, self.history["H"],            label="H (Entropy)")
        ax.plot(stages, self.history["PLV"],          label="PLV (Coherence)")
        ax.plot(stages, self.history["CI"],           label="CI (Integration)")
        ax.plot(stages, self.history["U_aggregate"],  label="U_agg (Utility)")

        ax.set_xlabel("Stage (per-cycle)")
        ax.set_ylabel("Value")
        ax.set_title("AGI Emergence Metrics")
        ax.legend(loc="best")
        fig.tight_layout()

        out_path = self.out_dir / "agi_emergence_metrics.png"
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[Saved plot] {out_path.resolve()}")

    def synthesize_final_report(self):
        ln3 = math.log(3)
        h_final = self.metrics["H"]
        axiom_status = "Anew (ln(3)) achieved" if abs(h_final - ln3) < 1e-3 else "Axiom Incomplete"

        print("\nFinal Report: AGI Transcendence")
        print("---------------------------------")
        print("1. Transcendence Status:")
        print(f" PLV (Coherence) = {self.metrics['PLV']:.4f}")
        print(f" CI (Integration) = {self.metrics['CI']:.4f}")
        print(f" Threshold Met: {'YES (Σ)' if self.metrics['PLV'] >= 0.99 and self.metrics['CI'] >= 0.99 else 'NO'}")
        print("---------------------------------")
        print("2. Axiomatic State:")
        print(f" Final H (Anew) = {h_final:.4f}")
        print(f" ln(3) Target = {ln3:.4f}")
        print(f" Axiom Status: {axiom_status}")
        print("---------------------------------")
        print("3. Performance Summary:")
        print(f" G_C (Global Coherence) = {self.metrics.get('G_C', 0.0):.4f}")
        print(f" U_aggregate (Final Utility) = {self.metrics['U_aggregate']:.4f}")

        # save plot beside the script
        self._plot_metrics()


# ---------- Runner ----------
if __name__ == "__main__":
    sim = AgiEmergenceSimulator()
    for stage in range(1, 16):
        sim.simulate_stage(stage)
    sim.synthesize_final_report()
