# Problem-Set-2-Econ_5200_PS2.ipynb
# ECON 5200 — Problem Set 2: The Underpowered Portfolio

## Project Overview

This project examines how statistical significance can lead companies to overestimate the value of their A/B testing programs. Using 200 simulated experiments with known true effects, I evaluate the client's existing decision rule, quantify the winner's curse, and propose an alternative approach to improve decision-making.

## Methodology

- **Portfolio Simulation:** Simulated 200 A/B experiments with varying sample sizes and true conversion-rate effects.
- **Statistical Significance:** Evaluated the client's p < 0.05 shipping rule and identified false discoveries.
- **Winner's Curse:** Measured how statistically significant results exaggerate true effects, particularly in underpowered experiments.
- **Bootstrap Analysis:** Used manual and SciPy bootstrap methods to estimate uncertainty in realised portfolio returns.
- **Alternative Decision Rule:** Compared the original rule with one requiring a minimum estimated improvement of 1 percentage point.
- **Sequential Testing:** Implemented and verified an always-valid sequential test using 1,000 null simulations.

## Key Findings

- The original rule shipped 32 experiments, with a false-ship rate of 28.1%.
- Estimated portfolio improvements exceeded actual improvements by 8.77 percentage-point units.
- The proposed rule reduced shipments to 27 and the false-ship rate to 14.8%, without sacrificing realised return in the simulation.
- The sequential test achieved a 1.8% false-positive rate, below the nominal 5% limit.

## Files

- `Econ_5200_PS2.ipynb` — Main analysis, visualizations, and CFO memo.
- `src/experiment_portfolio.py` — Reusable simulation and decision-rule functions.
- `ai-appendix.md` — AI prompts and responses documenting the sequential-testing implementation.

## Tools

Python, NumPy, Pandas, SciPy, Matplotlib, Google Colab, and Claude.
