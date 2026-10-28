"""Experiment portfolio simulation and decision rules for ECON 5200 PS2."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def simulate_portfolio(
    n: int = 200,
    seed: int = 42,
    prob_positive: float = 0.20,
    baseline_conversion: float = 0.10,
    positive_effect_min: float = 0.01,
    positive_effect_max: float = 0.035,
    min_sample: int = 200,
    max_sample: int = 4000,
) -> pd.DataFrame:
    """Simulate A/B experiments with known true conversion-rate effects."""

    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= prob_positive <= 1:
        raise ValueError("prob_positive must be between 0 and 1")
    if not 0 < baseline_conversion < 1:
        raise ValueError("baseline_conversion must be between 0 and 1")
    if not 0 <= positive_effect_min <= positive_effect_max:
        raise ValueError("Invalid positive-effect range")
    if baseline_conversion + positive_effect_max >= 1:
        raise ValueError("Conversion probabilities must be below 1")
    if min_sample < 1 or max_sample < min_sample:
        raise ValueError("Invalid sample-size range")

    rng = np.random.default_rng(seed)

    is_positive = rng.random(n) < prob_positive

    true_effect = np.where(
        is_positive,
        rng.uniform(positive_effect_min, positive_effect_max, n),
        0.0
    )

    sample_size = rng.integers(min_sample, max_sample + 1, n)

    treatment_conversion = baseline_conversion + true_effect

    standard_error = np.sqrt(
        baseline_conversion * (1 - baseline_conversion) / sample_size
        + treatment_conversion * (1 - treatment_conversion) / sample_size
    )

    estimated_effect = rng.normal(true_effect, standard_error)

    log = pd.DataFrame({
        "experiment": np.arange(1, n + 1),
        "sample_size": sample_size,
        "true_effect": true_effect,
        "estimated_effect": estimated_effect,
        "standard_error": standard_error,
    })

    log["z_score"] = log["estimated_effect"] / log["standard_error"]
    log["p_value"] = 2 * stats.norm.sf(np.abs(log["z_score"]))

    return log


def apply_rule(log: pd.DataFrame, rule: str = "client") -> pd.Series:
    """Return shipping decisions for the selected rule.

    Rules:
        client: Ship experiments with two-sided p < 0.05.
        proposed: Require p < 0.05 and estimated effect >= 0.01.
    """

    if rule == "client":
        return log["p_value"] < 0.05

    elif rule == "proposed":
        return (
            (log["p_value"] < 0.05)
            & (log["estimated_effect"] >= 0.01)
        )

    else:
        raise ValueError(f"Unknown rule: {rule}")


def evaluate(log: pd.DataFrame, rule: str = "client") -> dict:
    """Evaluate total realised return, shipping count and false-ship rate."""

    decisions = apply_rule(log, rule)
    shipped = log.loc[decisions]

    n_shipped = len(shipped)

    realised_return = shipped["true_effect"].sum()

    false_ship_rate = (
        (shipped["true_effect"] == 0).mean()
        if n_shipped > 0 else np.nan
    )

    return {
        "realised_return": float(realised_return),
        "ship_count": int(n_shipped),
        "false_ship_rate": float(false_ship_rate),
    }


if __name__ == "__main__":

    # Demonstration
    portfolio = simulate_portfolio(n=200, seed=42)

    client_results = evaluate(portfolio, "client")
    proposed_results = evaluate(portfolio, "proposed")

    print("Client rule:", client_results)
    print("Proposed rule:", proposed_results)

    # Assertions
    assert len(portfolio) == 200
    assert (portfolio["true_effect"] >= 0).all()
    assert (portfolio["standard_error"] > 0).all()
    assert portfolio["sample_size"].between(200, 4000).all()
    assert portfolio["p_value"].between(0, 1).all()

    assert (
        proposed_results["ship_count"]
        <= client_results["ship_count"]
    )

    assert (
        proposed_results["realised_return"]
        <= client_results["realised_return"] + 1e-12
    )

    # Reproducibility check
    duplicate = simulate_portfolio(n=200, seed=42)
    pd.testing.assert_frame_equal(portfolio, duplicate)

    print("All assertions passed.")
