# AI Appendix: Sequential Testing for the ECON 5200 A/B-Test Portfolio

**Tool:** Claude (Anthropic), claude.ai chat
**Date:** October 27, 2026
**Scope:** This appendix records the full exchange used to build and validate an always-valid sequential test for a 200-experiment simulated A/B-test portfolio. User prompts are reproduced verbatim. AI responses are reproduced in substance, with all reported figures copied from the code's actual output. All code is included exactly as delivered.

---

## Summary of the exchange

| Turn | User request | AI output | Status |
|---|---|---|---|
| 1 | Implement an always-valid sequential test (mSPRT or alpha-spending), explain naive peeking, evaluate on 1,000 null streams | Two-sample Gaussian mSPRT with plug-in variance option; null evaluation; horizon check | Superseded by Turn 3 |
| 2 | Put all code in one block | Combined single-file version of Turn 1 code | Superseded by Turn 3 |
| 3 | Fix statistical validity for Bernoulli data, match portfolio parameters, add checks, compare to naive testing, explain trade-offs | Exactly valid discordant-pair Beta-mixture e-process; exact Type I computation; 1,000-null simulation; power and portfolio comparison | **Final version** |
| 4 | Create this appendix | This file | — |

**Key correction made during the exchange:** the Turn 1 implementation used a plug-in variance for Bernoulli data, which is only approximately valid. The AI flagged this as a limitation but did not fix it. The user identified it as a substantive problem in Turn 3, and the final procedure replaces it with a test whose Type I guarantee is exact for two independent Bernoulli arms.

---

## Turn 1

### User prompt (verbatim)

```
[Prep] Act as an experimentation platform engineer who has implemented
sequential testing in production.

[Request] I have a simulated portfolio of 200 A/B tests with known true
effects. Implement an always-valid sequential test -- a mixture
sequential probability ratio test or an alpha-spending boundary -- that
allows continuous monitoring without inflating the Type I error rate.
Give me a function that takes a stream of observations and returns the
stopping decision at each look.

[Iterate] numpy and scipy only. Reuse my log structure. The function
must be testable against a pure-null stream.

[Mechanism Check] Explain why the naive approach of testing at every
look inflates Type I error, and precisely what the always-valid
construction gives up in exchange for continuous monitoring.

[Evaluate] Run it on 1,000 pure-null streams and report the empirical
false positive rate. If it is not close to 5%, tell me what is wrong.
```

### AI process

- Checked the upload folder and past chats for the user's log structure and portfolio file. Neither was found, so the AI built a stand-in per-look record isolated in a single function (`_log_record`) and stated this explicitly.
- Wrote a streaming two-sample mSPRT. The first full run (pure-Python loop, 6 configurations × 1,000 streams × 20,000 observations) exceeded the 300-second execution limit. The AI added a vectorized evaluator computing the identical statistic from cumulative sums and verified it against the streaming function (maximum |log Λ| difference 1.21e-16).

### AI response (substance)

**Method:** mixture SPRT with an N(0, τ²) mixture over the effect, closed form
Λₙ = √(V/(V+τ²)) · exp(d²τ² / (2V(V+τ²))), rejecting when Λ ≥ 1/α. Variance either known (`sigma`) or a running plug-in estimate after a burn-in.

**Results: 1,000 null streams, 20,000 observations each, α = 0.05, look after every observation**

| Configuration | FPR | 95% CI |
|---|---|---|
| mSPRT, Gaussian, known σ, τ=0.05 | 0.027 | 0.018–0.038 |
| mSPRT, Gaussian, plug-in variance | 0.027 | 0.018–0.038 |
| mSPRT, Bernoulli p=0.10, plug-in, τ=0.01 | 0.007 | 0.003–0.014 |
| mSPRT, Bernoulli, burn_in=2 (deliberately bad) | 0.021 | 0.013–0.031 |
| mSPRT, Gaussian, known σ, τ=0.5 | 0.040 | 0.029–0.054 |
| Naive z-test, 10 looks | 0.196 | 0.172–0.221 |
| Naive z-test, 200 looks | 0.439 | 0.408–0.470 |
| Naive z-test, every observation | 0.610 | 0.579–0.640 |

**Horizon check** (natural time u = τ²/V; the 20k-observation Gaussian run ends at u = 12.5):

| Horizon (u) | FPR |
|---|---|
| 1.25 | 0.001 |
| 12.5 | 0.027 |
| 125 | 0.041 |
| 1250 | 0.041 |

**Interpretation given:** 5% is an upper bound from Ville's inequality, not a target; the gap below 5% comes from the finite horizon and discrete monitoring. Plug-in variance with too small a burn-in roughly tripled the Bernoulli FPR (0.7% → 2.1%), identified as the real failure mode.

**Mechanism check given:**
- Naive peeking: each look is a valid 5% test, but the false-positive event is the union across correlated looks; by the law of the iterated logarithm |zₙ| eventually crosses any fixed bound, so FPR → 1 with unlimited peeking.
- What always-valid monitoring gives up: a higher boundary at any fixed n (|z| ≥ 3.05 at the end of the 20k run vs 1.96), unspent α on a finite horizon, power dependence on τ, exactness only under Gaussian known-variance assumptions, upward-biased effect estimates at stopping, and no multiplicity control across 200 tests.

---

## Turn 2

### User prompt (verbatim)

```
just give all the code in one block
```

### AI response (substance)

Combined the streaming test, naive comparator, generators, equivalence check, null evaluation and horizon check into one file. Verified it runs end to end in a `--quick` mode (small sizes, plumbing check only); the full-mode settings are identical to Turn 1.

### Code delivered (Turn 2, superseded)

```python
"""
Always-valid sequential A/B test: mixture SPRT (Robbins 1970; Johari, Pekelis, Walsh 2017).
numpy + scipy only. Single file: streaming test, naive comparator, null generators,
equivalence check, 1,000-stream null evaluation, horizon check.

Model: arm A vs arm B, H0: mu_B - mu_A = 0, mixture theta ~ N(0, tau^2).
    d = mean_B - mean_A,  V = var_A/n_A + var_B/n_B
    Lambda = sqrt(V/(V+tau^2)) * exp(d^2 tau^2 / (2 V (V+tau^2)))
Under H0 (Gaussian, known variance) Lambda is a nonnegative mean-1 martingale, so by
Ville's inequality P(sup_n Lambda_n >= 1/alpha) <= alpha at any stopping time.
"""
import sys
import numpy as np
from scipy import stats


# --------------------------------------------------------------------------
# LOG STRUCTURE: stand-in. Swap the dict keys for your schema; nothing else
# depends on these names.
# --------------------------------------------------------------------------
def _log_record(look, n_a, n_b, d, se, lam, p_av, decision):
    return {
        "look": look,
        "n_a": n_a,
        "n_b": n_b,
        "effect_hat": d,
        "se": se,
        "lambda": lam,
        "p_always_valid": p_av,
        "decision": decision,   # "continue" | "reject_h0"
    }


# --------------------------------------------------------------------------
# Streaming mSPRT: one log record per look
# --------------------------------------------------------------------------
def msprt_stream(stream, alpha=0.05, tau=0.1, sigma=None, look_every=1,
                 burn_in=20, stop_on_reject=True):
    """
    stream     : iterable of (arm, y), arm in {0 (A), 1 (B)}
    alpha      : Type I bound over the whole (unbounded) monitoring period
    tau        : sd of the N(0, tau^2) mixture, in units of y; affects power, not validity
    sigma      : known per-observation sd (float or (sd_A, sd_B)); None = plug-in
                 running variance (approximately valid only)
    look_every : evaluate every k observations (1 = continuous)
    burn_in    : min obs per arm before first look (plug-in case only)
    """
    n = np.zeros(2); s = np.zeros(2); ss = np.zeros(2)
    if sigma is not None:
        sig2 = np.broadcast_to(np.asarray(sigma, float) ** 2, (2,))
    tau2 = tau ** 2
    p_av = 1.0
    look = 0
    log_thresh = np.log(1.0 / alpha)
    min_n = 1 if sigma is not None else max(burn_in, 2)

    for t, (arm, y) in enumerate(stream, start=1):
        n[arm] += 1; s[arm] += y; ss[arm] += y * y
        if t % look_every or n.min() < min_n:
            continue
        mean = s / n
        var = sig2 if sigma is not None else np.maximum((ss - n * mean ** 2) / (n - 1), 1e-12)
        V = var[0] / n[0] + var[1] / n[1]
        d = mean[1] - mean[0]
        log_lam = 0.5 * np.log(V / (V + tau2)) + d * d * tau2 / (2 * V * (V + tau2))
        lam = np.exp(min(log_lam, 700.0))
        p_av = min(p_av, 1.0 / lam)            # monotone always-valid p-value
        reject = log_lam >= log_thresh
        look += 1
        yield _log_record(look, int(n[0]), int(n[1]), d, np.sqrt(V), lam, p_av,
                          "reject_h0" if reject else "continue")
        if reject and stop_on_reject:
            return


# --------------------------------------------------------------------------
# Comparator: fixed-n two-sided z-test re-run at every look (the wrong way)
# --------------------------------------------------------------------------
def naive_stream(stream, alpha=0.05, look_every=1, burn_in=20):
    n = np.zeros(2); s = np.zeros(2); ss = np.zeros(2)
    zcrit = stats.norm.ppf(1 - alpha / 2)
    look = 0
    for t, (arm, y) in enumerate(stream, start=1):
        n[arm] += 1; s[arm] += y; ss[arm] += y * y
        if t % look_every or n.min() < burn_in:
            continue
        mean = s / n
        var = np.maximum((ss - n * mean ** 2) / (n - 1), 1e-12)
        se = np.sqrt(var[0] / n[0] + var[1] / n[1])
        z = (mean[1] - mean[0]) / se
        look += 1
        rej = abs(z) >= zcrit
        yield _log_record(look, int(n[0]), int(n[1]), mean[1] - mean[0], se,
                          np.nan, 2 * stats.norm.sf(abs(z)),
                          "reject_h0" if rej else "continue")
        if rej:
            return


# --------------------------------------------------------------------------
# Null-stream generators + helper
# --------------------------------------------------------------------------
def null_gaussian(rng, n_total, mu=0.0, sd=1.0):
    return zip(rng.integers(0, 2, n_total), rng.normal(mu, sd, n_total))


def null_bernoulli(rng, n_total, p=0.10):
    return zip(rng.integers(0, 2, n_total), (rng.random(n_total) < p).astype(float))


def ever_rejects(test_iter):
    last = None
    for last in test_iter:
        pass
    return last is not None and last["decision"] == "reject_h0"


# --------------------------------------------------------------------------
# Vectorized evaluator: same statistic at every observation from cumulative sums
# (the pure-Python stream is too slow for 1,000 x 20k observations)
# --------------------------------------------------------------------------
def stats_path(arms, ys, sigma=None, burn_in=20):
    a = arms.astype(bool)
    nB = np.cumsum(a); nA = np.arange(1, len(a) + 1) - nB
    sB = np.cumsum(ys * a); sA = np.cumsum(ys * ~a)
    qB = np.cumsum(ys * ys * a); qA = np.cumsum(ys * ys * ~a)
    with np.errstate(divide="ignore", invalid="ignore"):
        mA, mB = sA / nA, sB / nB
        if sigma is None:
            vA = np.maximum((qA - nA * mA ** 2) / (nA - 1), 1e-12)
            vB = np.maximum((qB - nB * mB ** 2) / (nB - 1), 1e-12)
            ok = np.minimum(nA, nB) >= max(burn_in, 2)
        else:
            vA = vB = sigma ** 2
            ok = np.minimum(nA, nB) >= 1
        V = vA / nA + vB / nB
    return mB - mA, V, ok


def msprt_rejects(arms, ys, alpha, tau, sigma=None, burn_in=20, look_every=1):
    d, V, ok = stats_path(arms, ys, sigma, burn_in)
    ok &= (np.arange(1, len(arms) + 1) % look_every == 0)
    t2 = tau ** 2
    loglam = 0.5 * np.log(V[ok] / (V[ok] + t2)) + d[ok] ** 2 * t2 / (2 * V[ok] * (V[ok] + t2))
    return bool((loglam >= np.log(1 / alpha)).any()), loglam


def naive_rejects(arms, ys, alpha, look_every, burn_in=20):
    d, V, ok = stats_path(arms, ys, None, burn_in)
    ok &= (np.arange(1, len(arms) + 1) % look_every == 0)
    z = d[ok] / np.sqrt(V[ok])
    return bool((np.abs(z) >= stats.norm.ppf(1 - alpha / 2)).any())


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------
def smoke_test(n_streams=100, n_total=5000):
    rng = np.random.default_rng(0)
    k = sum(ever_rejects(msprt_stream(null_gaussian(rng, n_total), tau=0.05, sigma=1.0))
            for _ in range(n_streams))
    print(f"smoke test (streaming fn): {k}/{n_streams} null streams rejected")


def equivalence_check():
    rng = np.random.default_rng(1)
    maxdiff = 0.0
    for _ in range(5):
        arms = rng.integers(0, 2, 3000); ys = rng.normal(0, 1, 3000)
        recs = list(msprt_stream(zip(arms, ys), tau=0.05, stop_on_reject=False))
        _, ll = msprt_rejects(arms, ys, 0.05, 0.05)
        maxdiff = max(maxdiff, np.max(np.abs(np.log([r["lambda"] for r in recs]) - ll)))
    print(f"streaming vs vectorized max |log Lambda diff|: {maxdiff:.2e}")


def null_evaluation(N=1000, T=20_000, alpha=0.05):
    def run(label, gen, test):
        rng = np.random.default_rng(20261027)
        k = 0
        for _ in range(N):
            arms = rng.integers(0, 2, T)
            k += test(arms, gen(rng))
        lo, hi = stats.beta.ppf([0.025, 0.975], k + 0.5, N - k + 0.5)  # Jeffreys CI
        print(f"{label:<52} FPR={k / N:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
        sys.stdout.flush()

    G = lambda r: r.normal(0, 1, T)
    B = lambda r: (r.random(T) < 0.10).astype(float)
    run("mSPRT  Gaussian  known sigma  every obs", G, lambda a, y: msprt_rejects(a, y, alpha, 0.05, sigma=1.0)[0])
    run("mSPRT  Gaussian  plug-in var  every obs", G, lambda a, y: msprt_rejects(a, y, alpha, 0.05)[0])
    run("mSPRT  Bernoulli .10 plug-in  every obs", B, lambda a, y: msprt_rejects(a, y, alpha, 0.01)[0])
    run("mSPRT  Bernoulli .10 plug-in  burn_in=2 (bad)", B, lambda a, y: msprt_rejects(a, y, alpha, 0.01, burn_in=2)[0])
    run("mSPRT  Gaussian  known sigma  tau=0.5", G, lambda a, y: msprt_rejects(a, y, alpha, 0.5, sigma=1.0)[0])
    run("Naive z  Gaussian  10 looks", G, lambda a, y: naive_rejects(a, y, alpha, T // 10))
    run("Naive z  Gaussian  200 looks", G, lambda a, y: naive_rejects(a, y, alpha, T // 200))
    run("Naive z  Gaussian  every obs", G, lambda a, y: naive_rejects(a, y, alpha, 1))


def horizon_check(N=1000, steps=100_000, alpha=0.05, u_ends=(1.25, 12.5, 125, 1250)):
    """Null z-path in natural time u = tau^2/V. The 20k Gaussian run ends at u = 12.5."""
    rng = np.random.default_rng(7)
    for u_end in u_ends:
        c = u_end / steps; k = 0
        for _ in range(N // 100):
            Bm = np.cumsum(rng.standard_normal((100, steps)), axis=1)
            n = np.arange(1, steps + 1); u = c * n; z2 = Bm ** 2 / n
            loglam = -0.5 * np.log1p(u) + 0.5 * z2 * u / (1 + u)
            k += (loglam >= np.log(1 / alpha)).any(axis=1).sum()
        print(f"horizon u_end={u_end:>7}: FPR={k / N:.3f}")


if __name__ == "__main__":
    quick = "--quick" in sys.argv          # syntax/plumbing check with small sizes
    smoke_test(n_streams=20 if quick else 100, n_total=2000 if quick else 5000)
    equivalence_check()
    if quick:
        null_evaluation(N=50, T=2000)
        horizon_check(N=100, steps=5000)
    else:
        null_evaluation()                  # ~3-4 min
        horizon_check()                    # ~3-4 min
```

---

## Turn 3

### User prompt (verbatim)

```
Your initial implementation of the mixture sequential probability ratio test (mSPRT) is a useful starting point, but I identified several issues that need to be corrected to make it appropriate for my ECON 5200 experiment portfolio.
My existing portfolio consists of 200 simulated two-arm A/B tests with the following characteristics:

* Baseline conversion rate: 10%.
* True effects: 80% of experiments have no effect; 20% have positive effects uniformly distributed between 1 and 3.5 percentage points.
* Sample sizes: 200–4,000 observations per treatment group.
* Outcome: Binary conversion (0 or 1).
* Significance level: 5%.

Please revise your original implementation to address these specific issues:

1. Statistical validity: Your original mSPRT relies on Gaussian observations with known variance. For Bernoulli conversion data, your implementation uses plug-in variance estimates, which do not guarantee exact Type I error control. Replace this with a sequential testing procedure that provides a mathematically valid Type I error guarantee for two independent Bernoulli samples. You may use an appropriate e-process, mixture likelihood-ratio construction, or alpha-spending procedure with valid per-look tests. Explain why the procedure remains valid under repeated monitoring.
2. Consistency with my portfolio: Use two-arm A/B experiments with binary outcomes, a baseline conversion probability of 0.10, and a maximum of 4,000 observations per arm. Specify the monitoring schedule clearly.
3. Reusable function: Provide a function accepting treatment and control observation streams and returning the decision at every monitoring look, including whether to continue or reject the null hypothesis and the first stopping time. The function must not use the known true effect to make decisions.
4. Independent verification: Simulate 1,000 pure-null A/B experiments where both treatment and control groups have a conversion probability of 0.10. Report:
   * Number of false positives.
   * Empirical false-positive rate.
   * Monte Carlo standard error, calculated as sqrt(p_hat * (1 - p_hat) / 1000).
   * Comparison against the nominal 5% significance level.
5. Comparison: Demonstrate why repeatedly applying an ordinary p < 0.05 test at every look is invalid, preferably by comparing its empirical false-positive rate with the corrected sequential procedure.
6. Explain the trade-off: Identify what the corrected procedure sacrifices in exchange for valid continuous monitoring, particularly regarding statistical power and sample size.
7. Code requirements: Use only NumPy and SciPy. Provide complete, executable Python code suitable for one Google Colab cell. Include assertions or basic checks to verify the implementation.
8. Interpretation: After the code, briefly explain which lines determine when an experiment stops, whether the measured false-positive rate supports the validity of the procedure, and any limitations that remain.

Keep the implementation focused on these requirements.
```

### AI response (substance)

The AI acknowledged that the user's criticism was correct: the Turn 1 plug-in variance made the Bernoulli case only approximately valid, and the AI had treated that as a footnote rather than fixing it.

**Procedure: paired discordant-pair Beta-mixture e-process (sequential sign test)**

1. Pair the i-th treatment observation with the i-th control observation.
2. Keep only discordant pairs (exactly one unit converted).
3. Under H0: p_T = p_C = p, P(treatment is the converter | discordant) = p(1−p) / (2p(1−p)) = ½ for any p, so the nuisance parameter cancels.
4. Mix over θ ~ Beta(a, a): Eₙ = 2^m · B(a+k, a+m−k) / B(a, a), where m = discordant pairs and k = treatment wins.
5. Reject when Eₙ ≥ 1/α = 20.

**Validity under repeated monitoring:** for each fixed θ, the running product of likelihood-ratio factors is a nonnegative mean-1 martingale under H0; the Beta mixture of these is still a nonnegative mean-1 martingale with E₀ = 1. Ville's inequality gives P(Eₙ ≥ 1/α at any n) ≤ α for every stopping rule. The guarantee is exact, needs no variance estimate, and holds for any common baseline, including one that drifts over time, provided paired units share the same p.

**Monitoring schedule:** one look after every completed pair, n = 1 to 4,000 per arm (4,000 looks), stopping at the first rejection.

**Prior width:** a = 47.5, fixed before data, set so the prior's standard deviation equals the θ shift implied by a 2pp effect at a 10% baseline. No experiment's true effect is used in any decision.

**Verification checks (all passed):**
- E[E₆₀] = 1 under H0 by exact enumeration for p = 0.02, 0.10, 0.50.
- Exact Type I error by dynamic programming at p = 0.10, every-pair looks, 4,000 per arm: **0.0172**.
- `sequential_ab_test()` matches the vectorized core's stopping time on 20 streams.

**1,000 null experiments (p_T = p_C = 0.10)**

| Procedure | False positives | FPR | MCSE | (FPR − 0.05) / MCSE |
|---|---|---|---|---|
| Sequential e-process, every pair (4,000 looks) | 18 | 0.018 | 0.0042 | −7.6 |
| Naive p < 0.05, every pair (4,000 looks) | 519 | 0.519 | 0.0158 | +29.7 |
| Naive p < 0.05, every 100 per arm (40 looks) | 304 | 0.304 | 0.0145 | +17.5 |
| Single fixed-n z-test at 4,000 per arm | 52 | 0.052 | 0.0070 | +0.3 |

**Power at a maximum of 4,000 per arm (1,000 experiments per effect)**

| Effect | Sequential power | Fixed-n power | Median stop n, given rejection | Mean n used |
|---|---|---|---|---|
| 1.0pp | 0.125 | 0.319 | 2,337 | 3,785 |
| 2.0pp | 0.580 | 0.810 | 2,262 | 3,000 |
| 3.5pp | 0.981 | 0.999 | 1,395 | 1,616 |

Rejection boundary at 4,000 per arm (~720 discordant pairs): |z| ≥ 3.06, vs 1.96 for a single fixed-n test.

**Portfolio stand-in** (simulated from the stated parameters, not the user's actual 200 tests; 41 true effects, 159 nulls)

| Procedure | Discoveries | False | True | Power |
|---|---|---|---|---|
| Sequential | 21 | 3 | 18 | 0.439 |
| Fixed-n z-test at each test's n | 37 | 8 | 29 | 0.707 |

**Why repeated p < 0.05 testing is invalid:** the false-positive event is the union across many correlated looks; the null z-statistic behaves like a random walk and, by the law of the iterated logarithm, eventually crosses any fixed bound. Observed FPR rises from 5.2% (1 look) to 30.4% (40 looks) to 51.9% (4,000 looks).

**Trade-offs identified:**
1. Higher bar at any sample size: |z| ≥ 3.06 vs 1.96 at 4,000 per arm, costing power (12.5% / 58% / 98% vs 32% / 81% / 100% at 1 / 2 / 3.5pp). This gap combines the boundary cost with any information lost through pairing; the two were not measured separately.
2. Unspent α: on the capped horizon only 1.72% of the 5% is spent. A pre-committed alpha-spending design (e.g., O'Brien-Fleming) would spend the full 5% but loses validity if the experiment is extended past plan.
3. Benefit: large effects stop early (mean 1,616 per arm at 3.5pp), and stopping at any look carries no extra penalty.
4. Power, but not validity, depends on the prior width a, tuned to 2pp.

**Interpretation given:**
- Stopping is decided by `reject = log_e[i] >= thr` (with `thr = log(1/alpha)`) and the following `break` in `sequential_ab_test`; in the simulations by `(le0 >= THR).any(axis=1)`.
- The observed 1.8% (MCSE 0.42%) matches the exact 1.72% and is well below 5%; being 7.6 MCSEs under 5% is expected because Ville's inequality is an upper bound and a capped horizon leaves α unspent.
- Remaining limitations: pairing by arrival index assumes 1:1 allocation and contemporaneous pairs; the test is two-sided although true effects are positive; it controls per-test error, not portfolio FDR (e-BH on the e-values would address that); estimates at stopping are biased upward; units must be independent.

### Code delivered (Turn 3, final version)

```python
# =====================================================================================
# ECON 5200 — Always-valid sequential test for two-arm Bernoulli A/B experiments
# numpy + scipy only. One Colab cell.
#
# PROCEDURE: paired discordant-pair Beta-mixture e-process (sequential sign test).
#   Pair the i-th treatment obs with the i-th control obs. A pair is "discordant" if
#   exactly one converted. Under H0: p_T = p_C = p (ANY p, unknown, even time-varying),
#       P(treatment is the converter | pair is discordant) = p(1-p) / (2p(1-p)) = 1/2,
#   independent of the past. So the discordant pairs form a fair-coin sequence under H0,
#   with no nuisance parameter left to plug in.
#   For each theta, prod_i [2 theta]^W_i [2(1-theta)]^(1-W_i) over discordant pairs is a
#   nonnegative martingale with mean 1 under H0 (concordant pairs contribute a factor 1).
#   Mixing over theta ~ Beta(a, a) keeps it a nonnegative mean-1 martingale:
#       E_n = 2^m * B(a + k, a + m - k) / B(a, a),   m = #discordant, k = #treatment wins
#   Ville's inequality: P_H0( exists n : E_n >= 1/alpha ) <= alpha, exactly, for every
#   common p and every stopping rule, including a look after every pair, forever.
# =====================================================================================
import numpy as np
from scipy import stats
from scipy.special import betaln

ALPHA      = 0.05
P_BASE     = 0.10      # used ONLY to simulate data and to set the design prior below
N_MAX      = 4000      # max observations per arm
LOOK_EVERY = 1         # monitoring schedule: a look after every completed pair, n = 1..4000
N_SIM      = 1000
rng        = np.random.default_rng(5200)

# ---- Design prior (a design choice, fixed before data; never uses any test's true effect)
# Target effect 2pp at a 10% baseline -> theta0 = P(T wins | discordant) under that effect.
def theta_of(pc, pt):
    return pt * (1 - pc) / (pt * (1 - pc) + pc * (1 - pt))
DESIGN_DELTA = 0.02
_theta0 = theta_of(P_BASE, P_BASE + DESIGN_DELTA)
_sd = abs(_theta0 - 0.5)                          # Beta(a,a) sd  = 1 / (2 sqrt(2a+1))
A_DESIGN = (1 / (4 * _sd ** 2) - 1) / 2


# =====================================================================================
# Core: log e-process path. Works on 1-D streams or 2-D (experiments x n) matrices.
# =====================================================================================
def log_e_path(treat, ctrl, a=A_DESIGN):
    treat = np.asarray(treat, dtype=np.int8); ctrl = np.asarray(ctrl, dtype=np.int8)
    disc = treat != ctrl                                  # discordant pair indicator
    m = np.cumsum(disc, axis=-1)                          # discordant pairs so far
    k = np.cumsum(disc & (treat == 1), axis=-1)           # of which treatment converted
    return m * np.log(2.0) + betaln(a + k, a + m - k) - betaln(a, a), m, k


# =====================================================================================
# Reusable function: decision at every monitoring look + first stopping time
# =====================================================================================
def sequential_ab_test(treatment, control, alpha=ALPHA, a=A_DESIGN,
                       n_max=N_MAX, look_every=LOOK_EVERY):
    """
    treatment, control : 0/1 sequences in arrival order. Pairs are formed by index;
                         unpaired surplus in one arm waits for its partner.
    Returns dict: looks (one record per look up to stopping), decision, stop_n
                  (obs per arm at first rejection, None if never), p_always_valid.
    Uses only the observed data. No true effect enters anywhere.
    """
    n = min(len(treatment), len(control), n_max)
    log_e, m, k = log_e_path(np.asarray(treatment)[:n], np.asarray(control)[:n], a)
    thr = np.log(1.0 / alpha)
    looks, p_av, stop_n = [], 1.0, None
    for i in range(look_every - 1, n, look_every):        # look after pair i+1
        p_av = min(p_av, float(np.exp(-log_e[i])))          # always-valid p = 1/max E
        reject = log_e[i] >= thr                            # <-- STOPPING RULE
        final = (i + look_every >= n)
        decision = "reject_h0" if reject else ("stop_max_n" if final else "continue")
        looks.append({"look": len(looks) + 1, "n_per_arm": i + 1, "discordant": int(m[i]),
                      "treat_wins": int(k[i]), "e_value": float(np.exp(log_e[i])),
                      "p_always_valid": min(p_av, 1.0), "decision": decision})
        if reject:
            stop_n = i + 1
            break
    return {"looks": looks, "decision": looks[-1]["decision"] if looks else "no_data",
            "stop_n": stop_n, "p_always_valid": min(p_av, 1.0)}


# =====================================================================================
# Comparator: ordinary two-sided pooled two-proportion z-test, p < 0.05, at each look
# =====================================================================================
def naive_z_path(treat, ctrl):
    n = np.arange(1, treat.shape[-1] + 1)
    st, sc = np.cumsum(treat, axis=-1), np.cumsum(ctrl, axis=-1)
    pt, pc = st / n, sc / n
    pp = (st + sc) / (2 * n)
    se = np.sqrt(pp * (1 - pp) * 2 / n)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = np.where(se > 0, (pt - pc) / se, 0.0)
    return z


def simulate(n_exp, pc, pt, n=N_MAX):
    return ((rng.random((n_exp, n)) < pt).astype(np.int8),
            (rng.random((n_exp, n)) < pc).astype(np.int8))


# =====================================================================================
# Checks
# =====================================================================================
THR = np.log(1 / ALPHA)

# (1) E_0 = 1 and exact martingale property: E[E_n] = 1 under H0, by full enumeration
#     over (m discordant, k wins) at n = 60 pairs, for several baselines.
for p in (0.02, 0.10, 0.50):
    nn = 60; q = 2 * p * (1 - p)
    tot = 0.0
    for mm in range(nn + 1):
        kk = np.arange(mm + 1)
        e = np.exp(mm * np.log(2) + betaln(A_DESIGN + kk, A_DESIGN + mm - kk) - betaln(A_DESIGN, A_DESIGN))
        tot += stats.binom.pmf(mm, nn, q) * np.sum(stats.binom.pmf(kk, mm, 0.5) * e)
    assert abs(tot - 1) < 1e-10, (p, tot)
print(f"[check] E[E_60] = 1 under H0 for p in (0.02, 0.10, 0.50)  (a = {A_DESIGN:.1f})")

# (2) Exact Type I error of the full procedure at n_max = 4000, every-pair looks,
#     by dynamic programming over the discordant-pair walk (no simulation).
def exact_type1(p, n_max=N_MAX, a=A_DESIGN, alpha=ALPHA):
    q = 2 * p * (1 - p)
    M_hi = int(stats.binom.ppf(1 - 1e-14, n_max, q)) + 1
    dist = np.zeros(M_hi + 2); dist[0] = 1.0                  # unabsorbed mass over k
    H = np.zeros(M_hi + 1)                                    # P(hit by m discordant)
    for mm in range(1, M_hi + 1):
        new = 0.5 * dist; new[1:] += 0.5 * dist[:-1]
        kk = np.arange(len(new))
        valid = kk <= mm
        le = np.full(len(new), -np.inf)
        le[valid] = mm * np.log(2) + betaln(a + kk[valid], a + mm - kk[valid]) - betaln(a, a)
        hit = valid & (le >= np.log(1 / alpha))
        H[mm] = H[mm - 1] + new[hit].sum()
        new[hit] = 0.0
        dist = new
    Ms = np.arange(M_hi + 1)
    return float(np.sum(stats.binom.pmf(Ms, n_max, q) * H))

exact_null = exact_type1(P_BASE)
assert exact_null <= ALPHA
print(f"[check] exact Type I at p=0.10, n_max=4000, every-pair looks: {exact_null:.4f} <= {ALPHA}")

# (3) Reusable function agrees with the vectorized core on a few streams
t_chk, c_chk = simulate(20, P_BASE, P_BASE)
le_chk, _, _ = log_e_path(t_chk, c_chk)
for j in range(20):
    res = sequential_ab_test(t_chk[j], c_chk[j])
    hits = np.flatnonzero(le_chk[j] >= THR)
    assert res["stop_n"] == (hits[0] + 1 if hits.size else None)
    assert all(r["decision"] == "continue" for r in res["looks"][:-1])
print("[check] sequential_ab_test() matches vectorized core on 20 streams")


# =====================================================================================
# 4) 1,000 pure-null experiments, p_T = p_C = 0.10, look after every pair up to 4,000/arm
# =====================================================================================
t0, c0 = simulate(N_SIM, P_BASE, P_BASE)
le0, _, _ = log_e_path(t0, c0)
rej_seq = (le0 >= THR).any(axis=1)

z0 = naive_z_path(t0, c0)
zc = stats.norm.ppf(1 - ALPHA / 2)
rej_naive_every = (np.abs(z0) >= zc).any(axis=1)
rej_naive_40    = (np.abs(z0[:, 99::100]) >= zc).any(axis=1)     # every 100/arm, 40 looks
rej_fixed       = np.abs(z0[:, -1]) >= zc                         # single look at 4,000

def report(label, rej):
    k = int(rej.sum()); ph = k / N_SIM; se = np.sqrt(ph * (1 - ph) / N_SIM)
    print(f"{label:<44} FP={k:>4}  FPR={ph:.3f}  MCSE={se:.4f}  "
          f"(FPR-0.05)/MCSE={(ph - ALPHA) / se if se > 0 else float('nan'):+.1f}")
    return ph, se

print("\n=== 1,000 null experiments (p_T = p_C = 0.10, up to 4,000/arm) ===")
fpr_seq, se_seq = report("Sequential e-process, every pair (4,000 looks)", rej_seq)
report("Naive p<0.05, every pair (4,000 looks)", rej_naive_every)
report("Naive p<0.05, every 100/arm (40 looks)", rej_naive_40)
report("Single fixed-n z-test at 4,000/arm", rej_fixed)
assert fpr_seq <= ALPHA + 2 * se_seq + 1e-12, "sequential FPR exceeds 5% beyond MC noise"
assert rej_naive_every.mean() > 2 * ALPHA, "naive peeking should be clearly inflated"


# =====================================================================================
# 6) Trade-off: power and sample size vs a fixed-n test at 4,000/arm
# =====================================================================================
print("\n=== Power at max 4,000/arm (1,000 experiments per effect) ===")
print(f"{'effect':>7} {'seq power':>10} {'fixed-n power':>14} {'median stop n|reject':>21} "
      f"{'mean n used':>12}")
for d in (0.01, 0.02, 0.035):
    t1, c1 = simulate(N_SIM, P_BASE, P_BASE + d)
    le1, _, _ = log_e_path(t1, c1)
    hit = le1 >= THR
    rej = hit.any(axis=1)
    stop = np.where(rej, hit.argmax(axis=1) + 1, N_MAX)
    fixed = np.abs(naive_z_path(t1, c1)[:, -1]) >= zc
    med = np.median(stop[rej]) if rej.any() else float("nan")
    print(f"{d*100:>5.1f}pp {rej.mean():>10.3f} {fixed.mean():>14.3f} {med:>21.0f} {stop.mean():>12.0f}")

# Boundary in z-units at the end of the horizon (expected null discordant count)
m_end = 2 * P_BASE * (1 - P_BASE) * N_MAX
ks = np.arange(int(m_end) + 1)
lek = m_end * np.log(2) + betaln(A_DESIGN + ks, A_DESIGN + m_end - ks) - betaln(A_DESIGN, A_DESIGN)
k_crit = ks[(lek >= THR) & (ks > m_end / 2)].min()
print(f"\nBoundary at 4,000/arm (~{m_end:.0f} discordant pairs): |z| >= "
      f"{(k_crit - m_end/2) / (np.sqrt(m_end)/2):.2f}  vs 1.96 for a single fixed-n test")


# =====================================================================================
# Stand-in for the portfolio: 200 tests, 80% null, 20% effects ~ U(1, 3.5)pp,
# n ~ U{200..4000}/arm. (Simulated with your stated parameters; not your actual data.)
# =====================================================================================
n_tests = 200
is_eff = rng.random(n_tests) < 0.20
delta = np.where(is_eff, rng.uniform(0.01, 0.035, n_tests), 0.0)
n_arm = rng.integers(200, 4001, n_tests)
seq_rej = np.zeros(n_tests, bool); fix_rej = np.zeros(n_tests, bool)
for j in range(n_tests):
    tj = (rng.random(n_arm[j]) < P_BASE + delta[j]).astype(np.int8)
    cj = (rng.random(n_arm[j]) < P_BASE).astype(np.int8)
    seq_rej[j] = sequential_ab_test(tj, cj)["decision"] == "reject_h0"   # no delta used
    fix_rej[j] = abs(naive_z_path(tj, cj)[-1]) >= zc
print("\n=== Portfolio stand-in (200 tests) ===")
print(f"true effects: {is_eff.sum()}, nulls: {(~is_eff).sum()}")
for lbl, r in (("sequential", seq_rej), ("fixed-n z at each test's n", fix_rej)):
    print(f"{lbl:<28} discoveries={r.sum():>3}  false={np.sum(r & ~is_eff):>3}  "
          f"true={np.sum(r & is_eff):>3}  power={np.mean(r[is_eff]):.3f}")
```

---

## Turn 4

### User prompt (verbatim)

```
can you create an ai-appendix.md file of this entire conversation
```

### AI response

Produced this file.

---

## Notes on AI use and verification

- All figures above were produced by executing the code in the AI's sandbox; none were estimated or typed from memory. Re-running the final code in Colab with the same seed (`5200`) should reproduce the Turn 3 tables, up to any platform differences in NumPy's random generator.
- The user's actual portfolio data and log structure were never provided to the AI. Portfolio results are from a simulated stand-in using the user's stated parameters.
- The AI's initial approach (Turn 1) had a validity gap for Bernoulli data that the user identified; the final procedure (Turn 3) resolves it with an exact guarantee and an exact Type I calculation.
