import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Notebook illustration Monte Carlo approximation to the greeks

    In this notebook we consider a simple situation: we use the standard Black-Scholes market and consider the vega of a European call option. Of course, we have a closed-form formula available for this vega. We will approximate the vega using Monte Carlo methods and compare against the closed-form formula.

    *This is the interactive [marimo](https://marimo.io) version of
    [illustration_mc_methods_for_greeks.ipynb](https://github.com/ramonVDAKKER/teaching-quantitative-finance/blob/main/notebooks/illustration_mc_methods_for_greeks.ipynb);
    it runs entirely in your browser. Use the sliders to explore the effect of the step size $\delta$ and the number of replications.*
    """)
    return


@app.cell
def _():
    import numpy as np
    from scipy.stats import norm
    return norm, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Example

    Parameters: money-market account with $r=1\%$; stock with $S_0=100$, $\sigma=20\%$; European call with $T=1$, $K=100$.

    Vega, at $t=0$, using the closed-form formula $\nu = S_0\,\phi(d_1)\sqrt{T}$ (with $\phi$ the standard normal pdf):
    """)
    return


@app.cell
def _(mo, norm, np):
    r = 0.01
    S_0 = 100
    sigma = 0.2
    T = 1
    K = 100


    def vega_call_exact(current_stock_price, time_to_maturity, K, r, sigma):
        """Closed-form Black-Scholes vega of a European call (equals the vega of the put)."""

        d1 = (np.log(current_stock_price / K) + (r + 0.5 * sigma ** 2) * time_to_maturity) / (sigma * np.sqrt(time_to_maturity))
        return current_stock_price * norm.pdf(d1) * np.sqrt(time_to_maturity)


    vega_exact = vega_call_exact(S_0, T, K, r, sigma)
    mo.md(f"Exact vega: **{vega_exact:.4f}**")
    return K, S_0, T, r, sigma, vega_exact


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.1-1.4 The four estimators

    We estimate the vega by (i) bump-and-reprice with *non-common* random numbers, (ii) bump-and-reprice with *common* random numbers, (iii) the *pathwise* method, and (iv) the *likelihood ratio* method. Move the sliders and observe:

    * with **non-common** random numbers the bump estimate becomes very noisy for small $\delta$ (variance $O(1/(n\delta^2))$);
    * with **common** random numbers small $\delta$ is harmless;
    * pathwise and LRM do not require a step $\delta$ at all and are unbiased.

    Every move of a slider re-runs the simulation with fresh random draws.
    """)
    return


@app.cell
def _(mo):
    mc_delta = mo.ui.slider(0.001, 0.5, step=0.001, value=0.01, label=r"step $\delta$")
    mc_R = mo.ui.slider(10_000, 300_000, step=10_000, value=100_000, label="replications $n$")
    mo.hstack([mc_delta, mc_R], justify="start", gap=2)
    return mc_R, mc_delta


@app.cell
def _(K, S_0, T, mc_R, mc_delta, mo, norm, np, r, sigma, vega_exact):
    _n = mc_R.value
    _d = mc_delta.value


    def _price_prox(sig, W):
        _ST = S_0 * np.exp((r - 0.5 * sig ** 2) * T + sig * W)
        return np.exp(-r * T) * np.mean(np.maximum(_ST - K, 0))


    # (i) bump and reprice, non-common random numbers:
    _W1 = np.sqrt(T) * norm.rvs(size=_n)
    _W2 = np.sqrt(T) * norm.rvs(size=_n)
    vega_bump_noncommon = (_price_prox(sigma + _d, _W2) - _price_prox(sigma, _W1)) / _d

    # (ii) bump and reprice, common random numbers:
    _W = np.sqrt(T) * norm.rvs(size=_n)
    vega_bump_common = (_price_prox(sigma + _d, _W) - _price_prox(sigma, _W)) / _d

    # (iii) pathwise:
    _Wp = np.sqrt(T) * norm.rvs(size=_n)
    _STp = S_0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * _Wp)
    vega_pathwise = np.exp(-r * T) * np.mean((_STp > K) * _STp * (_Wp - sigma * T))

    # (iv) likelihood ratio; score (d/d sigma) log g(S_T; sigma) = (W^2/T - 1)/sigma - W:
    _Wl = np.sqrt(T) * norm.rvs(size=_n)
    _STl = S_0 * np.exp((r - 0.5 * sigma ** 2) * T + sigma * _Wl)
    vega_lrm = np.exp(-r * T) * np.mean(np.maximum(_STl - K, 0) * ((_Wl ** 2 / T - 1) / sigma - _Wl))

    mo.md(
        f"""
    | method | estimate | error vs exact ({vega_exact:.4f}) |
    |---|---|---|
    | bump and reprice, non-common random numbers | {vega_bump_noncommon:.4f} | {vega_bump_noncommon - vega_exact:+.4f} |
    | bump and reprice, common random numbers | {vega_bump_common:.4f} | {vega_bump_common - vega_exact:+.4f} |
    | pathwise | {vega_pathwise:.4f} | {vega_pathwise - vega_exact:+.4f} |
    | likelihood ratio | {vega_lrm:.4f} | {vega_lrm - vega_exact:+.4f} |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The three estimators on the same example

    We estimate the **delta** of (a) a European call and (b) a digital call by bump-and-reprice, pathwise and likelihood-ratio, and compare bias and variance - reproducing the summary table in the slides.
    """)
    return


@app.cell
def _(mo, np):
    from math import erf, log, sqrt, exp, pi

    def Phi(x):
        return 0.5 * (1 + erf(x / sqrt(2)))

    s2_rng = np.random.default_rng(0)
    s2_S0, s2_K, s2_r, s2_sigma, s2_T = 100.0, 100.0, 0.03, 0.20, 1.0
    s2_R = 200_000
    s2_Z = s2_rng.standard_normal(s2_R)
    s2_ST = s2_S0 * np.exp((s2_r - 0.5 * s2_sigma ** 2) * s2_T + s2_sigma * sqrt(s2_T) * s2_Z)
    s2_disc = exp(-s2_r * s2_T)

    # ---- delta of a European CALL ----
    s2_d1 = (log(s2_S0 / s2_K) + (s2_r + 0.5 * s2_sigma ** 2) * s2_T) / (s2_sigma * sqrt(s2_T))
    call_pw = s2_disc * (s2_ST > s2_K) * s2_ST / s2_S0                                  # pathwise
    call_lr = s2_disc * np.maximum(s2_ST - s2_K, 0) * s2_Z / (s2_sigma * sqrt(s2_T) * s2_S0)  # likelihood ratio
    s2_STd = (s2_S0 + 1.0) * np.exp((s2_r - 0.5 * s2_sigma ** 2) * s2_T + s2_sigma * sqrt(s2_T) * s2_Z)  # CRN, step delta=1
    call_bp = s2_disc * (np.maximum(s2_STd - s2_K, 0) - np.maximum(s2_ST - s2_K, 0)) / 1.0    # bump & reprice

    # ---- delta of a DIGITAL call, payoff 1{ST>K} ----
    s2_d2 = (log(s2_S0 / s2_K) + (s2_r - 0.5 * s2_sigma ** 2) * s2_T) / (s2_sigma * sqrt(s2_T))
    exact_dig = s2_disc * exp(-0.5 * s2_d2 ** 2) / sqrt(2 * pi) / (s2_S0 * s2_sigma * sqrt(s2_T))
    dig_lr = s2_disc * (s2_ST > s2_K) * s2_Z / (s2_sigma * sqrt(s2_T) * s2_S0)          # LRM still works

    mo.md(
        f"""
    **Delta of a European call** (exact $\\Phi(d_1)$ = {Phi(s2_d1):.4f}):

    | method | estimate | s.e. |
    |---|---|---|
    | pathwise | {call_pw.mean():.4f} | {call_pw.std()/sqrt(s2_R):.4f} |
    | LRM | {call_lr.mean():.4f} | {call_lr.std()/sqrt(s2_R):.4f} |
    | bump (CRN, $\\delta=1$) | {call_bp.mean():.4f} | {call_bp.std()/sqrt(s2_R):.4f} |

    **Delta of a digital call** (exact = {exact_dig:.4f}):

    | method | estimate | s.e. |
    |---|---|---|
    | pathwise | 0.0000 | - (WRONG: payoff insensitive to small $S_0$ moves) |
    | LRM | {dig_lr.mean():.4f} | {dig_lr.std()/sqrt(s2_R):.4f} |
    """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Conclusion (matches the summary table in the slides).** *Pathwise* has the lowest variance but only works for the smooth call payoff; *LRM* works for both - including the digital - at higher variance; *bump-and-reprice* is biased (here the step $\delta=1$ is deliberately large) but is always applicable.
    """)
    return


if __name__ == "__main__":
    app.run()
