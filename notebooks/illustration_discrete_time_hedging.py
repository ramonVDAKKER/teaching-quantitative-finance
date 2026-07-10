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
    # Notebook illustration (discrete-time) delta-hedging and delta-gamma-hedging

    *This is the interactive [marimo](https://marimo.io) version of
    [illustration_discrete_time_hedging.ipynb](https://github.com/ramonVDAKKER/teaching-quantitative-finance/blob/main/notebooks/illustration_discrete_time_hedging.ipynb);
    it runs entirely in your browser (with somewhat smaller simulation sizes to keep it responsive). Use the controls to explore the effect of the rebalancing frequency and the hedging instrument.*
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    return norm, np, pd, plt


@app.cell
def _(norm, np):
    class BlackScholesOptionPrice():
        """Class for Black-Scholes price and Greeks of European put and call options."""

        def __init__(self, strike: float, r: float, sigma: float):

            self.r = r
            self.sigma = sigma
            self.strike = strike

        def _d1_and_d2(self, current_stock_price, time_to_maturity):
            d1 = (np.log(current_stock_price / self.strike) + (self.r + 0.5 * self.sigma ** 2) * time_to_maturity) / (self.sigma * np.sqrt(time_to_maturity))
            d2 = d1 - self.sigma * np.sqrt(time_to_maturity)
            return d1, d2

        def price_put(self, current_stock_price, time_to_maturity):
            d1, d2 = self._d1_and_d2(current_stock_price, time_to_maturity)
            return np.exp(-self.r * time_to_maturity) * self.strike * norm.cdf(-d2) - current_stock_price * norm.cdf(-d1)

        def delta_put(self, current_stock_price, time_to_maturity):
            d1, _ = self._d1_and_d2(current_stock_price, time_to_maturity)
            return - norm.cdf(- d1)

        def price_call(self, current_stock_price, time_to_maturity):
            d1, d2 = self._d1_and_d2(current_stock_price, time_to_maturity)
            return current_stock_price * norm.cdf(d1) - np.exp(-self.r * time_to_maturity) * self.strike * norm.cdf(d2)

        def delta_call(self, current_stock_price, time_to_maturity):
            d1, _ = self._d1_and_d2(current_stock_price, time_to_maturity)
            return norm.cdf(d1)

        def _gamma(self, current_stock_price, time_to_maturity):
            d1, _ = self._d1_and_d2(current_stock_price, time_to_maturity)
            return norm.pdf(d1) / (current_stock_price * self.sigma * np.sqrt(time_to_maturity))

        def gamma_call(self, current_stock_price, time_to_maturity):
            return self._gamma(current_stock_price, time_to_maturity)

        def gamma_put(self, current_stock_price, time_to_maturity):
            return self._gamma(current_stock_price, time_to_maturity)
    return (BlackScholesOptionPrice,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Example discrete-time delta-hedging

    The setup is as follows:

    *   we adopt the standard models for $B$ and $S$;
    *   we consider a situation in which we are only allowed to trade on a discrete-time grid;
    *   financial institution that has a position of $a$ put options with exercise price $K$ and maturity $T$ at $t=0$ (in case $a<0$ the institution sells (writes) the puts, in case $a>0$ the institution buys the options);
    *   we will evaluate two strategies in this section:
        1.   No active risk management: the institution takes no actions apart from investing $-a p_0$, with $p_0$ the price of one put at $t=0$, in the money-market-account $B$. This implies that there are no net cashflows in the time interval $[0,T)$. And at maturity $T$ the net cashflow (profit/loss) equals $a\times (\max(K-S_T, 0) - p_0 \exp(rT))$.
        2.   Delta-hedging: the institution trades in $S$ and $B$, at each point-in-time (on the discrete grid), such that a) the total portfolio (of the puts and the positions in $S$ and $B$) is delta-neutral and b) the rebalancing of the positions in $S$ and $B$ is budget-neutral.

    *Remark:* Please recall that an implementation of 2) in continuous-time would imply that the total portfolio has value 0 at each point in time $t\in [0, T]$. This means that the institution would be able to take the position in the put (to serve its clients) without bearing any risk.

    Parameters: $S_0=100$, $\mu=5\%$, $\sigma=20\%$, $B_0=1$, $r=1\%$; the institution writes $1{,}000$ puts with $K=90$ and $T=1$.
    """)
    return


@app.cell
def _():
    S_0 = 100
    mu = 0.05
    sigma = 0.20
    B_0 = 1
    r = 0.01
    T = 1
    K = 90
    num_puts = -1000  # negative = institution writes puts
    return B_0, K, S_0, T, mu, num_puts, r, sigma


@app.cell
def _(mo):
    hedge_freq = mo.ui.dropdown(
        options={"monthly (12/year)": 12, "weekly (52/year)": 52, "daily (252/year)": 252},
        value="daily (252/year)",
        label="rebalancing frequency",
    )
    mc_M = mo.ui.slider(500, 5000, step=500, value=2000, label="Monte Carlo replications $M$")
    mo.hstack([hedge_freq, mc_M], justify="start", gap=2)
    return hedge_freq, mc_M


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""### A single realization: paths, positions and the mismatch at maturity""")
    return


@app.cell
def _(B_0, BlackScholesOptionPrice, K, S_0, T, hedge_freq, mu, norm, np, num_puts, pd, plt, r, sigma):
    _n = int(T * hedge_freq.value)
    _dt = T / _n
    _time = np.linspace(0, T, _n + 1)
    _put = BlackScholesOptionPrice(K, r, sigma)

    _S = np.empty(_n + 1)
    _S[0] = S_0
    _B = B_0 * np.exp(r * _time)
    _S[1:] = S_0 * np.exp(np.cumsum((mu - 0.5 * sigma ** 2) * _dt + sigma * np.sqrt(_dt) * norm.rvs(size=_n)))

    _phi = np.full(_n + 1, np.nan)
    _psi = np.full(_n + 1, np.nan)
    _price_puts = np.zeros(_n + 1)
    _total = np.zeros(_n + 1)

    _price_puts[0] = num_puts * _put.price_put(S_0, T)
    _phi[0] = -num_puts * _put.delta_put(S_0, T)
    _psi[0] = -(_price_puts[0] + _phi[0] * S_0) / B_0
    _total[0] = _price_puts[0] + _phi[0] * _S[0] + _psi[0] * _B[0]

    for _k in range(1, _n + 1):
        _value = _phi[_k - 1] * _S[_k] + _psi[_k - 1] * _B[_k]
        if _k == _n:
            _price_puts[_k] = num_puts * np.maximum(K - _S[_k], 0)
            _total[_k] = _price_puts[_k] + _value
            break
        _ttm = T - _time[_k]
        _price_puts[_k] = num_puts * _put.price_put(_S[_k], _ttm)
        _phi[_k] = -num_puts * _put.delta_put(_S[_k], _ttm)
        _psi[_k] = (_value - _phi[_k] * _S[_k]) / _B[_k]
        _total[_k] = _price_puts[_k] + _phi[_k] * _S[_k] + _psi[_k] * _B[_k]

    _df = pd.DataFrame({"t": _time, "S": _S, "B": _B, "position S": _phi, "position B": _psi,
                        "num_puts * put_price": _price_puts, "total_portfolio_value": _total})
    _fig, _ax = plt.subplots(2, 3, figsize=(13, 7))
    _df.plot(x="t", y="B", title="path B", ax=_ax[0, 0], legend=False)
    _df.plot(x="t", y="S", title="path S (red=strike puts)", ax=_ax[0, 1], legend=False)
    _ax[0, 1].axhline(y=K, color="r")
    _df.plot(x="t", y="num_puts * put_price", title=f"value {num_puts} puts", ax=_ax[0, 2], legend=False)
    _df.iloc[:-1].plot(x="t", y="position B", title="path psi (position B)", ax=_ax[1, 0], legend=False)
    _df.iloc[:-1].plot(x="t", y="position S", title="path phi (position S)", ax=_ax[1, 1], legend=False)
    _df.plot(x="t", y="total_portfolio_value", title="mismatch", ax=_ax[1, 2], legend=False)
    plt.tight_layout()
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Monte Carlo comparison of the two strategies

    Next we use Monte Carlo simulations to approximate the distribution of the total portfolio value at maturity $T$ for both strategies.
    """)
    return


@app.cell
def _(B_0, BlackScholesOptionPrice, K, S_0, T, mu, norm, np, num_puts, r, sigma):
    def simulate_multiple_paths_vectorized(K, T, S_0, mu, sigma, B_0, r,
                                           num_time_steps_per_unit_of_time, num_puts, M):
        """Vectorized Monte Carlo simulation for the delta hedging strategy."""
        n = int(T * num_time_steps_per_unit_of_time)
        dt = T / n
        put = BlackScholesOptionPrice(K, r, sigma)

        S = np.zeros((M, n + 1))
        S[:, 0] = S_0
        B = np.zeros((M, n + 1))
        B[:, 0] = B_0
        phi = np.zeros((M, n + 1))
        psi = np.zeros((M, n + 1))
        price_puts = np.zeros((M, n + 1))

        price_puts[:, 0] = num_puts * put.price_put(S_0, T)
        phi[:, 0] = -num_puts * put.delta_put(S_0, T)
        psi[:, 0] = -(price_puts[:, 0] + phi[:, 0] * S_0) / B_0

        Z = norm.rvs(size=(M, n))
        for k in range(1, n + 1):
            B[:, k] = B[:, k - 1] * np.exp(r * dt)
            S[:, k] = S[:, k - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, k - 1])
            if k == n:
                value = phi[:, k - 1] * S[:, k] + psi[:, k - 1] * B[:, k]
                price_puts[:, k] = num_puts * np.maximum(K - S[:, k], 0)
                total_portfolio_value = price_puts[:, k] + value
                break
            ttm = T - k * dt
            value = phi[:, k - 1] * S[:, k] + psi[:, k - 1] * B[:, k]
            price_puts[:, k] = num_puts * put.price_put(S[:, k], ttm)
            phi[:, k] = -num_puts * put.delta_put(S[:, k], ttm)
            psi[:, k] = (value - phi[:, k] * S[:, k]) / B[:, k]

        no_hedge = price_puts[:, -1] - price_puts[:, 0] * np.exp(r * T)
        return {"S_T": S[:, -1], "no_hedge": no_hedge, "delta_hedge": total_portfolio_value,
                "initial_put_value": price_puts[0, 0]}
    return (simulate_multiple_paths_vectorized,)


@app.cell
def _(K, T, S_0, B_0, hedge_freq, mc_M, mo, mu, np, num_puts, plt, r, sigma, simulate_multiple_paths_vectorized):
    mc_results = simulate_multiple_paths_vectorized(K, T, S_0, mu, sigma, B_0, r,
                                                    hedge_freq.value, num_puts, mc_M.value)
    _nh = mc_results["no_hedge"]
    _dh = mc_results["delta_hedge"]
    _fig2, _axs = plt.subplots(1, 2, figsize=(13, 4))
    _axs[0].hist(_nh, bins=50, density=True)
    _axs[0].set_title("net cashflow at maturity: no active risk management")
    _axs[1].hist(_dh, bins=50, density=True)
    _axs[1].set_title(f"net cashflow at maturity: delta hedging ({hedge_freq.value}/year)")
    plt.tight_layout()
    mo.vstack([
        mo.md(
            f"""
    | metric | no risk management | delta hedging |
    |---|---|---|
    | mean P&L | {_nh.mean():,.2f} | {_dh.mean():,.2f} |
    | std dev | {_nh.std():,.2f} | {_dh.std():,.2f} |
    | 5% quantile | {np.quantile(_nh, 0.05):,.2f} | {np.quantile(_dh, 0.05):,.2f} |
    | 95% quantile | {np.quantile(_nh, 0.95):,.2f} | {np.quantile(_dh, 0.95):,.2f} |
    | % of losses | {(_nh < 0).mean()*100:.1f}% | {(_dh < 0).mean()*100:.1f}% |
    """
        ),
        _fig2,
    ])
    return (mc_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Delta-Gamma Hedging

    While delta hedging neutralizes first-order price risk, it leaves the portfolio exposed to **gamma risk**. Delta-gamma hedging addresses this by making the portfolio both delta-neutral and gamma-neutral. To achieve this we need **two hedging instruments**: the stock $S$ (which has gamma $=0$) and a traded call option with a different strike (and, for simplicity, the same maturity $T$).

    At each rebalancing time we solve, for a position of $a$ puts, $\phi$ stocks, $\xi$ hedging calls and $\psi$ units of $B$:

    * **gamma neutrality**: $\xi = -a\,\Gamma_{\text{put}}/\Gamma_{\text{call}}$;
    * **delta neutrality**: $\phi = -a\,\Delta_{\text{put}} - \xi\,\Delta_{\text{call}}$;
    * **budget neutrality**: rebalancing the $(S, B, \text{call})$-positions is self-financing.
    """)
    return


@app.cell
def _(mo):
    dg_K_hedge = mo.ui.slider(95, 130, step=1, value=110, label="strike hedging call $K_{\\text{hedge}}$")
    dg_K_hedge
    return (dg_K_hedge,)


@app.cell
def _(B_0, BlackScholesOptionPrice, K, S_0, T, dg_K_hedge, hedge_freq, mc_M, mc_results, mo, mu, norm, np, num_puts, plt, r, sigma):
    def _simulate_delta_gamma(K_put, K_call, M, freq):
        n = int(T * freq)
        dt = T / n
        put = BlackScholesOptionPrice(K_put, r, sigma)
        call_hedge = BlackScholesOptionPrice(K_call, r, sigma)

        S = np.full(M, float(S_0))
        B = float(B_0)
        price_puts_0 = num_puts * put.price_put(S_0, T)
        xi = np.full(M, -num_puts * put.gamma_put(S_0, T) / call_hedge.gamma_call(S_0, T))
        phi = -num_puts * put.delta_put(S_0, T) - xi * call_hedge.delta_call(S_0, T)
        psi = -(price_puts_0 + xi * call_hedge.price_call(S_0, T) + phi * S) / B

        Z = norm.rvs(size=(M, n))
        for k in range(1, n + 1):
            B = B * np.exp(r * dt)
            S = S * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z[:, k - 1])
            if k == n:
                payoff_puts = num_puts * np.maximum(K_put - S, 0)
                call_final = xi * np.maximum(S - K_call, 0)
                return payoff_puts + phi * S + call_final + psi * B
            ttm = T - k * dt
            call_price = call_hedge.price_call(S, ttm)
            value = phi * S + xi * call_price + psi * B
            gamma_call = call_hedge.gamma_call(S, ttm)
            with np.errstate(divide="ignore", invalid="ignore"):
                xi_temp = -num_puts * put.gamma_put(S, ttm) / gamma_call
            xi = np.where(np.abs(gamma_call) > 1e-10, xi_temp, 0.0)
            phi = -num_puts * put.delta_put(S, ttm) - xi * call_hedge.delta_call(S, ttm)
            psi = (value - phi * S - xi * call_price) / B

    dg_pnl = _simulate_delta_gamma(K, dg_K_hedge.value, mc_M.value, hedge_freq.value)
    _dh2 = mc_results["delta_hedge"]
    _fig3, _ax3 = plt.subplots(figsize=(13, 4))
    _ax3.hist(_dh2, bins=50, density=True, alpha=0.6, label="delta hedging", color="blue")
    _ax3.hist(dg_pnl, bins=50, density=True, alpha=0.6, label="delta-gamma hedging", color="green")
    _ax3.axvline(0, color="red", linestyle="--", linewidth=1)
    _ax3.set_xlabel("net P&L at maturity")
    _ax3.legend()
    mo.vstack([
        mo.md(
            f"""
    | metric | delta hedging | delta-gamma hedging |
    |---|---|---|
    | mean P&L | {_dh2.mean():,.2f} | {dg_pnl.mean():,.2f} |
    | std dev | {_dh2.std():,.2f} | {dg_pnl.std():,.2f} |
    | 5% quantile | {np.quantile(_dh2, 0.05):,.2f} | {np.quantile(dg_pnl, 0.05):,.2f} |

    Std dev reduction relative to delta hedging: **{(1 - dg_pnl.std()/_dh2.std())*100:.1f}%**
    """
        ),
        _fig3,
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Hedge error and transaction costs vs rebalancing frequency

    How does the replication error depend on how often we rebalance - and what happens once trading costs money? With **transaction costs** (a proportional rate $\kappa$ per unit traded), rebalancing more often reduces the replication error ($\sim 1/\sqrt{n}$) but raises total turnover. The total is **U-shaped** - there is an optimal frequency (the Leland trade-off).
    """)
    return


@app.cell
def _(mo, np, plt):
    from math import erf, sqrt, exp

    def _Phiv(x):
        return 0.5 * (1 + np.vectorize(erf)(x / sqrt(2)))

    _S0, _K, _r, _sigma, _T, _mu = 100.0, 100.0, 0.03, 0.20, 1.0, 0.10
    _rng = np.random.default_rng(1)

    def _call_delta(s, tau):
        s = np.asarray(s, float)
        dd = (np.log(s / _K) + (_r + 0.5 * _sigma ** 2) * tau) / (_sigma * np.sqrt(tau))
        return _Phiv(dd)

    def _bs_call(s, tau):
        s = np.asarray(s, float)
        dd = (np.log(s / _K) + (_r + 0.5 * _sigma ** 2) * tau) / (_sigma * np.sqrt(tau))
        d2 = dd - _sigma * np.sqrt(tau)
        return s * _Phiv(dd) - _K * exp(-_r * tau) * _Phiv(d2)

    _premium = float(_bs_call(np.array([_S0]), _T)[0])
    _M = 1500
    _ns = [5, 10, 20, 40, 80, 160]

    def _simulate(n, kappa):
        dt = _T / n
        t = np.linspace(0, _T, n + 1)
        Z = _rng.standard_normal((_M, n))
        S = _S0 * np.exp(np.hstack([np.zeros((_M, 1)), np.cumsum((_mu - 0.5 * _sigma ** 2) * dt + _sigma * sqrt(dt) * Z, axis=1)]))
        sh = _call_delta(S[:, 0], _T)
        cash = _premium - sh * S[:, 0]
        turnover = np.abs(sh) * S[:, 0]
        for i in range(1, n):
            cash = cash * exp(_r * dt)
            di = _call_delta(S[:, i], _T - t[i])
            turnover = turnover + np.abs(di - sh) * S[:, i]
            cash = cash - (di - sh) * S[:, i]
            sh = di
        cash = cash * exp(_r * dt)
        pnl = cash + sh * S[:, -1] - np.maximum(S[:, -1] - _K, 0)
        return float(np.sqrt(np.mean(pnl ** 2))), float(kappa * np.mean(turnover))

    _kappa = 0.005
    _res = [_simulate(n, _kappa) for n in _ns]
    _err = np.array([a for a, _b in _res])
    _tc = np.array([b for _a, b in _res])

    _fig4, _axs4 = plt.subplots(1, 2, figsize=(13, 4))
    _axs4[0].loglog(_ns, _err, "o-", label="simulated RMS hedge error")
    _axs4[0].loglog(_ns, _err[0] * sqrt(_ns[0]) / np.sqrt(_ns), "r--", label=r"$\propto n^{-1/2}$")
    _axs4[0].set_xlabel("rebalancings $n$")
    _axs4[0].legend()
    _axs4[0].set_title("replication error shrinks like $1/\\sqrt{n}$")
    _axs4[1].plot(_ns, _err, "b-o", label="RMS hedge error")
    _axs4[1].plot(_ns, _tc, "g-s", label=f"transaction cost ($\\kappa$={_kappa})")
    _axs4[1].plot(_ns, _err + _tc, "k-^", lw=2, label="total")
    _axs4[1].set_xscale("log")
    _axs4[1].set_xlabel("rebalancings $n$")
    _axs4[1].legend()
    _axs4[1].set_title(f"total cost minimised near n = {_ns[int(np.argmin(_err + _tc))]}")
    plt.tight_layout()
    _fig4
    return


if __name__ == "__main__":
    app.run()
