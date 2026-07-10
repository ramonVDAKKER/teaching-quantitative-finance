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
    # Illustration Black-Scholes prices for European call and put options

    *This is the interactive [marimo](https://marimo.io) version of
    [illustration_black_scholes_price.ipynb](https://github.com/ramonVDAKKER/teaching-quantitative-finance/blob/main/notebooks/illustration_black_scholes_price.ipynb);
    it runs entirely in your browser. Use the sliders to explore the effect of the parameters.*
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    from scipy.stats import norm
    import matplotlib.pyplot as plt
    return norm, np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the standard Black-Scholes market two assets are traded. A risky asset (stock) is traded with price process described by the SDE
    $$dS_t = \mu S_t dt +\sigma S_t dW_t,$$ where $S_0=s_0$, $W$ is a standard Brownian motion, and $\mu,\sigma>0$. The second asset is a money-market-account described by $dB_t = r B_t dt$ where $r$ denotes the (deterministic) interest. As usual, 'frictionless' trading is assumed (no restrictions on short selling, no restrictions on fractional positions, trading in continuous-time, no transaction costs).

    Under these assumptions the price of a European put option, at time $t\in[0, T)$, with (remaining) time-to-expiration/maturity $T-t$ and strike $K>0$ is given by
    $$p_t = \operatorname{e}^{-r(T-t)}  K \Phi( -d_2) - S_t \Phi ( - d_1),$$
    where
    $$d_1 = \frac{\log( S_t / K) + ( r + 0.5\sigma^2 )( T-t)}{\sigma \sqrt{T-t}}
    \text{ and }   d_2 = d_1 - \sigma \sqrt{T-t}.$$
    And the price of a call option (with the same specs) is given by
    $$c_t = S_t \Phi(d_1) - \operatorname{e}^{-r(T-t)}  K \Phi( d_2).$$

    Please note that we can write $c_t = f(T-t, S_t)$ and $p_t =g(T-t,S_t)$ for suitable functions $f$ and $g$.
    """)
    return


@app.cell
def _(norm, np):
    class BlackScholesOptionPrice():
        """Class for Black-Scholes price of European put and call options."""

        def __init__(self, strike: float, r: float, sigma: float):

            self.r = r
            self.sigma = sigma
            self.strike = strike

        def _d1_and_d2(self, current_stock_price, time_to_maturity):
            """Calculates auxiliary d_1 and d_2 which enter the N(0, 1) cdf in the pricing formulas"""

            with np.errstate(divide="ignore", invalid="ignore"):
                d1 = (np.log(current_stock_price / self.strike) + (self.r + 0.5 * self.sigma ** 2) * time_to_maturity) / (self.sigma * np.sqrt(time_to_maturity))
                d2 = d1 - self.sigma * np.sqrt(time_to_maturity)
            return d1, d2

        def price_put(self, current_stock_price, time_to_maturity):
            """Calculates price of European put option.

            At time_to_maturity=0 the payoff max(K - S, 0) is returned."""

            d1, d2 = self._d1_and_d2(current_stock_price, time_to_maturity)
            price = np.exp(-self.r * time_to_maturity) * self.strike * norm.cdf(-d2) - current_stock_price * norm.cdf(-d1)
            return np.where(time_to_maturity > 0, price, np.maximum(self.strike - current_stock_price, 0))

        def price_call(self, current_stock_price, time_to_maturity):
            """Calculates price of European call option.

            At time_to_maturity=0 the payoff max(S - K, 0) is returned."""

            d1, d2 = self._d1_and_d2(current_stock_price, time_to_maturity)
            price = current_stock_price * norm.cdf(d1) - np.exp(-self.r * time_to_maturity) * self.strike *  norm.cdf(d2)
            return np.where(time_to_maturity > 0, price, np.maximum(current_stock_price - self.strike, 0))
    return (BlackScholesOptionPrice,)


@app.cell
def _(BlackScholesOptionPrice, mo, np):
    example_bs = BlackScholesOptionPrice(strike=100, r=0.02, sigma=0.2)
    example_price = example_bs.price_put(current_stock_price=100, time_to_maturity=1)
    mo.md(f"Price of put for strike=100, r=2%, sigma=20%, $S_t$=100, $T-t$=1: **{np.round(example_price, 2)}**")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the following plot we fix the time-to-maturity/expiration and consider the price of the put option as function of the current stock price. The dashed line shows the payoff $\max\{K-S_t, 0\}$. Note that for small $r$ and small time-to-maturity the put price is close to the payoff, and that a deep in-the-money put can be worth *less* than its payoff ('negative time value') - compare the food-for-thought box in the slides.
    """)
    return


@app.cell
def _(mo):
    put_strike = mo.ui.slider(50, 110, step=1, value=100, label="$K$")
    put_r = mo.ui.slider(0.0, 0.1, step=0.01, value=0.02, label="$r$")
    put_sigma = mo.ui.slider(0.01, 0.8, step=0.01, value=0.2, label=r"$\sigma$")
    put_tau = mo.ui.slider(0.01, 2.0, step=0.01, value=1.0, label="$T-t$")
    mo.hstack([put_strike, put_r, put_sigma, put_tau], justify="start", gap=2)
    return put_r, put_sigma, put_strike, put_tau


@app.cell
def _(BlackScholesOptionPrice, np, pd, put_r, put_sigma, put_strike, put_tau):
    _bs = BlackScholesOptionPrice(put_strike.value, put_r.value, put_sigma.value)
    _grid = np.linspace(0.01, 200, 2000)
    _price = _bs.price_put(_grid, put_tau.value)
    _ax = pd.Series(_price, index=_grid).plot(
        figsize=(13, 5),
        title=f"Price put option as function of $S_t$, r={100 * put_r.value:.0f}%, sigma={100 * put_sigma.value:.0f}%, K={put_strike.value}, T-t={put_tau.value}",
        xlabel="$S_t$",
        ylabel="put price",
        label="put price",
    )
    _ax.plot(_grid, np.maximum(put_strike.value - _grid, 0), "r--", label=r"payoff $\max\{K-S_t,0\}$")
    _ax.legend()
    _ax
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the following plots we first simulate paths of a Geometric Brownian motion, on $[0,T]$, and evaluate the evolution of the price of a put option along each scenario. Note that at $t=T$ the put price equals the payoff $\max\{K-S_T, 0\}$: paths that end above the strike (horizontal line in the first plot) end at price $0$.
    """)
    return


@app.cell
def _(np, pd):
    class GeometricBrownianMotion():
        r"""Class to simulate paths of a Geometric Brownian motion, i.e. X_t=X_0\exp((mu-0.5sigma^2)t+sigma*W_t)"""

        def __init__(self, starting_value, mu, sigma, n, T, time_step, seed=None):

            self.name = "Geometric Brownian Motion"
            self.n = n
            self.T = T
            self.starting_value = starting_value
            self.time_grid, self.time_step = np.linspace(
                0, self.T, num=1 + int(T / time_step), endpoint=True, retstep=True
            )  # note that time_step is adapted (if needed) in order to get equally-spaced grid
            if seed is not None:
                np.random.seed(seed)
            aux = np.random.normal(loc=0.0, scale=1.0, size=(self.n, len(self.time_grid)-1))
            aux = np.concatenate([np.zeros((self.n, 1)), aux], axis=1)
            W = np.cumsum(sigma * np.sqrt(self.time_step) * aux, axis=1)
            self.paths = starting_value * np.exp((mu - 0.5 * sigma ** 2) * self.time_grid + W)

        def plot(self):

            title = "Simulated sample paths from GBM"
            return pd.DataFrame(
                self.paths.T,
                columns=[f"path {j}" for j in range(1, 1 + len(self.paths))],
                index=self.time_grid,
            ).plot(kind="line", title=title, figsize=(13, 5))
    return (GeometricBrownianMotion,)


@app.cell
def _(mo):
    scen_strike = mo.ui.slider(50, 120, step=1, value=80, label="$K$")
    scen_mu = mo.ui.slider(-0.1, 0.3, step=0.01, value=0.1, label=r"$\mu$")
    scen_sigma = mo.ui.slider(0.01, 0.8, step=0.01, value=0.3, label=r"$\sigma$")
    mo.hstack([scen_strike, scen_mu, scen_sigma], justify="start", gap=2)
    return scen_mu, scen_sigma, scen_strike


@app.cell
def _(
    BlackScholesOptionPrice,
    GeometricBrownianMotion,
    mo,
    pd,
    plt,
    scen_mu,
    scen_sigma,
    scen_strike,
):
    # option parameters:
    scen_T = 1
    scen_r = 0.02
    # simulation setting:
    scen_n = 15
    scen_gbm = GeometricBrownianMotion(
        starting_value=100, mu=scen_mu.value, sigma=scen_sigma.value, n=scen_n, T=scen_T, time_step=0.001, seed=None
    )
    scen_gbm.plot()
    plt.axhline(scen_strike.value)
    _fig_gbm = plt.gcf()
    scen_bs = BlackScholesOptionPrice(scen_strike.value, scen_r, scen_sigma.value)
    scen_price = scen_bs.price_put(scen_gbm.paths, scen_T - scen_gbm.time_grid)
    pd.DataFrame(
        scen_price.T,
        index=scen_gbm.time_grid,
        columns=[f"path {j}" for j in range(1, 1 + len(scen_gbm.paths))],
    ).plot(figsize=(13, 5), title=f"Price of put with T={scen_T}, K={scen_strike.value} evaluated on scenarios above")
    _fig_put = plt.gcf()
    mo.vstack([_fig_gbm, _fig_put])
    return


if __name__ == "__main__":
    app.run()
