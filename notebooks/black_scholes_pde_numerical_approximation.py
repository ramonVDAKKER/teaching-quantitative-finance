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
    # Quantitative Finance - solving the Black-Scholes Partial Differential Equation numerically

    *This is the interactive [marimo](https://marimo.io) version of
    [black_scholes_pde_numerical_approximation.ipynb](https://github.com/ramonVDAKKER/teaching-quantitative-finance/blob/main/notebooks/black_scholes_pde_numerical_approximation.ipynb);
    it runs entirely in your browser. Use the sliders to explore the effect of the parameters.*
    """)
    return


@app.cell
def _():
    import numpy as np
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    import matplotlib.pyplot as plt
    from scipy.stats import norm
    return diags, norm, np, plt, spsolve


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Background and setup

    ### 1.1 Setup

    We consider the standard Black-Scholes market in which two assets are traded. A risky asset (stock) is traded with price process described by the SDE
    $$dS_t = \mu S_t dt +\sigma S_t dW_t,$$
    where $S_0=s_0>0$, $W$ is a standard Brownian motion, and $\mu,\sigma>0$. The second asset is a money-market-account described by
    $$dB_t = r B_t dt,$$
    where $r$ denotes the (deterministic) interest and $B_0=b_0>0$. As usual, 'frictionless' trading is assumed (no restrictions on short selling, no restrictions on fractional positions, trading in continuous-time, no transaction costs).

    ### 1.2 Self-financing Markovian portfolios and the Black-Scholes Partial Differential Equation

    Consider a maturity $T>0$ (corresponding to the expiration dates of options that we will consider).
    For the above market we consider self-financing, Markovian trading strategies. So the price/value of the portfolio at time $t$ can be written as $V_t = F(t, S_t)$ for a fixed function $F$.
    We have seen that such a function $F$ is a solution to the equation (the Black-Scholes Partial Differential Equation)
    $$
    \frac{\partial G}{\partial t}(t,s) + r s \frac{\partial G}{\partial
    s}(t,s) + \frac{1}{2}\sigma^2 s^2 \frac{\partial^2 G}{\partial
    s^2}(t,s) -r G(t,s)=0\quad \forall s\in (0,\infty),\quad \forall t\in [0,T].\qquad(\star)
    $$

    ### 1.3 Application to pricing of European options

    If we want to price, using the no-arbitrage principle, a European option that has payoff $h(S_T)$ at maturity $T$, then we can exploit the Black-Scholes Partial Differential Equation as follows:

    *   Solve the PDE $(\star)$ under the boundary condition $G(s,T) = h(s)$ for all $s>0$. Denote the solution by $F$.
    *   In this case $F(t,S_t)$ is the value/price of the self-financing portfolio that has value $h(S_T)$ at $t=T$.
    *   No-arbitrage thus implies that the price of the option, for $t\in [0,T)$, must be given by $p_t = F(t,S_t)$.

    ### 1.4 Need for numerical approach to solve the PDE

    Sometimes it is possible to find an analytical solution to the PDE in combination with a boundary condition (for example, the Black-Scholes formula for the price of a European call option). In case we are not able to obtain a closed-form solution to the PDE, we can resort to numerical techniques to obtain an approximation to the solution. Below we discuss one of the simplest numerical algorithms.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. The algorithm

    Divide the time-interval $[0, T]$ into $N$ equally sized subintervals of length $dt$. The price of the underlying asset will in principle take values in $[0,\infty)$. In the algorithm an artificial limit $S_{\text{max}}$ is introduced. The size of
    $S_{\text{max}}$ requires experimentation. Next, the interval $[0, S_{\text{max}}]$ is divided into $M$ equally sized subintervals of length $ds$.
    So we approximate the continuous space $[0, T]\times [0,\infty)$ by a finite grid $(t_i, s_j )$, where $t_i = i\cdot dt$ and
    $s_j = j\cdot ds$, $i\in\{0, 1, . . . ,N\}$ and $j\in\{0, 1, . . . ,M\}$.

    Abbreviating $G(t_i,s_j)$ to $G_{i,j}$ and using the approximations
    $$\frac{\partial G}{\partial s}(t_i,s_j)\approx \frac{ G_{i,j+1} -G_{i,j-1}} {2 ds },\qquad
    \frac{\partial^2 G}{\partial s^2}(t_i,s_j)\approx\frac{ G_{i,j+1} -2G_{i,j} +G_{i,j-1}} { (ds)^2 },\qquad
    \frac{\partial G}{\partial t}(t_i,s_j)\approx \frac{ G_{i+1,j} -G_{i,j}} {dt },
    $$
    the PDE $(\star)$ turns into the linear equations
    $$a_j G_{i,j-1}+b_j G_{i,j}+c_j G_{i,j+1}- G_{i+1,j}=0,$$
    with
    $$a_j = \tfrac{1}{2}rjdt - \tfrac{1}{2}\sigma^2j^2dt,\qquad b_j = 1 + \sigma^2j^2 dt + rdt,\qquad c_j = -\tfrac{1}{2}rjdt - \tfrac{1}{2}\sigma^2 j^2dt.$$
    For fixed $i$ this is a tridiagonal system in the unknowns $G_{i,1},\dots,G_{i,M-1}$ (given time level $i+1$), which we solve stepping backward from $t=T$.

    The boundary values $G_{i,0}$ and $G_{i,M}$ should be derived by ad hoc arguments and are specific for the derivative of interest.
    For example, for a European put option, $h(s)=\max\{K-s,0\}$, we set $G_{i,0}=K\operatorname{e}^{-r(T-t_i)}$ and $G_{i,M}=0$. (If the stock price hits $0$ it stays at $0$, so the put then pays $K$ at maturity with certainty; its value at time $t_i$ is therefore the *discounted* strike.)
    """)
    return


@app.cell
def _(diags, np, plt, spsolve):
    class NumericalProxyPDE:
        """Class implementing the (implicit finite-difference) algorithm that has been described above."""

        def __init__(
            self,
            Smax: float,
            dS: float,
            dT: float,
            T: float,
            r: float,
            sigma: float,
        ):

            self.M = int(np.ceil(Smax / dS))  # number of subintervals in grid for stock price
            self.ds = Smax / self.M  # mesh in grid for stock price
            self.N = int(np.ceil(T / dT))  # number of subintervals in grid for time
            self.dt = T / self.N  # mesh in grid for time
            self.t = np.linspace(0, T, self.N + 1)
            self.S = np.linspace(0, Smax, self.M + 1)
            J = np.arange(1, self.M - 1 + 1)
            self.a = 0.5 * r * J * self.dt - 0.5 * sigma ** 2 * J ** 2 * self.dt
            b = 1 + sigma ** 2 * self.dt * J ** 2 + r * self.dt
            self.c = -0.5 * r * self.dt * J - 0.5 * sigma ** 2 * self.dt * J ** 2
            self.A = diags([self.a[1:], b, self.c[:-1]], offsets=[-1, 0, 1]).tocsc()
            self.G = np.zeros((self.N + 1, self.M + 1))  # time x stock price
            self.Smax = Smax

        def solve_pde(self, boundary_equation_maturity, boundary_equation_smin, boundary_equation_smax):
            """Solves the PDE backward in time.

            Args:
                boundary_equation_maturity: payoff h(s), used at t=T;
                boundary_equation_smin: value of the derivative at s=0 as function of t;
                boundary_equation_smax: value of the derivative at s=Smax as function of t
                    (this approximation only makes sense if Smax is large enough!).
            """

            self.G[self.N, :] = boundary_equation_maturity(self.S)  # boundary at t=T, i.e. pay-off
            self.G[:, 0] = boundary_equation_smin(self.t)  # if S hits 0 then it stays at 0
            self.G[:, self.M] = boundary_equation_smax(self.t)
            for i in range(self.N, 0, -1):
                y = self.G[i, 1 : self.M].copy()  # copy() so we do not overwrite the stored solution
                y[0] = y[0] - self.a[0] * self.G[i - 1, 0]
                y[-1] = y[-1] - self.c[-1] * self.G[i - 1, self.M]
                self.G[i - 1, 1 : self.M] = spsolve(self.A, y)


    class SolvePDEBoundaryNumerically(NumericalProxyPDE):
        def __init__(
            self,
            Smax: float,
            dS: float,
            dT: float,
            T: float,
            r: float,
            sigma: float,
            boundary_equation_maturity,
            boundary_equation_smin,
            boundary_equation_smax,
        ):

            super().__init__(Smax, dS, dT, T, r, sigma)
            self.solve_pde(boundary_equation_maturity, boundary_equation_smin, boundary_equation_smax)

        def plot_price(self):

            f, ax = plt.subplots(figsize=(13, 5))
            ax.plot(self.S, self.G[0, :])
            ax.set_title("Price option at t=0 as function of $s_0$")
            ax.set_xlabel("$s_0$")
            ax.set_ylabel("price option")
            return ax
    return (SolvePDEBoundaryNumerically,)


@app.cell
def _(norm, np):
    class BlackScholesOptionPrice:
        """Class for Black-Scholes price of European put and call options."""

        def __init__(self, strike: float, r: float, sigma: float):

            self.r = r
            self.sigma = sigma
            self.strike = strike

        def _d1_and_d2(self, current_stock_price, time_to_maturity):
            d1 = (
                np.log(current_stock_price / self.strike)
                + (self.r + 0.5 * self.sigma ** 2) * time_to_maturity
            ) / (self.sigma * np.sqrt(time_to_maturity))
            d2 = d1 - self.sigma * np.sqrt(time_to_maturity)
            return d1, d2

        def price_put(self, current_stock_price, time_to_maturity):
            d1, d2 = self._d1_and_d2(current_stock_price, time_to_maturity)
            return np.exp(-self.r * time_to_maturity) * self.strike * norm.cdf(-d2) - current_stock_price * norm.cdf(-d1)

        def price_call(self, current_stock_price, time_to_maturity):
            d1, d2 = self._d1_and_d2(current_stock_price, time_to_maturity)
            return current_stock_price * norm.cdf(d1) - np.exp(-self.r * time_to_maturity) * self.strike * norm.cdf(d2)
    return (BlackScholesOptionPrice,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Application to a European put option

    We solve the PDE numerically for a European put (boundary conditions: $G_{i,0}=K\operatorname{e}^{-r(T-t_i)}$, $G_{i,M}=0$, $G_{N,j}=\max\{K-s_j,0\}$) and compare, at $t=0$, to the exact Black-Scholes formula. The grid uses $S_{\text{max}}=250$, $ds=0.5$, $dt=0.01$ (kept modest so the in-browser computation stays fast; a finer grid reduces the error further).
    """)
    return


@app.cell
def _(mo):
    pde_strike = mo.ui.slider(60, 120, step=1, value=90, label="$K$")
    pde_r = mo.ui.slider(0.0, 0.05, step=0.005, value=0.01, label="$r$")
    pde_sigma = mo.ui.slider(0.05, 0.5, step=0.01, value=0.15, label=r"$\sigma$")
    pde_T = mo.ui.slider(0.25, 3.0, step=0.25, value=3.0, label="$T$")
    mo.hstack([pde_strike, pde_r, pde_sigma, pde_T], justify="start", gap=2)
    return pde_T, pde_r, pde_sigma, pde_strike


@app.cell
def _(
    BlackScholesOptionPrice,
    SolvePDEBoundaryNumerically,
    mo,
    np,
    pde_T,
    pde_r,
    pde_sigma,
    pde_strike,
    plt,
):
    _K, _r, _sig, _T = pde_strike.value, pde_r.value, pde_sigma.value, pde_T.value
    pde_put = SolvePDEBoundaryNumerically(
        Smax=250,
        dS=0.5,
        dT=0.01,
        T=_T,
        r=_r,
        sigma=_sig,
        boundary_equation_maturity=lambda s: np.maximum(_K - s, 0),
        boundary_equation_smin=lambda t: _K * np.exp(-_r * (_T - t)),
        boundary_equation_smax=lambda t: 0 * t,
    )
    _bs = BlackScholesOptionPrice(_K, _r, _sig)
    _exact = _bs.price_put(pde_put.S[1:], _T)
    _ax = pde_put.plot_price()
    _ax.plot(pde_put.S[1:], _exact, "r--", label="exact (Black-Scholes)")
    _ax.plot(pde_put.S, np.zeros_like(pde_put.S), lw=0)  # keep y-axis anchored at 0
    _ax.legend(["numerical approximation", "exact (Black-Scholes)"])
    _ax.set_title(f"Put price at t=0: numerical vs exact (K={_K}, r={100*_r:g}%, sigma={100*_sig:g}%, T={_T})")
    _err = np.max(np.abs(pde_put.G[0, 1:] - _exact))
    mo.vstack([plt.gcf(), mo.md(f"Maximum absolute difference between the numerical approximation and the exact price on the grid: **{_err:.4f}**")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Explicit scheme: the trinomial-tree view and stability

    The implementation above is an **implicit** scheme: at each time step we solved a linear system, and it is unconditionally stable.

    A simpler variant evaluates the *space* derivatives at the **later** time level. Then no linear solve is needed and we get an **explicit** scheme,
    $$ G_{i,j} \;=\; \alpha_j\,G_{i+1,j-1} + \beta_j\,G_{i+1,j} + \gamma_j\,G_{i+1,j+1}, $$
    with
    $$ \alpha_j=\tfrac12\,dt\,(\sigma^2 j^2 - r j),\qquad \beta_j = 1 - dt\,(\sigma^2 j^2 + r),\qquad \gamma_j=\tfrac12\,dt\,(\sigma^2 j^2 + r j). $$

    Each grid value is a weighted combination of **three** neighbours one step later - a *trinomial tree*. The weights behave like (discounted) risk-neutral probabilities, and the scheme is well-behaved only while they stay **nonnegative**. In particular $\beta_j\ge 0$ requires
    $$ dt \;\le\; \frac{1}{\sigma^2 j^2 + r}, $$
    i.e. $dt$ small relative to $ds^2$ (a *stability / CFL* condition). When it is violated the 'probabilities' turn negative and the solution explodes - the same phenomenon that forces the binomial-tree spacing to scale like $\sigma\sqrt{dt}$ so that $q\in(0,1)$.

    **Try it yourself:** with the slider below you control the number of time steps $N$ (i.e. $dt = T/N$) for a call with $K=100$, $r=3\%$, $\sigma=20\%$, $T=1$ on a grid with $M=100$ price steps. The CFL threshold is $N_{\text{CFL}} = T\,(\sigma^2 (M-1)^2 + r) \approx 393$: for smaller $N$ the scheme explodes.
    """)
    return


@app.cell
def _(np):
    def explicit_call(Smax, M, N, T, r, sigma, K):
        """Explicit (trinomial) finite-difference scheme for a European call."""
        dt = T / N
        S = np.linspace(0, Smax, M + 1)
        j = np.arange(M + 1)
        G = np.zeros((N + 1, M + 1))
        G[N, :] = np.maximum(S - K, 0.0)                 # terminal payoff
        jj = j[1:M]
        alpha = 0.5 * dt * (sigma**2 * jj**2 - r * jj)
        beta = 1.0 - dt * (sigma**2 * jj**2 + r)
        gamma = 0.5 * dt * (sigma**2 * jj**2 + r * jj)
        for i in range(N, 0, -1):                        # step backward in time
            t_prev = (i - 1) * dt
            G[i-1, 1:M] = alpha*G[i, 0:M-1] + beta*G[i, 1:M] + gamma*G[i, 2:M+1]
            G[i-1, 0] = 0.0                              # call is worthless at s=0
            G[i-1, M] = Smax - K*np.exp(-r*(T - t_prev)) # deep in-the-money boundary
        return S, G, (alpha, beta, gamma)
    return (explicit_call,)


@app.cell
def _(mo):
    cfl_N = mo.ui.slider(100, 700, step=25, value=500, label="number of time steps $N$")
    cfl_N
    return (cfl_N,)


@app.cell
def _(BlackScholesOptionPrice, cfl_N, explicit_call, mo, np, plt):
    cfl_K, cfl_r, cfl_sigma, cfl_T = 100.0, 0.03, 0.20, 1.0
    cfl_S, cfl_G, (_al, cfl_beta, _ga) = explicit_call(
        Smax=200, M=100, N=cfl_N.value, T=cfl_T, r=cfl_r, sigma=cfl_sigma, K=cfl_K
    )
    cfl_stable = cfl_beta.min() >= 0
    _bs_cfl = BlackScholesOptionPrice(cfl_K, cfl_r, cfl_sigma)
    _exact_cfl = _bs_cfl.price_call(cfl_S[1:], cfl_T)
    _fig, _ax2 = plt.subplots(figsize=(13, 5))
    _ax2.plot(cfl_S, cfl_G[0, :], "b", label="explicit FD (t=0)")
    _ax2.plot(cfl_S[1:], _exact_cfl, "r--", label="Black-Scholes")
    _ax2.set_ylim(-10, 120)
    _ax2.set_xlabel("$s_0$")
    _ax2.set_ylabel("call price")
    _ax2.set_title(
        f"N={cfl_N.value}, dt={cfl_T/cfl_N.value:.4f}: min weight = {cfl_beta.min():.3f} "
        + ("(stable)" if cfl_stable else "(NEGATIVE: unstable, solution explodes!)")
    )
    _ax2.legend()
    _msg = (
        "All weights nonnegative: the scheme is **stable** and matches the Black-Scholes formula."
        if cfl_stable
        else f"The smallest weight is **negative** ({cfl_beta.min():.2f}); the solution blows up "
        f"(max |price| = {np.max(np.abs(cfl_G[0])):.2e}) and leaves the plotting window."
    )
    mo.vstack([plt.gcf(), mo.md(_msg)])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Takeaway.** The explicit finite-difference scheme *is* a trinomial tree; the binomial tree is its two-neighbour cousin. Numerical stability $\Leftrightarrow$ nonnegative weights $\Leftrightarrow$ a valid risk-neutral measure $\Leftrightarrow$ no arbitrage. The PDE, the trees, and the grid are three views of the same object.
    """)
    return


if __name__ == "__main__":
    app.run()
