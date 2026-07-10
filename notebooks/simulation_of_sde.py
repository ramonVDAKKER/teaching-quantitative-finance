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
    # Notebook simulation of stochastic processes

    This notebook accompanies the course **Quantitative Finance** (BSc Econometrics and OR, Tilburg University) and considers the simulation of stochastic processes. The following processes will be considered:

    *   Brownian motion;
    *   Brownian motion with drift;
    *   Geometric Brownian motion (GBM);
    *   Ito diffusion processes $dX_t=a(t, X_t)dt + b(t, X_t) dW_t$, where $W$ is a standard Brownian motion and $X_0=x_0$.

    *This is the interactive [marimo](https://marimo.io) version of
    [simulation_of_sde.ipynb](https://github.com/ramonVDAKKER/teaching-quantitative-finance/blob/develop/notebooks/simulation_of_sde.ipynb);
    it runs entirely in your browser. Use the sliders to explore the effect of the parameters.*
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## 0. Imports""")
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    return np, pd, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Brownian motion, Brownian motion with drift, and Geometric Brownian motion

    The following cell introduces classes to simulate a Brownian motion, a Brownian motion with drift, and a Geometric Brownian motion.
    """)
    return


@app.cell
def _(np, pd, plt):
    class StochasticProcess:
        """Base class for simulating stochastic processes.

        Warning: This class should not be used directly. Use derived classes instead.
        """

        def __init__(self, name, time_grid, paths):
            self.name = name
            self.time_grid = time_grid
            self.paths = paths
            self.params = None

        def plot(self, num_paths=5):
            """Plots (minimum of number of available sample paths and num_paths) sample paths.

            Args:
                num_paths (int): number of sample paths to be plotted. Defaults to 5.
            """

            num_paths = min(self.paths.shape[0], num_paths)
            paths = self.paths[:num_paths, :].T
            title = f"{num_paths} simulated sample paths from {self.name}"
            if self.params is not None:
                title += f" with parameters: {[(k, v) for k, v in self.params.items()]}"
            return pd.DataFrame(
                paths,
                columns=[f"path {j}" for j in range(1, 1 + num_paths)],
                index=self.time_grid,
            ).plot(kind="line", title=title, figsize=(13, 5))

        def avg_and_var_over_simulations(self):
            """Calculates, for each point on time-grid, the mean and variance over sample paths."""

            if self.paths.shape[0] < 2:
                raise ValueError("Requires minimum of two paths.")
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
            aux = np.mean(self.paths, axis=0)
            avg_df = pd.DataFrame(aux, columns=["Average over paths"], index=self.time_grid)
            avg_df.plot(kind="line", title="Average over sample paths", ax=ax1)
            aux = np.var(self.paths, axis=0)
            var_df = pd.DataFrame(
                aux, columns=["Var over sample paths"], index=self.time_grid
            )
            var_df.plot(kind="line", title="Var over sample paths", ax=ax2)
            return avg_df, var_df


    class BrownianMotionWithDrift(StochasticProcess):
        """Class to simulate paths of a Brownian motion with drift, i.e. X_t = ct + W_t"""

        def __init__(self, drift, sigma, n, T, time_step, seed=None):

            if T < time_step:
                raise ValueError("Maturity T should be larger than time step.")
            if sigma < 0:
                raise ValueError("sigma should be nonnegative.")
            self.n = n
            self.T = T
            self.time_grid, self.time_step = np.linspace(
                0, self.T, num=1 + int(T / time_step), endpoint=True, retstep=True
            )  # note that time_step is adapted (if needed) in order to get equally-spaced grid
            if seed is not None:
                np.random.seed(seed)
            aux = np.random.normal(loc=0.0, scale=1.0, size=(self.n, len(self.time_grid) - 1))
            aux = np.concatenate([np.zeros((self.n, 1)), aux], axis=1)
            self.paths = (
                np.cumsum(sigma * np.sqrt(self.time_step) * aux, axis=1) + drift * self.time_grid
            )
            super().__init__("Brownian Motion with drift", self.time_grid, self.paths)
            self.params = {"drift c": drift, "sigma": sigma}


    class BrownianMotion(BrownianMotionWithDrift):
        """Class to simulate paths of a Brownian motion"""

        def __init__(self, sigma, n, T, time_step, seed=None):

            super().__init__(0, sigma, n, T, time_step, seed)
            self.name = "Brownian Motion"
            self.params = {"sigma": sigma}


    class GeometricBrownianMotion(BrownianMotionWithDrift):
        r"""Class to simulate paths of a Geometric Brownian motion, i.e. X_t=X_0\exp((mu-0.5sigma^2)t+sigma*W_t)"""

        def __init__(self, starting_value, mu, sigma, n, T, time_step, seed=None):

            super().__init__(mu - 0.5 * sigma ** 2, sigma, n, T, time_step, seed)
            self.paths = starting_value * np.exp(self.paths)
            self.name = "Geometric Brownian Motion"
            self.params = {"starting_value": starting_value, "mu": mu, "sigma": sigma}
    return BrownianMotion, BrownianMotionWithDrift, GeometricBrownianMotion


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.1 Brownian motion

    The following cell simulates and plots sample paths of a standard Brownian motion on the time-interval $[0, 5]$ where we use a time-step equal to $0.001$. Re-run the cell (press the play button) a few times to get a feeling for the variety of the sample paths.
    """)
    return


@app.cell
def _(BrownianMotion, mo, plt):
    bm = BrownianMotion(sigma=1, n=10, T=5, time_step=0.001, seed=None)
    bm.plot(3)
    _out = mo.vstack([plt.gca(), mo.md("First values of the first three paths:"), bm.paths[0:3, 0:5]])
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""In case we want to be able to reproduce the simulated paths, we should fix the seed. See the following cell.""")
    return


@app.cell
def _(BrownianMotion):
    bm_seeded = BrownianMotion(sigma=1, n=10, T=5, time_step=0.001, seed=42)
    bm_seeded.plot(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We have $\mathbb{E}W_t=0$ and $\operatorname{var}(W_t) = \sigma^2 t$. As a check on our code, we can try to verify these properties. Indeed, if we use a large number of sample paths $n$, then the law of large numbers implies that the sample mean of $W_t^{(i)}$, $i=1,\dots,n$, i.e. the simulated values of the process at time $t$, provides an approximation to $\mathbb{E}W_t=0$. Similarly, the sample variance of $W_t^{(i)}$, $i=1,\dots,n$, provides an approximation to  $\operatorname{var}(W_t)$.

    In the next cell we use $n=2000$ (and a time-step of $0.005$, to keep the in-browser computation light) and compute the sample mean and sample variances of $W_t^{(i)}$, $i=1,\dots,n$, for all $t$ on the time-grid. Please note that, as a next step, we could develop a statistical test.
    """)
    return


@app.cell
def _(BrownianMotion, mo, plt):
    bm_check = BrownianMotion(sigma=3, n=2000, T=5, time_step=0.005, seed=None)
    avg_df, var_df = bm_check.avg_and_var_over_simulations()
    mo.vstack([
        plt.gcf(),
        mo.md("Sample mean and variance at $t=5$ (population values: $0$ and $\\sigma^2 t = 45$):"),
        mo.hstack([avg_df.tail(1), var_df.tail(1)]),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.2 Brownian motion with drift

    In this section we consider a Brownian motion with drift, i.e. $X_t = ct + W_t$, where $W$ is a Brownian motion with variance $\sigma^2$ per unit-of-time.
    """)
    return


@app.cell
def _(BrownianMotionWithDrift):
    bmwd = BrownianMotionWithDrift(drift=0.7, sigma=1, n=10, T=25, time_step=0.001, seed=None)
    bmwd.plot(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1.3 Geometric Brownian motion

    In this section we discuss the Geometric Brownian motion. Recall that this process is described by the SDE $dX_t = \mu X_t dt + \sigma X_t dW_t$, where $X_0=x_0$ and $W$ is a standard Brownian motion. We have seen that this SDE has as solution $$X_t = X_0 \exp\left( (\mu - 0.5\sigma^2) t+ \sigma W_t\right).\qquad(\star)$$ So given a simulated path of $W$ we can obtain the simulated path of $X$.

    Note: in the next section we will discuss the Euler-method. This is a numerical approximation technique which we could have used to simulate, using the SDE, an approximation to $X_t$ in case we would not have been able to derive the closed-form solution $(\star)$.

    Use the sliders to explore the effect of the drift $\mu$ and the volatility $\sigma$ on the sample paths.
    """)
    return


@app.cell
def _(mo):
    gbm_mu = mo.ui.slider(0.0, 0.2, step=0.01, value=0.08, label=r"$\mu$")
    gbm_sigma = mo.ui.slider(0.01, 0.8, step=0.01, value=0.3, label=r"$\sigma$")
    mo.hstack([gbm_mu, gbm_sigma], justify="start", gap=2)
    return gbm_mu, gbm_sigma


@app.cell
def _(GeometricBrownianMotion, gbm_mu, gbm_sigma):
    gbm = GeometricBrownianMotion(
        starting_value=100, mu=gbm_mu.value, sigma=gbm_sigma.value, n=5, T=1, time_step=0.001, seed=None
    )
    gbm.plot(5)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Simulating Ito Diffusion processes using the Euler method

    In this section we consider the simulation of processes that are described by SDEs of the form $$d X_t = a(t,X_t) dt + b(t, X_t) dW_t,$$ with $X_0=x_0$, $W$ a standard Brownian motion and where $a$ and $b$ are deterministic functions.

    In special cases it is possible to obtain a closed-form solution which means that we can express $X_t$ in terms of $W_s$, $s\leq t$ and $t$. As an example, let us consider the GBM $dX_t = \mu X_t dt + \sigma X_t dW_t$, $X_0=x_0$. The solution is given by $X_t = X_0 \exp\left( (\mu - 0.5\sigma^2) t+ \sigma W_t\right)$. So given a simulated path of $W$ we can obtain the simulated path of $X$.

    But how to proceed if we are not able to obtain a closed-form solution to the SDE? In that case we can resort to numerical approximations. Here we will discuss the Euler-method (also known as the Euler–Maruyama method).

    To motivate this method let us recall that the above SDE means that, for $t,h\geq 0$,
    \begin{equation}
    X_{t+h} - X_t  = \int_t^{t+h} a(u, X_u) du + \int_t^{t+h} b(u, X_u) dW_u.
    \end{equation}
    Now insert the approximations, for $u\in [t,t+h]$, $a(u,X_u) \approx a(t, X_t)$ and
    $b(u,X_u)\approx b(t, X_t)$. This yields
    \begin{equation}
    X_{t+h} - X_t  \approx a(t, X_t) h + b(t, X_t) (W_{t+h} -W_t).
    \end{equation}
    This motivates the Euler-method. Simulate an approximation to $X$ via the following recursive scheme. Choose a (small) time-step $h>0$, set $X_0=x_0$, and simulate $X_{(k+1)h}$, for $k\geq 0$, by
    $$X_{(k+1)h} = X_{kh} + a(kh, X_{kh}) h +  b(kh, X_{kh}) \times \epsilon_k,$$
    where $\epsilon_k$, $k\in\mathbb{N}$, are i.i.d. draws from the $N(0,h)$ distribution.

    Warning: if we use the Euler-method for a given SDE, then we do not try or did not succeed to obtain a closed-form solution. However, one should still try to check (using the available sufficient conditions on the functions $a$ and $b$) if a solution to the SDE exists!
    """)
    return


@app.cell
def _(BrownianMotion):
    class ItoDiffusionEuler(BrownianMotion):
        """Class to simulate approximations to solution of SDE dX_t = a(t, X_t) dt + b(t, X_t) dW_t,
        X_0=x_0 and where W is a standard Brownian motion."""

        def __init__(
            self,
            starting_value,
            drift_function,
            volatility_function,
            n,
            T,
            time_step,
            seed=None,
        ):

            super().__init__(1, n, T, time_step, seed)
            self.name = "Euler approximation to Ito diffusion dX_t=a(t,X_t)dt+b(t,X_t)dW_t"
            self.dW = (
                self.paths[:, 1:] - self.paths[:, :-1]
            )  # increments standard Brownian motion
            self.paths[:, 0] = starting_value
            for j in range(1, self.paths.shape[1]):
                previous = self.paths[:, j - 1]
                self.paths[:, j] = (
                    previous
                    + drift_function(self.time_grid[j - 1], previous) * self.time_step
                    + volatility_function(self.time_grid[j - 1], previous)
                    * self.dW[:, j - 1]
                )
            self.params = {"starting_value": starting_value}
    return (ItoDiffusionEuler,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.1 Geometric Brownian motion reconsidered

    $dX_t = \mu X_t dt + \sigma X_t dW_t$, so $a(t,x)=\mu x$ and $b(t,x)=\sigma x$.
    """)
    return


@app.cell
def _(ItoDiffusionEuler):
    euler_mu = 0.07
    euler_sigma = 0.2
    euler_gbm = ItoDiffusionEuler(
        starting_value=100,
        drift_function=lambda t, x: euler_mu * x,
        volatility_function=lambda t, x: euler_sigma * x,
        n=10,
        T=3,
        time_step=0.001,
        seed=None,
    )
    euler_gbm.plot(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2.2 Cox-Ingersoll-Ross

    $dX_t = \alpha(\beta-X_t) dt + \sigma \sqrt{ X_t^+} dW_t$, so $a(t,x)= \alpha(\beta-x)$ and $b(t,x)=\sigma \sqrt{\max\{0,x\}}$.

    Use the sliders to explore the effect of the mean-reversion speed $\alpha$, the long-run level $\beta$, and the volatility $\sigma$.
    """)
    return


@app.cell
def _(mo):
    cir_alpha = mo.ui.slider(-2.0, 2.0, step=0.1, value=1.0, label=r"$\alpha$")
    cir_beta = mo.ui.slider(50, 120, step=1, value=85, label=r"$\beta$")
    cir_sigma = mo.ui.slider(0.01, 0.5, step=0.01, value=0.25, label=r"$\sigma$")
    mo.hstack([cir_alpha, cir_beta, cir_sigma], justify="start", gap=2)
    return cir_alpha, cir_beta, cir_sigma


@app.cell
def _(ItoDiffusionEuler, cir_alpha, cir_beta, cir_sigma, np):
    cir = ItoDiffusionEuler(
        starting_value=100,
        drift_function=lambda t, x: cir_alpha.value * (cir_beta.value - x),
        volatility_function=lambda t, x: cir_sigma.value * np.sqrt(np.maximum(x, 0)),
        n=10,
        T=25,
        time_step=0.01,
        seed=None,
    )
    cir.plot(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Remarks**
    - It can be proved, under parameter conditions $\alpha,\beta\geq 0$ and $2\alpha\beta >\sigma^2$, that the exact solution to the SDE is positive. So $X_t^+$ can be replaced by $X_t$. However, this does not mean that the Euler-method is guaranteed to yield nonnegative paths, so we use $x^+$ in our implementation.
    - Although a closed-form solution of the form $X_t = f( W_s,\, s\leq t)$ is not available for the CIR-process,  the conditional distribution of $X_{t+h}$ given $\mathcal{F}_t$ is known (under the aforementioned parameter conditions). This can be used to obtain simulations from the exact solution to the SDE (on a (discrete) time-grid).
    """)
    return


if __name__ == "__main__":
    app.run()
