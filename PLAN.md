# Course improvement plan — Quantitative Finance (BSc EOR, Tilburg)

A living plan for improving the lecture material in this repository. It records the
intended narrative arc, a **notation standard** to enforce across decks, the status of
enhancements, and a prioritised backlog.

> Scope: the valuation/hedging block taught from these slides — the options recap, the
> three valuation decks, the hedging deck, and the Monte-Carlo-Greeks deck — plus the
> companion notebooks.

---

## 1. Course narrative (the spine)

The material tells one story: **how to price and hedge a European option, and why it works.**

| # | Deck | Role in the story |
|---|------|-------------------|
| 0 | `slides/0. options recap` | Prerequisites: what an option is, payoffs, moneyness, time value. |
| 1 | `A. valuation part 1` | The idea, in a one-/multi-period **binomial toy model**: replication, risk-neutral pricing, pricing kernel. |
| 2 | `A. valuation part 2` | Continuous time, method 1: the **Black-Scholes PDE** (+ Feynman-Kac, finite differences). |
| 3 | `A. valuation part 3` | Continuous time, methods 2-3: **risk-neutral pricing / Girsanov** and the **pricing kernel**; completeness. |
| B | `B. hedging and risk management` | Using the price: **hedging** with the Greeks; the gamma-theta P&L. |
| C | `C. MC methods for greeks` | **Computing the Greeks** by simulation when there is no closed form. |

**Through-lines to make explicit in every deck** (most now are):
- *Three methods, one price* — replication / risk-neutral / kernel agree (toy model -> Parts 2-3 capstones).
- *Discrete <-> continuous* — binomial `q` <-> market price of risk `gamma`; binomial hedge ratio <-> delta; explicit FD scheme <-> trinomial tree.
- *Error scales like `1/sqrt(n)`* — discrete-hedging error (deck B) <-> Monte-Carlo error (deck C).

---

## 2. Notation standard (enforce across all decks)

Several inconsistencies exist across (and within) decks. Target conventions:

| Object | Standard symbol | Notes / current offenders |
|--------|-----------------|---------------------------|
| Stock price process | `S_t` (capital) | Part 1 toy model originally mixed `s_0`/`S_0` (fixed). Realised outcomes stay lowercase `s_u, s_d`. |
| Bond / MMA | `B_t`, `B_0=1` | consistent. |
| Option price (process) | `C_t = F(t,S_t)` | Keep `F` for the *pricing function*, `C_t` for its value. |
| Generic portfolio value | `V_t = G(t,S_t)` | deck B uses `G`; keep. |
| **Payoff function** | **`h(.)`** | **MAIN FIX:** deck C uses `h`, `f`, and `F` for the payoff on different slides. Standardise on `h` for the payoff, reserve `f` for a generic smooth function, `F` for the *price*. |
| Payoff at maturity | `C_T = h(S_T)` | consistent in 1-3; align deck C. |
| Greeks | `Delta, Gamma, nu, rho, Theta` | deck C recap table should show the *symbols*, not just names; add `nu` for vega. |
| Volatilities | `sigma` (model), `sigma_imp`, `sigma_real` | only deck B (gamma-theta) currently distinguishes; fine. |
| MC parameters | `theta` (differentiated), `eta` (other dist. parameter), `gamma` (payoff parameter) | **FIX:** deck C's CRN slide drops `gamma` and reuses `eta` in the payoff argument; make payoff `h(.;theta,gamma)` everywhere. |
| Indicator | `1{.}` | consistent (recap deck generalised it). |
| `d` (differential), `e` | `\rd`, `\e` (from `commands.txt`) | decks vary (`\exp` vs `\e`); harmless, leave unless a deck mixes both on one slide. |

**Action:** a dedicated *notation pass* per deck (low-risk; do on each deck's branch so each PR stays self-consistent). Highest priority: deck C payoff symbol + `theta,eta,gamma`.

---

## 3. Status of enhancements (per deck)

Legend: [x] done (in an open PR), [~] partly, [ ] planned.

### Options recap (PR #6/#7)
- [x] Standalone improved deck (definitions, payoff diagrams, payoff-vs-profit, moneyness, exercise styles, AEX examples, time value, quiz).

### Valuation Part 1 (PR #6)
- [x] Worked examples (1-period, 3 methods agree), multi-period tree + backward induction, pricing kernel, payoff diagrams, quiz/FFT, appendix proof, binomial notebook.

### Valuation Part 2 (PR #8)
- [x] Call-price visual, verify-a-solution example, binomial<->delta, finite differences (+ stability<->no-arbitrage), Feynman-Kac sketch, heat-equation reduction, uniqueness/max-principle, FD notebook extension, quiz/FFT.

### Valuation Part 3 (PR #9)
- [x] Roadmap, P-vs-Q density visual, "stock earns r under Q", discrete-q->market-price-of-risk, Girsanov proof -> appendix, kernel<->Part 1, duality diagram, "three methods one price" capstone, MC notebook, quiz/FFT.

### Hedging - deck B (PR #10)
- [x] gamma-theta P&L (realised-vs-implied vol bet), Delta(s)/Gamma(s) visuals, bridge to C, quiz/FFT, notation fix.
- [x] hedge-error vs rebalancing-frequency slide; worked delta-gamma-vega hedge; transaction-cost remark; "From the Greeks to Value-at-Risk" slide; extended `illustration_discrete_time_hedging.ipynb` (error vs n + transaction-cost U-curve).

### MC Greeks - deck C (PR #10)
- [x] MSE-vs-h U-curve, "three methods at a glance" capstone, quiz/FFT.
- [x] `theta,eta,gamma` parameter fixed; recap-table Greek symbols; led with the concrete vega example; three-estimator variance comparison added to `illustration_mc_methods_for_greeks.ipynb`. [ ] residual: the payoff symbol still varies (h/f/F) across slides -- a convention call for the instructor.

---

## 4. Prioritised backlog (after the open PRs merge)

**P0 - correctness & consistency** -- DONE
1. [x] Cross-deck notation pass: price unified to capital `S_0/S_t` (Parts 1-3); `F_s` lowercase (Part 2); `W^P` on kernel slides + `B_0=1` (Part 3); deck-C parameters + recap symbols.
2. [x] Reviewer agent run on every deck (Parts 1, 2, 3 and B, C): **all mathematics verified correct**; flagged notation items fixed.
   - residual (instructor's call): the payoff-function symbol in deck C still varies (h abstract / f generic / F option) -- defensible but could be unified to `h`.

**P1 - high-value teaching content** -- DONE
3. [x] Deck B hedge-error vs rebalancing-frequency slide + notebook (error vs n, transaction-cost U-curve).
4. [x] Deck C three-estimator variance comparison in the notebook (call vs digital; bump/pathwise/LRM).
5. [x] Deck B worked delta-gamma-vega hedge (the 2x2 system).

**P2 - polish & extensions** -- DONE (except optional pointers)
6. [x] Course-map slide added to Part 1.
7. [x] Appendix convention: Parts 1 & 3 use `\appendix`; Part 2 needs none (no interrupting proof).
8. [x] Deck B "From the Greeks to Value-at-Risk" slide.
9. [~] Advanced pointers partly added as food-for-thought (autodiff/Malliavin in C; transaction costs in B); further pointers optional.

---

## 5. Notebooks

| Notebook | Status |
|----------|--------|
| `binomial_option_pricing.ipynb` | [x] new (Part 1 companion). |
| `black_scholes_pde_numerical_approximation.ipynb` | [x] extended (explicit/trinomial scheme + stability). |
| `risk_neutral_pricing.ipynb` | [x] new (Part 3 companion: P-vs-Q, MC vs closed form). |
| `illustration_discrete_time_hedging.ipynb` | [x] extended: hedge error vs frequency (1/sqrt(n)) + transaction-cost U-curve. |
| `illustration_mc_methods_for_greeks.ipynb` | [x] extended: bump/pathwise/LRM compared on call & digital delta. |
| `illustration_black_scholes_price.ipynb`, `simulation_of_sde.ipynb` | [x] reviewed -- clean (no notation slip). |

All new notebook code is `numpy`+`matplotlib` only (no SciPy) and asserts against the slide numbers.

---

## 6. Working conventions

- Feature branch + PR per deck; visuals built natively in **TikZ/pgfplots**; quiz/food-for-thought via shared **tcolorbox** environments; **quiz answers are not shown** on slides.
- Every deck compiles with `pdflatex -shell-escape`; the unused `pstricks` package was removed everywhere.
- Verify each new/changed slide by rendering it to PNG and eyeballing; check for overfull boxes.
