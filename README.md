# Does GDP per Capita Causally Affect Life Expectancy?

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyMC](https://img.shields.io/badge/PyMC-5.x-black.svg)](https://www.pymc.io/)
[![ArviZ](https://img.shields.io/badge/ArviZ-0.1x-orange.svg)](https://python.arviz.org/)
[![Dataset](https://img.shields.io/badge/data-Gapminder-green.svg)](https://www.gapminder.org/data/)

A Bayesian causal analysis of the Gapminder dataset, estimating the effect of GDP per capita on
life expectancy while treating population as a potential confounder.

**Course:** CS-GY 6053 Foundations of Data Science (Fall 2024), NYU Tandon — Prof. Reeves
**Team:** David Hong · Yanka Sikder · Farhen Shefa

> Note: the notebook is large (~4 MB) because every plot is embedded. If GitHub fails to render it,
> open it through [nbviewer](https://nbviewer.org/github/citypooh/Bayesian_Causal_Analysis_Gapminder/blob/main/Final_Project.ipynb)
> or read [`Final_Project.pdf`](Final_Project.pdf).

---

## Overview

**Research question:** Does higher GDP per capita causally increase life expectancy?

Correlation between wealth and longevity is easy to show, but a correlation alone cannot separate
the effect of GDP from the effect of things that drive both. This project builds an explicit causal
model, encodes it as a Bayesian regression, validates the estimator on simulated data with known
parameters, and only then fits it to the real Gapminder panel.

**Headline result:** GDP per capita has a strong positive effect on life expectancy
(β_gdp = 0.968, 89% HDI [0.932, 1.011] in standardized units), and population has only a small
effect (β_pop = 0.126) once GDP is controlled for — so population is a weak confounder here, not a
driver of the GDP–longevity relationship.

---

## Repository Structure

| File | Description |
| --- | --- |
| [`Final_Project.ipynb`](Final_Project.ipynb) | Main analysis notebook — EDA, prior predictive simulation, model comparison, simulation-based validation, posterior inference, and posterior predictive checks |
| [`Final_Project.pdf`](Final_Project.pdf) | Rendered notebook (all code, plots, and written analysis) |
| [`Final_Project_Proposal.pdf`](Final_Project_Proposal.pdf) | Initial proposal — question, data, DAG, and planned statistical model |
| [`gapminder.csv`](gapminder.csv) | Dataset (1,704 rows × 6 columns) |
| [`data.py`](data.py) | statsmodels-style loader for the Gapminder CSV, from the `causaldata` package |

---

## Data

[Gapminder](https://www.gapminder.org/data/) (via Jennifer Bryan's R `gapminder` package, packaged
for Python in [`causaldata`](https://github.com/NickCH-K/causaldata)).

- **1,704 observations** — 142 countries × 12 time points (1952–2007, every 5 years)
- **6 variables** — `country`, `continent`, `year`, `lifeExp`, `pop`, `gdpPercap`

The assignment allowed three variables, so the analysis uses:

| Role | Variable | Description |
| --- | --- | --- |
| Treatment | `gdpPercap` | GDP per capita (US$, inflation-adjusted) |
| Outcome | `lifeExp` | Life expectancy at birth, in years |
| Confounder | `pop` | Total population |

---

## Causal Model

```mermaid
graph LR
    POP[Population] --> GDP[GDP per capita]
    POP --> LIFE[Life Expectancy]
    GDP --> LIFE
    style GDP fill:#cfe2ff,stroke:#0d6efd
    style LIFE fill:#d1e7dd,stroke:#198754
    style POP fill:#fff3cd,stroke:#ffc107
```

Wealthier countries tend to have better healthcare, nutrition, and clean water, so GDP per capita
should raise life expectancy. Population plausibly opens a back-door path: a larger workforce can
shift economic productivity per person, and more people competing for the same healthcare and food
can depress longevity. Population therefore has to be in the model to identify GDP's direct effect.

**Identifying assumptions:** GDP affects life expectancy and not the reverse, and conditioning on
population is sufficient to block confounding.

---

## Method

The workflow follows the six-step structure required by the course.

### 1. Preprocessing

`lifeExp` is close to symmetric, so it is only standardized. `gdpPercap` and `pop` are both heavily
right-skewed across several orders of magnitude, so each is log-transformed *and then* standardized.
Comparing the original, standardized, and log-scaled histograms side by side made the choice
explicit rather than assumed: log-scaling pulls both skewed variables toward normality, and the
subsequent standardization puts all three variables on one interpretable scale.

After transformation, log GDP vs. life expectancy shows a clear positive linear trend, while
population vs. life expectancy and population vs. GDP stay diffuse and non-linear — consistent with
population acting as a messy confounder rather than a clean predictor.

### 2. Prior Predictive Simulation

Five prior configurations — from *Very Conservative* to *Weakly Informative* — were varied
simultaneously (rather than one parameter at a time) and their implied regression lines plotted.
Tight priors produced nearly identical lines that would ignore the data; loose priors produced
implausibly wild slopes. The **Balanced** setting was selected:

```
α       ~ Normal(0, 0.6)
β_gdp   ~ Normal(0.5, 0.3)     # positive but not dogmatic
β_pop   ~ Normal(0, 0.3)       # agnostic about direction
σ       ~ Exponential(1)
```

It keeps the substantive belief that GDP helps, while leaving enough room for the data to move the
posterior.

### 3. Model Comparison

**Which variables?** Three nested models were compared with PSIS-LOO and WAIC (deviance scale —
lower is better):

| Model | elpd_loo | elpd_diff | elpd_waic |
| --- | --- | --- | --- |
| **Direct Effect** (GDP + population) | **2915.09** | — | **2915.08** |
| Total Effect (GDP only) | 3041.29 | 126.20 | 3041.28 |
| Intercept-only | 4838.12 | 1923.03 | 4838.12 |

GDP alone improves enormously over the intercept-only baseline, and adding population improves the
fit further — the confounder belongs in the model.

**Which functional form?** Linear, quadratic, and cubic versions of the direct-effect model:

| Model | elpd_loo | elpd_diff | elpd_waic |
| --- | --- | --- | --- |
| **Cubic** | **2851.69** | — | **2851.64** |
| Quadratic | 2901.96 | 50.27 | 2901.94 |
| Linear | 2914.86 | 63.17 | 2914.85 |

The cubic model wins on both criteria. Higher-order terms received progressively tighter priors
(σ = 0.3 → 0.2 → 0.1) as regularization, since polynomial terms grow fast and destabilize sampling.

### 4. Validation on Simulated Data

Before touching the real data, 1,000 observations were simulated from known parameters
(α = 0, β_gdp = 0.5, all higher-order terms = 0, σ = 1) and the cubic model was fit to them. The
model recovered every parameter: α near 0, β_gdp near 0.5, all higher-order coefficients near 0.
Rank plots were uniform, r̂ < 1.01, and ESS exceeded 1,000 — the estimator works before it is
trusted on real data.

### 5. Posterior Inference

Final model — Bayesian cubic regression, NUTS via PyMC (500 draws, 250 tuning, seed 42):

```
lifeExp_std ~ Normal(μ, σ)
μ = α + β_gdp·GDP + β_gdp2·GDP² + β_gdp3·GDP³
      + β_pop·POP + β_pop2·POP² + β_pop3·POP³
```

| Parameter | Mean | SD | 89% HDI | ESS (bulk) | r̂ |
| --- | --- | --- | --- | --- | --- |
| α | 0.024 | 0.021 | [−0.007, 0.058] | 1851 | 1.00 |
| **β_gdp** | **0.968** | 0.025 | **[0.932, 1.011]** | 1616 | 1.00 |
| β_gdp² | −0.024 | 0.013 | [−0.045, −0.003] | 1844 | 1.00 |
| β_gdp³ | −0.079 | 0.011 | [−0.096, −0.062] | 1606 | 1.00 |
| **β_pop** | **0.126** | 0.022 | **[0.090, 0.160]** | 1665 | 1.00 |
| β_pop² | 0.009 | 0.009 | [−0.006, 0.021] | 1933 | 1.00 |
| β_pop³ | 0.008 | 0.005 | [−0.001, 0.015] | 1773 | 1.00 |
| σ | 0.557 | 0.009 | [0.542, 0.572] | 2099 | 1.00 |

All chains converged cleanly (r̂ = 1.00 everywhere, uniform rank plots, ESS well above 1,000).

### 6. Posterior Predictive Check

Observed data, posterior mean line, HDI of the mean, and HDI of individual predictions were plotted
together. The posterior mean tracks the observed trend closely; the intervals widen at both extremes
of the GDP range, exactly where observations are sparse.

---

## Findings

1. **GDP per capita strongly raises life expectancy.** A one-standard-deviation increase in log GDP
   per capita is associated with a **0.968 SD increase in life expectancy**, with an 89% HDI far
   from zero.
2. **The relationship is mostly linear.** The cubic model fits best, but the curvature is small
   (β_gdp² = −0.024, β_gdp³ = −0.079) — mild diminishing returns at the high end rather than a
   fundamentally non-linear relationship.
3. **Population is a weak confounder.** Conditioning on population improves model fit, but its own
   effect is small (β_pop = 0.126) and its higher-order terms are indistinguishable from zero. It
   does not explain away GDP's effect.

**Conclusion:** the data support a robust positive effect of GDP per capita on life expectancy that
survives adjustment for population.

---

## Limitations & Future Work

- **Model form.** A Student-t likelihood would be more robust to outliers; other non-linear forms
  (splines, GPs) may capture the curvature better than a cubic polynomial.
- **Omitted variables.** GDP is a coarse proxy for the mechanisms that actually extend life. Adding
  child mortality, healthcare spending, or education would sharpen the causal story — child
  mortality in particular should map almost directly onto life expectancy.
- **Structure in the data.** The panel is ignored: country and continent effects and the time
  dimension (1952–2007) are all pooled away. Hierarchical or fixed-effects models over
  country/region, and explicit handling of within-country trends over time, are the natural next
  step.

---

## Running the Analysis

Requires Python 3.9+.

```bash
pip install numpy pandas matplotlib seaborn scipy networkx scikit-learn pymc arviz
jupyter notebook Final_Project.ipynb
```

The notebook pulls the dataset directly from the `causaldata` repository, so no local data setup is
needed — [`gapminder.csv`](gapminder.csv) and [`data.py`](data.py) are included for offline use and
reproducibility. Sampling is seeded (`rnd_seed = 42`) and the full run takes a few minutes on a
laptop.

---

## References

- Gapminder Foundation — <https://www.gapminder.org/data/>
- Bryan, J. (2017). *gapminder: Data from Gapminder.* R package version 0.3.0.
- Huntington-Klein, N. *causaldata* — <https://github.com/NickCH-K/causaldata>
- Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016). *Probabilistic programming in Python using PyMC3.*
- Kumar, R. et al. (2019). *ArviZ: a unified library for exploratory analysis of Bayesian models in Python.*
