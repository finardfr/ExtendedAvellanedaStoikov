# Avellaneda-Stoikov Market Making Simulator

This repository implements an HFT Market Making simulator based on the **Avellaneda-Stoikov (2008)** framework. It compares an **Optimal Inventory Strategy** against a **Symmetric Baseline** across different volatility regimes.

![PnL Path](img/fig1_pnl_path.png)

---

## Key Features

* **Optimal Quotes**: Dynamic adjustment of reservation price and spread based on inventory ($q$) and time ($T-t$).
* **Monte Carlo Engine**: 1,000 simulations per scenario (Low, Medium, High volatility).
* **Realistic Dynamics**: U-shaped intraday order arrival ($\alpha(t)$) and Gamma-distributed fill sizes.
* **Analysis**: Automatic generation of P&L paths, inventory distributions, and Sharpe ratio tables.

## Requirements

```bash
pip install numpy matplotlib scipy

```

## How to Run

1. Clone the repository.
2. Run the simulation:
```bash
python market_maker.py

```

3. Check the `img/` folder for generated performance plots (PDF).

## Mathematical Core

The optimal reservation price $r$ is calculated as:


$$r(s, t, q) = s - q\gamma\sigma^2(T-t)$$

Where:

* **$s$**: Mid-price
* **$\gamma$**: Risk aversion
* **$q$**: Inventory
* **$\sigma$**: Volatility

---
