# Optimal Market Making for Cryptocurrency Options

Optimal market making for Bitcoin options using the El Aoud & Abergel (2015) stochastic control framework, with Avellaneda-Stoikov (2008) as a comparison benchmark. Implemented and backtested on OKX BTC options data (May 2026).

## Overview

A market maker posts bid and ask quotes continuously, earning the spread on each fill while managing inventory risk. For equity market making, Avellaneda & Stoikov (2008) provides the canonical solution. For options, the problem is fundamentally harder — the asset being quoted is nonlinear, time-decaying, and sensitive to volatility. This project implements the El Aoud & Abergel (2015) framework, which is purpose-built for options market making.

The core insight is the **M0 mispricing signal** — the gap between the option price under the risk-neutral measure Q (what the market prices) and the physical measure P (what the asset actually does). When M0 > 0 the option is overpriced, and the market maker should lean short by posting a wide ask and tight bid.

---

## Notebooks

| Notebook | Description |
|---|---|
| [`elaoud-abergel.ipynb`](elaoud-abergel.ipynb) | Full El Aoud & Abergel pipeline — calibration, strategy, Monte Carlo backtest, event-driven backtest, visualizations |
| [`avellaneda-stoikov.ipynb`](avellaneda-stoikov.ipynb) | AS model applied to the same contract — lambda calibration, simulation, event-driven backtest, comparison with El Aoud |

---

## Contract

`BTC-USD-260626-80000-C` — June 2026 call, $80,000 strike

| Period | Dates |
|---|---|
| Training | May 1–10, 2026 |
| Test | May 11–15, 2026 |

---

## Calibration Pipeline

```
OKX orderbook data (4 expiries)
    ↓
Forward price via put-call parity
    ↓
Implied volatility surface (Black-Scholes inversion)
    ↓
SVI calibration — smooth arbitrage-free surface
    ↓
Heston Q calibration — risk-neutral dynamics
    ↓
Heston P calibration — physical dynamics (1-year daily BTC data)
    ↓
M0 = C_Q - C_P  (mispricing signal)
    ↓
Lambda calibration — order arrival intensity from training trades
    ↓
El Aoud optimal quotes
```

### Heston Q Parameters
| Parameter | Value |
|---|---|
| v0 | 0.1995 |
| κ | 4.99 |
| θ | 0.202 |
| η | 1.308 |
| ρ | -0.685 |
| RMSE | 0.054 |

### Heston P Parameters
| Parameter | Value |
|---|---|
| v0 | 0.059 |
| κ | 6.66 |
| θ | 0.119 |
| η | 0.708 |
| ρ | -0.038 |

### Lambda Calibration (El Aoud)
| Parameter | Value |
|---|---|
| A | 176.53 |
| B | 11.78 |
| R² | 0.703 |

### Lambda Calibration (AS)
| Parameter | Value |
|---|---|
| A_sym | 0.001500 |
| κ_sym | 28.9419 |

---

## Strategies

| Strategy | Notebook | Description |
|---|---|---|
| **El Aoud RN** | El Aoud | Risk-neutral optimal quotes driven by M0 signal |
| **El Aoud RA** | El Aoud | Risk-neutral + additive inventory penalty (ε = 0.1) |
| **Zero-Intelligence** | El Aoud | Fixed 2% symmetric spread — naive benchmark |
| **AS Optimal** | AS | Avellaneda-Stoikov with inventory-adjusted reservation price |
| **AS Symmetric** | AS | Avellaneda-Stoikov symmetric quotes, no inventory adjustment |

---

## Results

### Monte Carlo Backtest (1000 simulations, May 11–15)

| Strategy | Mean PnL | PnL Std | Total Fills | Mean \|Inventory\| | Max Drawdown |
|---|---|---|---|---|---|
| El Aoud RN | $56,695 | $2,497 | 335 | 145.1 | $163,718 |
| El Aoud RA | $48,058 | $2,755 | 246 | 22.5 | $12,714 |
| Zero-Intelligence | $44,460 | $2,765 | 488 | 6.7 | $5,467 |

El Aoud RN achieves the highest mean PnL by aggressively exploiting the persistently positive M0 signal. El Aoud RA sacrifices ~15% PnL for dramatically better inventory control (mean inventory 22 vs 145).

### Event-Driven Backtest (82 actual trades, May 11–15)

| Strategy | Final PnL | Fills | Mean \|Inventory\| | Max \|Inventory\| |
|---|---|---|---|---|
| El Aoud RN | $6,292 | 29 | 11.4 | — |
| El Aoud RA | $1,654 | 9 | 5.3 | — |
| AS Optimal | $36,963 | 92 | 12.3 | 26 |
| AS Symmetric | $32,753 | 256 | 69.0 | 177 |

AS Optimal outperforms El Aoud on real trades because El Aoud's M0-driven spread (~$30–80 above mid) is too wide for a thin options market — most real trades do not cross the threshold. AS quotes near mid and catches almost every trade, earning ~$402 per fill vs El Aoud's ~$217.

### Key Finding

El Aoud's M0 signal is theoretically correct and dominates in simulation. In a thin real market, wide spreads miss fills. The practical solution is El Aoud RA — which balances signal exploitation with inventory control. AS Optimal is a strong benchmark for thin markets where fill rate matters more than spread per fill.

---

## File Structure

```
├── elaoud-abergel.ipynb    # El Aoud pipeline
├── avellaneda-stoikov.ipynb # AS comparison
├── extractor.py             # OKX data loading and parsing
├── utils.py                 # BS pricing, Greeks, IV inversion
├── surface.py               # SVI calibration, Heston Q/P, arbitrage checks
├── model.py                 # El Aoud strategy — quotes, lambda, backtest
└── backtest.py              # Monte Carlo simulation engine
```

---

## Data

- **Orderbook:** OKX BTC options orderbook snapshots (15-minute), 4 expiries, May 1–15 2026
- **Trades:** OKX BTC options trade tape, May 2026
- **Spot:** OKX BTC-USDT 15-minute candles (May 2026) and daily candles (1 year)

Data is cached as parquet files after first extraction.

---

## Dependencies

```
numpy
pandas
scipy
torch
matplotlib
sortedcontainers
```

---

## References

- El Aoud, S. & Abergel, F. (2015). *A stochastic control approach to option market making.* Market Microstructure and Liquidity.
- Avellaneda, M. & Stoikov, S. (2008). *High-frequency trading in a limit order book.* Quantitative Finance.
- Gatheral, J. (2004). *A parsimonious arbitrage-free implied volatility parameterization with application to the valuation of volatility derivatives.* (SVI)
- Heston, S. (1993). *A closed-form solution for options with stochastic volatility.* Review of Financial Studies.
