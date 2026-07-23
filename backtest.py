# backtest.py

import numpy as np
import pandas as pd
from model import (arrival_probability, el_aoud_quotes,
                   el_aoud_quotes_risk_averse)


def run_backtest(strategy_df, lambda_params,
                 B_param, n_sims=1000,
                 zi_spread_pct=0.02,
                 dt_minutes=15,
                 strategy='el_aoud',
                 heston_p = None,
                 epsilon = 0.1) -> tuple:
    """
    Simulate market making strategy on test period.

    At each 15-minute timestamp:
        1. Compute optimal quotes
        2. Simulate order arrivals (Poisson process)
        3. Update inventory, cash, delta hedge
        4. Record wealth and P&L

    Parameters:
        strategy_df   : DataFrame  per-timestamp strategy inputs
        lambda_params : dict       calibrated intensity params
        B_param       : float      El Aoud B parameter (% units)
        n_sims        : int        number of Monte Carlo simulations
        zi_spread_pct : float      zero-intelligence spread (% of mid)
        dt_minutes    : float      interval length in minutes
        strategy      : str        'el_aoud' | 'el_aoud_risk_averse'
                                   | 'zero_intelligence'

    Returns:
        terminal_wealth : np.array  shape (n_sims,)
        path_df         : DataFrame single simulation path (sim 0)
    """
    terminal_wealth = np.zeros(n_sims)
    final_inventories  = np.zeros(n_sims)  # ← add this
    path_df = None

    for sim in range(n_sims):
        q1   = 0.0
        q2   = 0.0
        cash = 0.0
        path = []

        for _, row in strategy_df.iterrows():
            C_Q   = row['C_Q']
            S     = row['S']
            delta = row['delta']
            M0    = row['M0']
            v_t   = row['v_t']
            T     = row['T']

            # ── compute quotes ──────────────────────────────
            if strategy == 'el_aoud':
                dp, dm = el_aoud_quotes(
                    M0=M0, C_Q=C_Q, B_param=B_param
                )

            # in backtest.py, pass epsilon
            elif strategy == 'el_aoud_risk_averse':
                dp, dm = el_aoud_quotes_risk_averse(
                    M0=M0, C_Q=C_Q, q1=q1,
                    delta=delta, v_t=v_t, S=S, T=T,
                    B_param=B_param,
                    heston_p=heston_p,
                    epsilon=epsilon
                )

            else:  # zero_intelligence
                dp = C_Q * zi_spread_pct
                dm = C_Q * zi_spread_pct

            # ── simulate fills ───────────────────────────────
            p_ask = arrival_probability(
                dp, C_Q, lambda_params, dt_minutes
            )
            p_bid = arrival_probability(
                dm, C_Q, lambda_params, dt_minutes
            )

            fill_ask = np.random.random() < p_ask
            fill_bid = np.random.random() < p_bid

            spread_income = 0.0

            if fill_ask:
                cash          += C_Q + dp
                q1            -= 1
                spread_income += dp

            if fill_bid:
                cash          -= C_Q - dm
                q1            += 1
                spread_income += dm

            # ── delta hedge rebalance ────────────────────────
            q2_new     = -q1 * delta
            hedge_cost = (q2_new - q2) * S
            cash      -= hedge_cost
            q2         = q2_new

            # ── total wealth (cash + option + stock hedge) ───
            W = cash + q1 * C_Q + q2 * S

            if sim == 0:
                path.append({
                    'datetime'     : row['datetime'],
                    'C_Q'          : C_Q,
                    'S'            : S,
                    'M0'           : M0,
                    'v_t'          : v_t,
                    'T'            : T,
                    'delta'        : delta,
                    'delta_plus'   : dp,
                    'delta_minus'  : dm,
                    'fill_ask'     : int(fill_ask),
                    'fill_bid'     : int(fill_bid),
                    'q1'           : q1,
                    'q2'           : q2,
                    'cash'         : cash,
                    'spread_income': spread_income,
                    'hedge_cost'   : hedge_cost,
                    'W'            : W
                })

        # terminal wealth includes all positions
        last_row = strategy_df.iloc[-1]
        terminal_wealth[sim] = (
            cash +
            q1 * last_row['C_Q'] +
            q2 * last_row['S']
        )
        final_inventories[sim] = q1

        # in run_backtest, after the inner loop:

        if sim == 0:
            path_df = pd.DataFrame(path)

    
    return terminal_wealth, final_inventories, path_df


def compute_metrics(terminal_wealth, path_df, label='') -> dict:

    W  = path_df['W'].values

    # terminal wealth distribution
    total_pnl  = terminal_wealth.mean()
    median_pnl = np.median(terminal_wealth)
    pnl_std    = terminal_wealth.std()
    pnl_skew   = pd.Series(terminal_wealth).skew()
    pnl_kurt   = pd.Series(terminal_wealth).kurt()

    # drawdown
    running_max = np.maximum.accumulate(W)
    max_dd      = (running_max - W).max()

    # fill statistics
    fills       = path_df['fill_ask'] + path_df['fill_bid']
    fill_rate   = fills.mean()
    total_fills = fills.sum()

    nonzero    = path_df['spread_income'][path_df['spread_income'] > 0]
    avg_spread = nonzero.mean() if len(nonzero) > 0 else 0.0

    # inventory statistics
    mean_inv   = path_df['q1'].abs().mean()
    max_inv    = path_df['q1'].abs().max()
    final_inv  = path_df['q1'].iloc[-1]
    inv_std    = path_df['q1'].std()

    # spread income
    total_spread = path_df['spread_income'].sum()

    if label:
        print(f"\n{'='*55}")
        print(f"  {label}")
        print(f"{'='*55}")
        print(f"  Mean terminal PnL:       ${total_pnl:>12.2f}")
        print(f"  Median terminal PnL:     ${median_pnl:>12.2f}")
        print(f"  PnL std:                 ${pnl_std:>12.2f}")
        print(f"  PnL skewness:            {pnl_skew:>12.4f}")
        print(f"  PnL kurtosis:            {pnl_kurt:>12.4f}")
        print(f"  Max drawdown:            ${max_dd:>12.2f}")
        print(f"  Total fills:             {total_fills:>12.0f}")
        print(f"  Fill rate per step:      {fill_rate:>12.4f}")
        print(f"  Avg spread per fill:     ${avg_spread:>12.2f}")
        print(f"  Mean |inventory|:        {mean_inv:>12.4f}")
        print(f"  Max |inventory|:         {max_inv:>12.0f}")
        print(f"  Final inventory:         {final_inv:>12.0f}")
        print(f"  Inventory std:           {inv_std:>12.4f}")
        print(f"  Total spread income:     ${total_spread:>12.2f}")

    return {
        'total_pnl'          : total_pnl,
        'median_pnl'         : median_pnl,
        'pnl_std'            : pnl_std,
        'pnl_skew'           : pnl_skew,
        'pnl_kurt'           : pnl_kurt,
        'max_drawdown'       : max_dd,
        'total_fills'        : total_fills,
        'fill_rate'          : fill_rate,
        'avg_spread'         : avg_spread,
        'mean_inventory'     : mean_inv,
        'max_inventory'      : max_inv,
        'final_inventory'    : final_inv,
        'inventory_std'      : inv_std,
        'total_spread_income': total_spread
    }

def compare_strategies(*args) -> pd.DataFrame:

    metrics_list = [
    ('Mean Terminal PnL ($)',  'total_pnl',           '${:.2f}'),
    ('Median Terminal PnL ($)','median_pnl',          '${:.2f}'),
    ('PnL Std ($)',            'pnl_std',             '${:.2f}'),
    ('PnL Skewness',           'pnl_skew',            '{:.4f}'),
    ('PnL Kurtosis',           'pnl_kurt',            '{:.4f}'),
    ('Max Drawdown ($)',       'max_drawdown',        '${:.2f}'),
    ('Total Fills',            'total_fills',         '{:.0f}'),
    ('Fill Rate per Step',     'fill_rate',           '{:.4f}'),
    ('Avg Spread/Fill ($)',    'avg_spread',          '${:.2f}'),
    ('Mean |Inventory|',       'mean_inventory',      '{:.4f}'),
    ('Max |Inventory|',        'max_inventory',       '{:.0f}'),
    ('Final Inventory',        'final_inventory',     '{:.0f}'),
    ('Inventory Std',          'inventory_std',       '{:.4f}'),
    ('Spread Income ($)',      'total_spread_income', '${:.2f}'),
]

    rows = []
    for display_name, key, fmt in metrics_list:
        row = {'Metric': display_name}
        for metrics, label in args:
            row[label] = fmt.format(metrics[key])
        rows.append(row)

    return pd.DataFrame(rows)