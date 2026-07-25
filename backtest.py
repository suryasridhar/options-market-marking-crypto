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
                 heston_p=None,
                 epsilon=0.1,
                 seed=42) -> tuple:
    """
    Simulate market making strategy on test period.

    At each 15-minute timestamp:
        1. Compute optimal quotes
        2. Simulate order arrivals (Poisson process)
        3. Update inventory, cash, delta hedge
        4. Record wealth and P&L

    Arrival draws are pre-generated from `seed`, so passing the same seed to
    every strategy gives common random numbers (paired comparison).

    Path metrics (drawdown, fills, inventory, spread income) are accumulated
    across ALL sims, not read off sim 0.

    Parameters:
        strategy_df   : DataFrame  per-timestamp strategy inputs
        lambda_params : dict       calibrated intensity params
        B_param       : float      El Aoud B parameter (% units)
        n_sims        : int        number of Monte Carlo simulations
        zi_spread_pct : float      zero-intelligence spread (% of mid)
        dt_minutes    : float      interval length in minutes
        strategy      : str        'el_aoud' | 'el_aoud_risk_averse'
                                   | 'zero_intelligence'
        seed          : int        RNG seed

    Returns:
        terminal_wealth   : np.array  shape (n_sims,)
        final_inventories : np.array  shape (n_sims,)
        path_df           : DataFrame single simulation path (sim 0)
        stats             : dict of np.array (n_sims,) per-sim path metrics
    """
    # ── hoist columns out of the inner loop ──────────────────────────────
    dts     = strategy_df['datetime'].to_numpy()
    C_Q_a   = strategy_df['C_Q'].to_numpy(dtype=float)
    S_a     = strategy_df['S'].to_numpy(dtype=float)
    delta_a = strategy_df['delta'].to_numpy(dtype=float)
    M0_a    = strategy_df['M0'].to_numpy(dtype=float)
    v_t_a   = strategy_df['v_t'].to_numpy(dtype=float)
    T_a     = strategy_df['T'].to_numpy(dtype=float)
    n       = len(C_Q_a)

    # ── common random numbers ────────────────────────────────────────────
    rng = np.random.default_rng(seed)
    U   = rng.random((n_sims, n, 2))    # [:, :, 0] = ask, [:, :, 1] = bid

    # ── precompute quotes/probs where they don't depend on inventory ─────
    path_independent = strategy in ('el_aoud', 'zero_intelligence')
    if path_independent:
        if strategy == 'el_aoud':
            _q   = [el_aoud_quotes(M0=M0_a[i], C_Q=C_Q_a[i], B_param=B_param)
                    for i in range(n)]
            dp_a = np.array([x[0] for x in _q])
            dm_a = np.array([x[1] for x in _q])
        else:
            dp_a = C_Q_a * zi_spread_pct
            dm_a = C_Q_a * zi_spread_pct

        p_ask_a = np.array([arrival_probability(dp_a[i], C_Q_a[i],
                                                lambda_params, dt_minutes)
                            for i in range(n)])
        p_bid_a = np.array([arrival_probability(dm_a[i], C_Q_a[i],
                                                lambda_params, dt_minutes)
                            for i in range(n)])

    # ── accumulators ─────────────────────────────────────────────────────
    terminal_wealth   = np.zeros(n_sims)
    final_inventories = np.zeros(n_sims)
    stats = {k: np.zeros(n_sims) for k in
             ('max_drawdown', 'total_fills', 'ask_fills', 'bid_fills',
              'spread_income', 'mean_abs_inv', 'max_abs_inv', 'inv_std')}
    path_df = None

    for sim in range(n_sims):
        q1 = q2 = cash = 0.0
        W_hist  = np.empty(n)
        q1_hist = np.empty(n)
        ask_f = bid_f = 0
        spr_tot = 0.0
        path = []

        for i in range(n):
            C_Q, S, delta = C_Q_a[i], S_a[i], delta_a[i]

            # ── compute quotes ──────────────────────────────
            if path_independent:
                dp, dm = dp_a[i], dm_a[i]
                p_ask, p_bid = p_ask_a[i], p_bid_a[i]
            else:  # risk-averse — quotes depend on current inventory
                dp, dm = el_aoud_quotes_risk_averse(
                    M0=M0_a[i], C_Q=C_Q, q1=q1,
                    delta=delta, v_t=v_t_a[i], S=S, T=T_a[i],
                    B_param=B_param,
                    heston_p=heston_p,
                    epsilon=epsilon
                )
                p_ask = arrival_probability(dp, C_Q, lambda_params, dt_minutes)
                p_bid = arrival_probability(dm, C_Q, lambda_params, dt_minutes)

            # ── simulate fills ───────────────────────────────
            fill_ask = U[sim, i, 0] < p_ask
            fill_bid = U[sim, i, 1] < p_bid

            spread_income = 0.0

            if fill_ask:
                cash          += C_Q + dp
                q1            -= 1
                spread_income += dp
                ask_f         += 1

            if fill_bid:
                cash          -= C_Q - dm
                q1            += 1
                spread_income += dm
                bid_f         += 1

            spr_tot += spread_income

            # ── delta hedge rebalance ────────────────────────
            q2_new     = -q1 * delta
            hedge_cost = (q2_new - q2) * S
            cash      -= hedge_cost
            q2         = q2_new

            # ── total wealth (cash + option + stock hedge) ───
            W = cash + q1 * C_Q + q2 * S
            W_hist[i]  = W
            q1_hist[i] = q1

            if sim == 0:
                path.append({
                    'datetime'     : dts[i],
                    'C_Q'          : C_Q,
                    'S'            : S,
                    'M0'           : M0_a[i],
                    'v_t'          : v_t_a[i],
                    'T'            : T_a[i],
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
        terminal_wealth[sim]   = cash + q1 * C_Q_a[-1] + q2 * S_a[-1]
        final_inventories[sim] = q1

        stats['max_drawdown'][sim]  = (np.maximum.accumulate(W_hist)
                                       - W_hist).max()
        stats['total_fills'][sim]   = ask_f + bid_f
        stats['ask_fills'][sim]     = ask_f
        stats['bid_fills'][sim]     = bid_f
        stats['spread_income'][sim] = spr_tot
        stats['mean_abs_inv'][sim]  = np.abs(q1_hist).mean()
        stats['max_abs_inv'][sim]   = np.abs(q1_hist).max()
        stats['inv_std'][sim]       = q1_hist.std()

        if sim == 0:
            path_df = pd.DataFrame(path)

    return terminal_wealth, final_inventories, path_df, stats


def compute_metrics(terminal_wealth, final_inventories, stats,
                    label='', n_steps=None) -> dict:

    tw    = terminal_wealth
    fills = stats['total_fills']
    spr   = stats['spread_income']

    # per FILL, not per filled step — a two-sided step contributes dp + dm
    avg_spread = spr.sum() / max(fills.sum(), 1)

    m = {
        'total_pnl'   : tw.mean(),
        'median_pnl'  : np.median(tw),
        'pnl_std'     : tw.std(),
        'pnl_skew'    : pd.Series(tw).skew(),
        'pnl_kurt'    : pd.Series(tw).kurt(),
        'pnl_p5'      : np.percentile(tw, 5),
        'pnl_p95'     : np.percentile(tw, 95),

        'max_drawdown'     : stats['max_drawdown'].mean(),
        'max_drawdown_p95' : np.percentile(stats['max_drawdown'], 95),

        'total_fills' : fills.mean(),
        'ask_fills'   : stats['ask_fills'].mean(),
        'bid_fills'   : stats['bid_fills'].mean(),
        'ask_share'   : stats['ask_fills'].sum() / max(fills.sum(), 1),
        'fill_rate'   : fills.mean() / n_steps if n_steps else np.nan,
        'avg_spread'  : avg_spread,

        'mean_inventory'  : stats['mean_abs_inv'].mean(),
        'max_inventory'   : stats['max_abs_inv'].mean(),
        'final_inventory' : final_inventories.mean(),
        'final_inv_p5'    : np.percentile(final_inventories, 5),
        'final_inv_p95'   : np.percentile(final_inventories, 95),
        'inventory_std'   : stats['inv_std'].mean(),

        'total_spread_income': spr.mean(),
        'spread_pct_of_pnl'  : spr.mean() / tw.mean() if tw.mean() else np.nan,
    }

    if label:
        print(f"\n{'='*58}")
        print(f"  {label}   (n = {len(tw)} simulations)")
        print(f"{'='*58}")
        print(f"  Mean terminal PnL:       ${m['total_pnl']:>12.2f}")
        print(f"  Median terminal PnL:     ${m['median_pnl']:>12.2f}")
        print(f"  PnL std:                 ${m['pnl_std']:>12.2f}")
        print(f"  PnL 5th / 95th pct:      ${m['pnl_p5']:>12.2f} / ${m['pnl_p95']:.2f}")
        print(f"  PnL skewness:            {m['pnl_skew']:>12.4f}")
        print(f"  PnL kurtosis:            {m['pnl_kurt']:>12.4f}")
        print(f"  --- path metrics, mean across sims ---")
        print(f"  Max drawdown:            ${m['max_drawdown']:>12.2f}")
        print(f"  Max drawdown 95th pct:   ${m['max_drawdown_p95']:>12.2f}")
        print(f"  Total fills:             {m['total_fills']:>12.2f}")
        print(f"    ask fills:             {m['ask_fills']:>12.2f}")
        print(f"    bid fills:             {m['bid_fills']:>12.2f}")
        print(f"    ask share of fills:    {m['ask_share']:>12.2%}")
        print(f"  Fill rate per step:      {m['fill_rate']:>12.4f}")
        print(f"  Avg spread per fill:     ${m['avg_spread']:>12.2f}")
        print(f"  Mean |inventory|:        {m['mean_inventory']:>12.4f}")
        print(f"  Max |inventory|:         {m['max_inventory']:>12.2f}")
        print(f"  Final inventory:         {m['final_inventory']:>12.2f}")
        print(f"  Final inv 5th / 95th:    {m['final_inv_p5']:>12.2f} / {m['final_inv_p95']:.2f}")
        print(f"  Inventory std:           {m['inventory_std']:>12.4f}")
        print(f"  Total spread income:     ${m['total_spread_income']:>12.2f}")
        print(f"  Spread as % of PnL:      {m['spread_pct_of_pnl']:>12.1%}")

    return m


def compare_strategies(*args) -> pd.DataFrame:

    metrics_list = [
        ('Mean Terminal PnL ($)',   'total_pnl',           '${:,.0f}'),
        ('Median Terminal PnL ($)', 'median_pnl',          '${:,.0f}'),
        ('PnL Std ($)',             'pnl_std',             '${:,.0f}'),
        ('PnL 5th pct ($)',         'pnl_p5',              '${:,.0f}'),
        ('PnL 95th pct ($)',        'pnl_p95',             '${:,.0f}'),
        ('Mean Max Drawdown ($)',   'max_drawdown',        '${:,.0f}'),
        ('Max DD 95th pct ($)',     'max_drawdown_p95',    '${:,.0f}'),
        ('Mean Total Fills',        'total_fills',         '{:.1f}'),
        ('Ask Share of Fills',      'ask_share',           '{:.1%}'),
        ('Fill Rate per Step',      'fill_rate',           '{:.4f}'),
        ('Avg Spread/Fill ($)',     'avg_spread',          '${:,.2f}'),
        ('Mean |Inventory|',        'mean_inventory',      '{:.1f}'),
        ('Mean Final Inventory',    'final_inventory',     '{:.1f}'),
        ('Spread Income ($)',       'total_spread_income', '${:,.0f}'),
        ('Spread % of PnL',         'spread_pct_of_pnl',   '{:.0%}'),
    ]

    rows = []
    for display_name, key, fmt in metrics_list:
        row = {'Metric': display_name}
        for metrics, label in args:
            row[label] = fmt.format(metrics[key])
        rows.append(row)

    return pd.DataFrame(rows)