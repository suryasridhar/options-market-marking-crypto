# model.py

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from surface import heston_call_fft, svi_iv
from utils import bs_delta, bs_vega


# ============================================================
# VARIANCE STATE
# ============================================================

def compute_v_t(svi_params_row) -> float:
    """
    Current variance state from ATM SVI implied variance.
    v_t = IV(k=0, T)^2
    
    Updated at each 15-minute timestamp — captures intraday
    surface dynamics without full Heston recalibration.
    
    Parameters:
        svi_params_row : Series  one row from svi_params_all
    
    Returns:
        v_t : float  current instantaneous variance
    """
    params = {
        'split': False,
        'a': svi_params_row['a'],
        'b': svi_params_row['b'],
        'rho': svi_params_row['rho'],
        'm': svi_params_row['m'],
        'sigma': svi_params_row['sigma']
    }
    T = svi_params_row['T']
    atm_iv = svi_iv(0.0, T, params)
    return atm_iv ** 2 if not np.isnan(atm_iv) else np.nan


# ============================================================
# REAL-WORLD OPTION PRICE
# ============================================================

def compute_cp(S, K, T, v_t, heston_p) -> float:
    """
    Heston option price under P-measure parameters.
    
    Uses current variance state v_t which updates each timestamp.
    Structural parameters (k, theta, eta, rho) fixed from training.
    
    Parameters:
        S        : float  current spot price
        K        : float  strike
        T        : float  time to expiry in years
        v_t      : float  current instantaneous variance
        heston_p : dict   {k, theta, eta, rho}
    
    Returns:
        C_P : float  real-world option price in USD
    """
    price = heston_call_fft(
        S=S,
        K_arr=np.array([float(K)]),
        T=T,
        r=0.0,
        v0=v_t,
        k_h=heston_p['k'],
        theta=heston_p['theta'],
        eta=heston_p['eta'],
        rho=heston_p['rho']
    )[0]
    return float(price)


# ============================================================
# STRATEGY INPUTS
# ============================================================

def compute_strategy_df(orderbook_test_df, spot_15m,
                         svi_params_all, heston_p,
                         expiry, strike,
                         contract_type) -> pd.DataFrame:
    """
    Build per-timestamp strategy inputs for the test period.
    
    At each 15-minute timestamp:
    - C_Q  : observed market mid price
    - v_t  : current variance from ATM SVI
    - C_P  : Heston P-measure price using v_t
    - M0   : C_Q - C_P (mispricing signal)
    - delta: BS delta using market IV
    - vega : BS vega
    
    Parameters:
        orderbook_test_df : DataFrame  test period orderbook
        spot_15m          : DataFrame  15-minute spot prices
        svi_params_all    : DataFrame  SVI parameters (all timestamps)
        heston_p          : dict       P-measure parameters
        expiry            : str
        strike            : float
        contract_type     : str
    
    Returns strategy_df with columns:
        ts, datetime, S, T, k, F, C_Q, v_t, C_P, M0, iv, delta, vega
    """
    # filter to chosen contract
    contract = orderbook_test_df[
        (orderbook_test_df['expiry'] == expiry) &
        (orderbook_test_df['strike'] == strike) &
        (orderbook_test_df['option_type'] == contract_type)
    ].copy()

    # merge 15m spot
    contract = pd.merge_asof(
        contract.sort_values('ts'),
        spot_15m[['ts','close_usd']].rename(columns={'close_usd':'S'}),
        on='ts', direction='backward'
    )

    # observed market mid in USD
    contract['C_Q'] = contract['mid_btc'] * contract['S']

    records = []

    for _, row in contract.iterrows():
        ts  = row['ts']
        S   = row['S']
        T   = row['T']
        k   = row['k']
        F   = row['F']
        C_Q = row['C_Q']

        if T <= 0 or np.isnan(C_Q) or np.isnan(S):
            continue

        # SVI params at this timestamp
        svi_row = svi_params_all[
            (svi_params_all['ts'] == ts) &
            (svi_params_all['expiry'] == expiry)
        ]
        if svi_row.empty:
            continue
        svi_row = svi_row.iloc[0]

        # current variance from ATM SVI
        v_t = compute_v_t(svi_row)
        if np.isnan(v_t):
            continue

        # IV at contract moneyness
        params = {
            'split': False,
            'a': svi_row['a'], 'b': svi_row['b'],
            'rho': svi_row['rho'], 'm': svi_row['m'],
            'sigma': svi_row['sigma']
        }
        iv = svi_iv(k, T, params)
        if np.isnan(iv):
            continue

        # Greeks
        delta = bs_delta(S, strike, T, 0.0, iv, contract_type)
        vega  = bs_vega(S, strike, T, 0.0, iv)

        # C_P with current v_t
        C_P = compute_cp(S, strike, T, v_t, heston_p)
        M0  = C_Q - C_P

        records.append({
            'ts'      : ts,
            'datetime': row['datetime'],
            'S'       : S,
            'T'       : T,
            'k'       : k,
            'F'       : F,
            'C_Q'     : C_Q,
            'v_t'     : v_t,
            'C_P'     : C_P,
            'M0'      : M0,
            'iv'      : iv,
            'delta'   : delta,
            'vega'    : vega
        })

    df = pd.DataFrame(records)
    return df.sort_values('ts').reset_index(drop=True)


# ============================================================
# LAMBDA CALIBRATION
# ============================================================

def calibrate_lambda(trades_train, orderbook_df,
                      n_bins=20) -> dict:
    """
    Calibrate order arrival intensity from all training period trades.
    
    Uses percentage delta = |trade_price - mid| / mid * 100
    for cross-contract comparability.
    
    Intensity function (El Aoud, beta=0.5, gamma=1.5 fixed):
        lambda(delta_pct) = A / (B + sqrt(delta_pct))^1.5
    
    Only A and B are calibrated from data.
    gamma=1.5 and beta=0.5 are fixed from El Aoud & Abergel (2015).
    
    Parameters:
        trades_train : DataFrame  all training period trades
        orderbook_df : DataFrame  training orderbook (for mid prices)
        n_bins       : int        number of delta bins
    
    Returns dict: {A, B, gamma, beta}
    """
    from scipy.optimize import curve_fit

    GAMMA = 1.5
    BETA  = 0.5

    if len(trades_train) < 10:
        print("Insufficient trades — using defaults")
        return {'A': 10.0, 'B': 1.0, 'gamma': GAMMA, 'beta': BETA}

    # get mid prices for all contracts from orderbook
    ob_mid = orderbook_df[
        ['ts','strike','mid_btc','spot_usd']
    ].copy()
    ob_mid['mid_usd'] = ob_mid['mid_btc'] * ob_mid['spot_usd']

    # extract strike from instrument_name
    trades = trades_train.copy()
    trades['strike'] = trades['instrument_name'].apply(
        lambda x: int(x.split('-')[3])
    )

    # merge mid price into trades
    trades = pd.merge_asof(
        trades.sort_values('ts'),
        ob_mid[['ts','strike','mid_usd',
                'spot_usd']].sort_values('ts'),
        on='ts',
        by='strike',
        direction='nearest'
    )

    # trade price in USD
    trades['price_usd'] = trades['price_btc'] * trades['spot_usd']

    # percentage delta
    trades['delta_pct'] = (
        (trades['price_usd'] - trades['mid_usd']).abs() /
        trades['mid_usd']
    ) * 100

    # filter invalid and extreme values
    trades = trades[
        (trades['delta_pct'] > 0.001) &
        (trades['delta_pct'] < 50) &
        trades['mid_usd'].notna() &
        trades['spot_usd'].notna()
    ].copy()

    print(f"Trades used: {len(trades)}")
    print(f"Delta % stats:")
    print(trades['delta_pct'].describe().round(3))

    if len(trades) < 5:
        print("Insufficient valid trades — using defaults")
        return {'A': 10.0, 'B': 1.0, 'gamma': GAMMA, 'beta': BETA}

    # observation time in hours
    T_obs = (
        pd.to_datetime(trades_train['ts'].max(), unit='ms') -
        pd.to_datetime(trades_train['ts'].min(), unit='ms')
    ).total_seconds() / 3600

    if T_obs <= 0:
        return {'A': 10.0, 'B': 1.0, 'gamma': GAMMA, 'beta': BETA}

    # bin by delta_pct
    delta_max = trades['delta_pct'].quantile(0.95)
    bins = np.linspace(0, delta_max, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_width = bins[1] - bins[0]

    counts, _ = np.histogram(trades['delta_pct'].values, bins=bins)
    rates = counts / (T_obs * bin_width)

    valid = counts >= 2
    delta_vals = bin_centers[valid]
    rate_vals  = rates[valid]

    print(f"Valid bins: {valid.sum()}")
    print(f"Delta range: {delta_vals.min():.2f}% to "
          f"{delta_vals.max():.2f}%")

    if len(delta_vals) < 3:
        print("Insufficient bins — using defaults")
        return {'A': 10.0, 'B': 1.0, 'gamma': GAMMA, 'beta': BETA}

    # fit A and B only — gamma and beta fixed from paper
    def lambda_func(delta, A, B):
        return A / (B + delta ** (1/BETA)) ** GAMMA

    try:
        popt, _ = curve_fit(
            lambda_func, delta_vals, rate_vals,
            p0=[10.0, 1.0],
            bounds=([0, 0.01], [10000, 50]),
            maxfev=5000
        )
        A, B = popt

        # verify fit quality
        rate_pred = lambda_func(delta_vals, A, B)
        ss_res = np.sum((rate_vals - rate_pred)**2)
        ss_tot = np.sum((rate_vals - rate_vals.mean())**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0

        print(f"\nLambda calibration complete:")
        print(f"  A     = {A:.4f}  (base rate)")
        print(f"  B     = {B:.4f}  (scale, in % units)")
        print(f"  gamma = {GAMMA}  (fixed from El Aoud)")
        print(f"  beta  = {BETA}   (fixed from El Aoud)")
        print(f"  R²    = {r2:.4f}")

        return {'A': A, 'B': B, 'gamma': GAMMA, 'beta': BETA}

    except Exception as e:
        print(f"Calibration failed: {e} — using defaults")
        return {'A': 10.0, 'B': 1.0, 'gamma': GAMMA, 'beta': BETA}
    
# ============================================================
# ARRIVAL PROBABILITY
# ============================================================

def arrival_probability(delta_usd, mid_price,
                         lambda_params, dt_minutes=15) -> float:
    """
    P(fill) = 1 - exp(-lambda(delta_pct) * dt)
    
    Parameters:
        delta_usd     : float  quote distance in USD
        mid_price     : float  current option mid in USD
        lambda_params : dict   calibrated in % space
        dt_minutes    : float
    """
    delta_pct = (delta_usd / mid_price) * 100

    A     = lambda_params['A']
    B     = lambda_params['B']
    gamma = lambda_params['gamma']
    beta  = lambda_params['beta']

    dt_hours  = dt_minutes / 60
    intensity = A / (B + delta_pct ** (1/beta)) ** gamma
    return 1 - np.exp(-intensity * dt_hours)

# ============================================================
# EL AOUD OPTIMAL QUOTES
# ============================================================

def el_aoud_quotes(M0, C_Q, B_param) -> tuple:
    """
    El Aoud optimal quotes — beta=0.5, gamma=1.5 fixed from paper.
    """
    gamma = 1.5
    M0_pct = (M0 / C_Q) * 100

    discriminant = gamma**2 * M0_pct**2 + B_param * (2*gamma - 1)
    sqrt_term = np.sqrt(max(discriminant, 0))
    denom = 2 * gamma - 1  # = 2.0, always positive

    delta_plus_pct  = (sqrt_term - gamma * M0_pct) / denom
    delta_minus_pct = (sqrt_term + gamma * M0_pct) / denom

    delta_plus_pct  = max(delta_plus_pct,  0.001)
    delta_minus_pct = max(delta_minus_pct, 0.001)

    delta_plus_usd  = delta_plus_pct  / 100 * C_Q
    delta_minus_usd = delta_minus_pct / 100 * C_Q

    return delta_plus_usd, delta_minus_usd

def expected_integrated_variance(v_t, T, heston_p):
    """
    E^P[integral_t^T v_u du | v_t] under CIR (Heston P).
    Closed form from CIR expectation.
    
    Parameters:
        v_t      : float  current variance
        T        : float  time to expiry in years
        heston_p : dict   P-measure parameters {k, theta}
    
    Returns float — expected integrated variance
    """
    k_R     = heston_p['k']
    theta_R = heston_p['theta']
    
    if k_R <= 0 or T <= 0:
        return v_t * T
    
    exp_term = np.exp(-k_R * T)
    
    # CIR expected integrated variance
    E_int_v = (
        v_t * (1 - exp_term) / k_R +
        theta_R * (T - (1 - exp_term) / k_R)
    )
    
    return max(E_int_v, 0.0)


def el_aoud_quotes_risk_averse(M0, C_Q, q1, delta,
                                v_t, S, T,
                                B_param, heston_p,
                                epsilon=0.01) -> tuple:
    """
    Risk-averse El Aoud via singular perturbation (El Aoud Section 4).
    
    Start from risk-neutral quotes, apply first-order inventory correction:
        delta+_RA = delta+_RN - epsilon * |theta4| * q1
        delta-_RA = delta-_RN + epsilon * |theta4| * q1
    
    When long  (q1 > 0): tighter ask, wider bid → shed long inventory
    When short (q1 < 0): wider ask, tighter bid → shed short inventory
    """
    gamma = 1.5

    # risk-neutral quotes first
    M0_pct = (M0 / C_Q) * 100
    disc   = gamma**2 * M0_pct**2 + B_param * (2*gamma - 1)

    if disc < 0:
        sym = max(np.sqrt(abs(B_param/(2*gamma-1))), 0.001)
        return sym/100*C_Q, sym/100*C_Q

    sqrt_t = np.sqrt(disc)
    denom  = 2*gamma - 1

    dp_pct_rn = (sqrt_t - gamma*M0_pct) / denom
    dm_pct_rn = (sqrt_t + gamma*M0_pct) / denom

    # inventory correction from singular perturbation
    E_int_v    = expected_integrated_variance(v_t, T, heston_p)
    theta4_abs = delta**2 * S**2 * E_int_v / (C_Q**2)  # positive
    correction = epsilon * theta4_abs * q1  # positive when long, negative when short

    # apply correction
    dp_pct = dp_pct_rn - correction  # tighten when long, widen when short
    dm_pct = dm_pct_rn + correction  # widen when long, tighten when short

    dp_pct = max(dp_pct, 0.001)
    dm_pct = max(dm_pct, 0.001)

    return dp_pct/100*C_Q, dm_pct/100*C_Q


