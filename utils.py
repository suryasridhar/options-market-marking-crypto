# utils.py

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm
from datetime import datetime
import pandas as pd


def bs_call(S, K, T, r, sigma):
    """Black-Scholes call price"""
    if T <= 0 or sigma <= 0:
        return max(S - K, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)


def bs_put(S, K, T, r, sigma):
    """Black-Scholes put price"""
    if T <= 0 or sigma <= 0:
        return max(K - S, 0)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)


def bs_iv(price_usd, spot, strike, T, r=0.0, option_type='C'):
    """
    Implied volatility via Brent root finding.
    Returns np.nan if inversion fails.
    """
    if T <= 0:
        return np.nan

    if option_type == 'C':
        intrinsic = max(spot - strike*np.exp(-r*T), 0)
        fn = lambda s: bs_call(spot, strike, T, r, s) - price_usd
    else:
        intrinsic = max(strike*np.exp(-r*T) - spot, 0)
        fn = lambda s: bs_put(spot, strike, T, r, s) - price_usd

    if price_usd <= intrinsic:
        return np.nan

    try:
        return brentq(fn, 1e-6, 10.0, maxiter=200)
    except:
        return np.nan
    
def bs_iv_with_forward(price_usd, F, K, T, option_type='C'):
    """
    BS IV inversion using forward price F directly.
    Avoids needing to specify r separately.
    F comes from put-call parity.
    """
    from scipy.stats import norm
    
    def bs_call_F(sigma):
        if sigma <= 0 or T <= 0:
            return max(F - K, 0)
        d1 = (np.log(F/K) + 0.5*sigma**2*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return F*norm.cdf(d1) - K*norm.cdf(d2)  # undiscounted
    
    def bs_put_F(sigma):
        if sigma <= 0 or T <= 0:
            return max(K - F, 0)
        d1 = (np.log(F/K) + 0.5*sigma**2*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        return K*norm.cdf(-d2) - F*norm.cdf(-d1)  # undiscounted
    
    if option_type == 'C':
        intrinsic = max(F - K, 0)
        fn = lambda s: bs_call_F(s) - price_usd
    else:
        intrinsic = max(K - F, 0)
        fn = lambda s: bs_put_F(s) - price_usd
    
    if price_usd <= intrinsic:
        return np.nan
    
    try:
        return brentq(fn, 1e-6, 10.0, maxiter=200)
    except:
        return np.nan


def time_to_expiry(expiry_code, ts_ms):
    """
    Time to expiry in years.
    expiry_code : str  e.g. '260626'
    ts_ms       : int  Unix timestamp in milliseconds
    """
    expiry_dt = datetime.strptime('20' + expiry_code, '%Y%m%d')
    obs_dt = datetime.fromtimestamp(ts_ms / 1000)
    seconds = (expiry_dt - obs_dt).total_seconds()
    return max(seconds / (365.25 * 24 * 3600), 0)

def bs_delta(S, K, T, r, sigma, option_type='C'):
    """
    Black-Scholes delta.
    For a call: N(d1)
    For a put:  N(d1) - 1
    """
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        if option_type == 'C':
            return 1.0 if S > K else 0.0
        else:
            return -1.0 if S < K else 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    if option_type == 'C':
        return norm.cdf(d1)
    else:
        return norm.cdf(d1) - 1.0


def bs_vega(S, K, T, r, sigma):
    """
    Black-Scholes vega — same for calls and puts.
    dC/d(sigma)
    """
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S * norm.pdf(d1) * np.sqrt(T)


def bs_gamma(S, K, T, r, sigma):
    """
    Black-Scholes gamma — same for calls and puts.
    d2C/dS2
    """
    from scipy.stats import norm
    if T <= 0 or sigma <= 0:
        return 0.0
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))

from scipy.stats import norm
import numpy as np

def bs_greeks(S, K, T, sigma, r=0.0, option_type='C'):
    """
    Black-Scholes Greeks for a European option.
    S     : spot price (USD)
    K     : strike (USD)
    T     : time to expiry in years
    sigma : implied volatility
    r     : risk-free rate
    returns dict: delta, gamma, vega (per 1% vol), theta (per day)
    """
    if T <= 0 or sigma <= 0:
        return {'delta': 0.0, 'gamma': 0.0, 'vega': 0.0, 'theta': 0.0}

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100

    if option_type == 'C':
        delta = norm.cdf(d1)
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        delta = norm.cdf(d1) - 1
        theta = (-(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))
                 + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}


def compute_greeks_df(df, K, r=0.0, option_type='C'):
    records = []
    for _, row in df.iterrows():
        S     = row['S']
        T     = row['T']
        sigma = row['iv']

        g = bs_greeks(S, K, T, sigma, r, option_type)

        records.append({
            'datetime': row['datetime'],
            'S'       : S,
            'delta'   : row['delta'] if 'delta' in row.index else g['delta'],
            'gamma'   : g['gamma'],
            'vega'    : row['vega']  if 'vega'  in row.index else g['vega'],
            'theta'   : g['theta']
        })
    return pd.DataFrame(records)

import torch

def aggregate_greek_inventory(positions: torch.Tensor, 
                               greek_matrix: torch.Tensor) -> torch.Tensor:
    """
    Net portfolio Greeks from positions and per-contract Greek matrix.
    positions   : (n_options,)
    greek_matrix: (n_options, k_greeks)
    returns     : (k_greeks,) net Greeks
    """
    if positions.ndim != 1:
        raise ValueError("positions must be 1D")
    if greek_matrix.ndim != 2:
        raise ValueError("greek_matrix must be 2D")
    if greek_matrix.shape[0] != positions.shape[0]:
        raise ValueError("dimension mismatch")
    return positions.unsqueeze(1).mul(greek_matrix).sum(dim=0)


def apply_fill(positions: torch.Tensor,
               greek_inventory: torch.Tensor,
               fill_index: int,
               signed_qty: float,
               greek_matrix: torch.Tensor):
    """
    Apply one fill — update positions and Greek inventory atomically.
    signed_qty > 0 : we bought (bid was hit)
    signed_qty < 0 : we sold (ask was hit)
    """
    new_positions = positions.clone()
    new_inventory = greek_inventory.clone()
    qty = positions.new_tensor(float(signed_qty))
    new_positions[fill_index] = new_positions[fill_index] + qty
    new_inventory = new_inventory + qty * greek_matrix[fill_index]
    return new_positions, new_inventory


def maker_signed_qty(taker_direction: str, amount: float) -> float:
    """
    Convert taker direction to maker signed quantity.
    taker buys  → ask lifted → we sold → negative
    taker sells → bid hit   → we bought → positive
    """
    direction = taker_direction.lower()
    if direction == 'buy':
        return -float(amount)
    if direction == 'sell':
        return float(amount)
    raise ValueError(f"Unknown direction: {direction}")