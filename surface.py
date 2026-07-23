# surface.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize, brentq
from scipy.stats import norm
from scipy.fft import fft


def fit_svi_slice(k_arr, w_arr, weights=None):
    """
    Fit SVI to one expiry slice.
    Single curve — smile is continuous after forward price correction.
    """
    if weights is None:
        weights = np.ones(len(k_arr))
    
    def svi_formula(k, a, b, rho, m, sigma):
        return a + b*(rho*(k-m) + np.sqrt((k-m)**2 + sigma**2))
    
    def objective(params):
        a, b, rho, m, sigma = params
        w_hat = svi_formula(k_arr, a, b, rho, m, sigma)
        return np.sum(weights * (w_hat - w_arr)**2)
    
    bounds = [
        (-1, 1),
        (1e-6, 2),
        (-0.999, 0.999),
        (-1, 1),
        (1e-6, 1)
    ]
    
    starting_points = [
        [np.mean(w_arr), 0.1, -0.5, 0.0, 0.1],
        [np.mean(w_arr), 0.2, -0.3, 0.0, 0.2],
        [np.min(w_arr),  0.1, -0.7, 0.0, 0.1],
        [np.mean(w_arr), 0.3,  0.0, 0.0, 0.3],
    ]
    
    best_result = None
    best_loss = np.inf
    
    for x0 in starting_points:
        try:
            result = minimize(
                objective, x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-12}
            )
            if result.success and result.fun < best_loss:
                best_loss = result.fun
                best_result = result
        except:
            continue
    
    if best_result is None:
        return None
    
    a, b, rho, m, sigma = best_result.x
    return {
        'a': a, 'b': b, 'rho': rho,
        'm': m, 'sigma': sigma,
        'loss': best_loss,
        'split': False
    }

def svi_w(k, params):
    
    a,b,rho,m,sigma = (params['a'], params['b'], params['rho'],
                           params['m'], params['sigma'])
    
    return a + b*(rho*(k-m) + np.sqrt((k-m)**2 + sigma**2))


def svi_iv(k, T, params):
    """Convert SVI total variance to IV"""
    w = svi_w(k, params)
    if w <= 0:
        return np.nan
    return np.sqrt(w / T)


def fit_surface(orderbook_df, min_strikes=5):
    """
    Fit SVI to each (ts, expiry) slice.
    Returns DataFrame with SVI parameters.
    """
    results = []
    grouped = orderbook_df.groupby(['ts','expiry'])
    total = len(grouped)
    
    for i, ((ts, expiry), group) in enumerate(grouped):
        if i % 500 == 0:
            print(f"  Fitting {i}/{total}...")
        
        if len(group) < min_strikes:
            continue
        
        k_arr = group['k'].values
        w_arr = group['w'].values
        weights = np.exp(-2 * k_arr**2)
        
        params = fit_svi_slice(k_arr, w_arr, weights)
        
        if params is None:
            continue
        
        row = {
            'ts': ts,
            'expiry': expiry,
            'T': group['T'].iloc[0],
            'n_strikes': len(group),
            'loss': params['loss'],
            'split': params.get('split', False)
        }
        
        # store parameters
        if params.get('split', False):
            for wing in ['put', 'call']:
                for p in ['a','b','rho','m','sigma']:
                    row[f'{p}_{wing}'] = params[f'{p}_{wing}']
        else:
            for p in ['a','b','rho','m','sigma']:
                row[p] = params[p]
        
        results.append(row)
    
    return pd.DataFrame(results)

def check_arbitrage(svi_params_df, k_grid=None):
    if k_grid is None:
        k_grid = np.linspace(-0.5, 0.5, 200)
    
    results = []
    grouped = svi_params_df.groupby('ts')
    
    for ts, group in grouped:
        group = group.sort_values('T')
        w_surfaces = {}
        
        for _, row in group.iterrows():
            expiry = row['expiry']
            T = row['T']
            
            params = {
                'split': False,
                'a': row['a'], 'b': row['b'],
                'rho': row['rho'], 'm': row['m'],
                'sigma': row['sigma']
            }
            
            w_vals = np.array([svi_w(k, params) for k in k_grid])
            w_surfaces[expiry] = (T, w_vals)
            
            butterfly_ok = np.all(w_vals >= 0)
            wing_ok = row['b'] * (1 + abs(row['rho'])) < 4
            
            results.append({
                'ts': ts,
                'expiry': expiry,
                'T': T,
                'butterfly_ok': butterfly_ok,
                'wing_ok': wing_ok,
                'w_min': w_vals.min(),
                'w_max': w_vals.max()
            })
        
        sorted_expiries = group.sort_values('T')['expiry'].values
        
        for j in range(len(sorted_expiries)-1):
            exp1 = sorted_expiries[j]
            exp2 = sorted_expiries[j+1]
            
            if exp1 in w_surfaces and exp2 in w_surfaces:
                T1, w1 = w_surfaces[exp1]
                T2, w2 = w_surfaces[exp2]
                calendar_ok = np.all(w2 >= w1)
                n_violations = np.sum(w2 < w1)
                
                for r in results:
                    if r['ts'] == ts and r['expiry'] == exp1:
                        r['calendar_ok_vs_next'] = calendar_ok
                        r['calendar_violations'] = n_violations
    
    return pd.DataFrame(results)

def heston_call_fft(S, K_arr, T, r, v0, k_h, theta, eta, rho,
                    N=4096, alpha=1.5, eta_grid=0.1):
    i = 1j
    K_arr = np.asarray(K_arr, dtype=float)
    
    lambda_grid = 2*np.pi / (N * eta_grid)
    v = np.arange(N) * eta_grid
    b = -N/2 * lambda_grid
    x_grid = b + np.arange(N) * lambda_grid
    
    def heston_cf_normalized(xi):
        a = k_h - rho*eta*i*xi
        d = np.sqrt(a**2 + eta**2*(xi**2 + i*xi))
        r_minus = (a - d) / (a + d)
        exp_dT = np.exp(-d*T)
        denom = 1 - r_minus*exp_dT
        D = (a - d)/eta**2 * (1 - exp_dT)/denom
        C = (r*i*xi*T +
             k_h*theta/eta**2 * (
                 (a - d)*T - 2*np.log(denom/(1 - r_minus))
             ))
        return np.exp(C + D*v0)
    
    xi = v - (alpha + 1)*i
    phi = heston_cf_normalized(xi)
    
    # Carr-Madan denominator — DO NOT zero out v=0
    # at v=0: denom = alpha^2 + alpha = 3.75 (for alpha=1.5)
    # no singularity here — the damping factor handles it
    denom_cm = alpha**2 + alpha - v**2 + i*(2*alpha+1)*v
    
    psi = np.exp(-r*T) * phi / denom_cm
    
    # Simpson weights
    w = np.ones(N)
    w[0] = 1/3
    w[-1] = 1/3
    w[1:-1:2] = 4/3
    w[2:-2:2] = 2/3
    
    fft_input = w * psi * np.exp(i*v*(-b)) * eta_grid
    fft_output = np.fft.fft(fft_input)
    
    call_grid = (S * np.exp(-alpha * x_grid) / np.pi * 
                 np.real(fft_output))
    call_grid = np.maximum(call_grid, 0)
    
    x_arr = np.log(K_arr / S)
    call_prices = np.interp(x_arr, x_grid, call_grid)
    
    return np.maximum(call_prices, 0)


def heston_iv_surface_fft(S, T_arr, k_arr, r, v0, k_h,
                           theta, eta, rho):
    """
    Compute Heston IVs across multiple (T, k) points.
    One FFT call per unique expiry — vectorized over strikes.
    
    Parameters:
        S     : float  spot price
        T_arr : array  time to expiry per point
        k_arr : array  log moneyness per point
        r     : float  risk-free rate
        v0    : float  initial variance
        k_h   : float  mean reversion speed
        theta : float  long-run variance
        eta   : float  vol of vol
        rho   : float  spot-vol correlation
    
    Returns array of IVs same length as T_arr/k_arr.
    """
    from utils import bs_iv
    
    T_arr = np.asarray(T_arr)
    k_arr = np.asarray(k_arr)
    results = np.full(len(T_arr), np.nan)
    unique_T = np.unique(T_arr)
    
    for T in unique_T:
        if T <= 0:
            continue
        
        mask = T_arr == T
        k_targets = k_arr[mask]
        K_arr_T = S * np.exp(k_targets)
        
        # price all calls at once via FFT
        call_prices = heston_call_fft(
            S, K_arr_T, T, r, v0, k_h, theta, eta, rho
        )
        
        # convert to IV
        for j, (K, call_price, k_target) in enumerate(
            zip(K_arr_T, call_prices, k_targets)
        ):
            if k_target <= 0:
                put_price = call_price - S + K*np.exp(-r*T)
                iv = bs_iv(max(put_price, 0), S, K, T, r, 'P')
            else:
                iv = bs_iv(call_price, S, K, T, r, 'C')
            
            global_idx = np.where(mask)[0][j]
            results[global_idx] = iv
    
    return results

def calibrate_heston_q(svi_params_df, orderbook_df,
                        S0, r=0.0,
                        n_points_per_slice=10) -> dict:
    """
    Calibrate Heston Q-parameters to SVI surface using FFT pricing.
    
    Parameters:
        svi_params_df    : DataFrame  fitted SVI parameters
        orderbook_df     : DataFrame  market data
        S0               : float      current spot
        r                : float      risk-free rate
        n_points_per_slice: int       k points per expiry
    
    Returns dict: {v0, k, theta, eta, rho, loss, rmse, success}
    """
    # build target points from SVI surface
    target_points = []
    
    for _, row in svi_params_df.iterrows():
        expiry = row['expiry']
        T = row['T']
        
        # get forward price for this slice
        slice_data = orderbook_df[
            orderbook_df['expiry'] == expiry
        ]
        if slice_data.empty:
            continue
        
        # sample k points across both wings
        k_put  = np.linspace(-0.4, -0.02, n_points_per_slice//2)
        k_call = np.linspace(0.02,  0.4,  n_points_per_slice//2)
        k_points = np.concatenate([k_put, k_call])
        
        params = {
            'split': False,
            'a': row['a'], 'b': row['b'],
            'rho': row['rho'], 'm': row['m'],
            'sigma': row['sigma']
        }
        
        for k in k_points:
            iv_target = svi_iv(k, T, params)
            if np.isnan(iv_target):
                continue
            target_points.append({
                'expiry': expiry,
                'T': T,
                'k': k,
                'iv_target': iv_target
            })
    
    target_df = pd.DataFrame(target_points)
    T_arr = target_df['T'].values
    k_arr = target_df['k'].values
    iv_targets = target_df['iv_target'].values
    
    print(f"Calibrating to {len(target_df)} points "
          f"across {svi_params_df['expiry'].nunique()} expiries...")
    
    def objective(params):
        v0, k_h, theta, eta, rho = params
        
        if (v0 <= 0 or k_h <= 0 or theta <= 0 or
            eta <= 0 or abs(rho) >= 1):
            return 1e10
        if 2*k_h*theta <= eta**2:
            return 1e10
        
        try:
            iv_model = heston_iv_surface_fft(
                S0, T_arr, k_arr, r,
                v0, k_h, theta, eta, rho
            )
            valid = ~np.isnan(iv_model)
            if valid.sum() < 5:
                return 1e10
            return np.mean((iv_model[valid] - iv_targets[valid])**2)
        except:
            return 1e10
    
    # initial guess from ATM IV
    atm_mask = np.abs(k_arr) < 0.05
    atm_iv = np.mean(iv_targets[atm_mask]) if atm_mask.any() else 0.4
    v0_init = atm_iv**2
    
    bounds = [
        (0.001, 2.0),     # v0
        (0.001, 10.0),    # k
        (0.001, 2.0),     # theta
        (0.001, 2.0),     # eta
        (-0.999, 0.999)   # rho
    ]
    
    starting_points = [
        [v0_init, 2.0, v0_init,     0.5, -0.7],
        [v0_init, 1.0, v0_init*0.8, 0.3, -0.5],
        [v0_init, 3.0, v0_init*1.2, 0.8, -0.8],
        [v0_init, 5.0, v0_init,     1.0, -0.6],
    ]
    
    best_result = None
    best_loss = np.inf
    
    for idx, x0 in enumerate(starting_points):
        print(f"  Starting point {idx+1}/{len(starting_points)}...")
        try:
            result = minimize(
                objective, x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-8}
            )
            if result.fun < best_loss:
                best_loss = result.fun
                best_result = result
        except:
            continue
    
    v0, k_h, theta, eta, rho = best_result.x
    
    print(f"\nCalibration complete.")
    print(f"RMSE: {np.sqrt(best_loss):.4f}")
    print(f"v0    = {v0:.4f}  (current variance, sqrt={np.sqrt(v0):.4f})")
    print(f"k     = {k_h:.4f}  (mean reversion speed)")
    print(f"theta = {theta:.4f}  (long-run variance, sqrt={np.sqrt(theta):.4f})")
    print(f"eta   = {eta:.4f}  (vol of vol)")
    print(f"rho   = {rho:.4f}  (spot-vol correlation)")
    
    return {
        'v0': v0, 'k': k_h, 'theta': theta,
        'eta': eta, 'rho': rho,
        'loss': best_loss,
        'rmse': np.sqrt(best_loss),
        'success': best_result.success
    }

def calibrate_heston_p(spot_df) -> dict:
    """
    Estimate Heston P-measure parameters from historical BTC data.
    Fits CIR process to realized variance time series.
    
    Parameters:
        spot_df : DataFrame  with columns close_usd, log_return
    
    Returns dict: {v0, k, theta, eta, rho}
    """
    df = spot_df.dropna(subset=['log_return']).copy()
    
    # realized variance — 21-day rolling
    df['rv'] = df['log_return'].rolling(21).var() * 252
    df = df.dropna(subset=['rv'])
    
    # estimate rho_R — correlation between returns and variance changes
    df['drv'] = df['rv'].diff()
    df = df.dropna(subset=['drv'])
    
    rho_R = df['log_return'].corr(df['drv'])
    
    # OLS on discretized CIR
    # dv_t = k*(theta - v_t)*dt + eta*sqrt(v_t)*dW
    # divide by sqrt(v_t)*sqrt(dt):
    # dv_t / (sqrt(v_t)*sqrt(dt)) = k*theta*sqrt(dt)/sqrt(v_t) 
    #                               - k*sqrt(v_t)*sqrt(dt) + eta*eps
    
    dt = 1/252  # daily
    v = df['rv'].values[:-1]
    dv = df['drv'].values[1:]
    
    # filter valid values
    valid = (v > 0) & np.isfinite(dv) & np.isfinite(v)
    v = v[valid]
    dv = dv[valid]
    
    # OLS: dv = a*dt + b*v*dt + noise
    # where a = k*theta, b = -k
    X = np.column_stack([
        np.ones(len(v)) * dt,
        v * dt
    ])
    y = dv
    
    # least squares
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    k_theta = coeffs[0]  # k*theta
    neg_k = coeffs[1]    # -k
    
    k_R = max(-neg_k, 0.01)
    theta_R = max(k_theta / k_R, 0.001) if k_R > 0 else np.mean(v)
    
    # eta from residuals
    residuals = y - X @ coeffs
    eta_R = np.std(residuals) / np.sqrt(dt * np.mean(v)) if np.mean(v) > 0 else 0.5
    
    # v0 — current realized variance
    v0_R = df['rv'].iloc[-1]
    
    print(f"P-measure parameters estimated from {len(df)} daily observations:")
    print(f"v0    = {v0_R:.4f}  (sqrt={np.sqrt(v0_R):.4f})")
    print(f"k     = {k_R:.4f}  (mean reversion speed)")
    print(f"theta = {theta_R:.4f}  (long-run variance, sqrt={np.sqrt(theta_R):.4f})")
    print(f"eta   = {eta_R:.4f}  (vol of vol)")
    print(f"rho   = {rho_R:.4f}  (spot-vol correlation)")
    
    return {
        'v0': v0_R,
        'k': k_R,
        'theta': theta_R,
        'eta': eta_R,
        'rho': rho_R
    }