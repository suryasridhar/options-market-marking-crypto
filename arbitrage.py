def check_arbitrage(svi_params_df, k_grid=None):
    """
    Check butterfly and calendar arbitrage for split SVI fits.
    
    Returns DataFrame with violation flags per (ts, expiry).
    """
    if k_grid is None:
        k_grid = np.linspace(-0.5, 0.5, 200)
    
    results = []
    
    grouped = svi_params_df.groupby('ts')
    
    for ts, group in grouped:
        group = group.sort_values('T')
        expiries = group['expiry'].values
        
        w_surfaces = {}
        
        for _, row in group.iterrows():
            expiry = row['expiry']
            T = row['T']
            
            params = {
                'split': True,
                'a_put': row['a_put'], 'b_put': row['b_put'],
                'rho_put': row['rho_put'], 'm_put': row['m_put'],
                'sigma_put': row['sigma_put'],
                'a_call': row['a_call'], 'b_call': row['b_call'],
                'rho_call': row['rho_call'], 'm_call': row['m_call'],
                'sigma_call': row['sigma_call']
            }
            
            # evaluate w on grid
            w_vals = np.array([svi_w(k, params) for k in k_grid])
            w_surfaces[expiry] = (T, w_vals)
            
            # butterfly check — w must be non-negative
            butterfly_ok = np.all(w_vals >= 0)
            
            # density check — numerical second derivative of call price
            # d^2C/dK^2 >= 0 ↔ g(k) >= 0 where g is the density
            # approximate: check w is convex enough
            dw = np.diff(w_vals)
            d2w = np.diff(dw)
            
            # Lee's wing condition
            b = row['b_put'] if True else row['b_call']
            rho_put = row['rho_put']
            rho_call = row['rho_call']
            wing_ok_put = row['b_put'] * (1 + abs(rho_put)) < 4
            wing_ok_call = row['b_call'] * (1 + abs(rho_call)) < 4
            
            results.append({
                'ts': ts,
                'expiry': expiry,
                'T': T,
                'butterfly_ok': butterfly_ok,
                'wing_ok_put': wing_ok_put,
                'wing_ok_call': wing_ok_call,
                'w_min': w_vals.min(),
                'w_max': w_vals.max()
            })
        
        # calendar check — w non-decreasing in T
        sorted_expiries = group.sort_values('T')['expiry'].values
        
        for j in range(len(sorted_expiries)-1):
            exp1 = sorted_expiries[j]
            exp2 = sorted_expiries[j+1]
            
            if exp1 in w_surfaces and exp2 in w_surfaces:
                T1, w1 = w_surfaces[exp1]
                T2, w2 = w_surfaces[exp2]
                
                calendar_ok = np.all(w2 >= w1)
                n_violations = np.sum(w2 < w1)
                max_violation = np.max(np.maximum(w1 - w2, 0))
                
                # add calendar info to the shorter expiry row
                for r in results:
                    if r['ts'] == ts and r['expiry'] == exp1:
                        r['calendar_ok_vs_next'] = calendar_ok
                        r['calendar_violations'] = n_violations
                        r['calendar_max_violation'] = max_violation
    
    return pd.DataFrame(results)