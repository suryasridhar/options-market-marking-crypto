# extractor.py

import tarfile
import json
import numpy as np
import pandas as pd
from datetime import datetime
import requests
import os

OKX_URL = "https://www.okx.com/api/v5/market/history-candles"

# ============================================================
# PRIVATE HELPERS
# ============================================================

def _get_snapshots_only(f):
    """
    Read only snapshot messages — ignore all updates.
    OKX sends full book snapshot every 15 minutes.
    """
    records = []
    
    for line in f:
        msg = json.loads(line)
        
        if msg['action'] != 'snapshot':
            continue
        
        if not msg['bids'] or not msg['asks']:
            continue
        
        ts = int(msg['ts'])
        best_bid = float(msg['bids'][0][0])
        best_ask = float(msg['asks'][0][0])
        
        if best_bid < best_ask:
            records.append({
                'ts': ts,
                'best_bid_btc': best_bid,
                'best_ask_btc': best_ask,
                'mid_btc': (best_bid + best_ask) / 2
            })
    
    return records


def _parse_filename(filename):
    """
    Parse contract info from OKX filename.
    'BTC-USD-260626-80000-C-L2orderbook-400lv-2026-05-01.data'
    Returns (expiry, strike, option_type)
    """
    parts = filename.split('-')
    expiry = parts[2]
    strike = int(parts[3])
    option_type = parts[4]
    return expiry, strike, option_type


# ============================================================
# PUBLIC INTERFACE
# ============================================================

def get_spot_df(start_ts_ms, end_ts_ms, bar='1D') -> pd.DataFrame:
    
    all_candles = []
    after = str(end_ts_ms + (15 * 60 * 1000))
    
    # stop point depends on bar size
    if bar == '1D':
        # need 1 year back for P-measure estimation
        stop_ts = start_ts_ms - (365 * 24 * 60 * 60 * 1000)
    else:
        # just the observation window
        stop_ts = start_ts_ms
    
    while True:
        params = {
            'instId': 'BTC-USDT',
            'bar': bar,
            'after': after,
            'limit': 300
        }
        response = requests.get(OKX_URL, params=params).json()
        
        if 'data' not in response or not response['data']:
            break
        
        candles = response['data']
        all_candles.extend(candles)
        
        oldest_ts = int(candles[-1][0])
        if oldest_ts <= stop_ts:
            break
        
        after = str(oldest_ts)
    
    if not all_candles:
        raise ValueError(f"No spot data for bar={bar}")
    
    df = pd.DataFrame(
        all_candles,
        columns=['ts','open','high','low','close',
                 'vol','volCcy','volCcyQuote','confirm']
    )
    df['ts'] = df['ts'].astype(int)
    df['close_usd'] = df['close'].astype(float)
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.sort_values('ts').reset_index(drop=True)
    
    # filter to requested window
    df = df[
        (df['ts'] >= stop_ts) & 
        (df['ts'] <= end_ts_ms)
    ].copy()
    
    if bar == '1D':
        df['date'] = df['datetime'].dt.date
        df['log_return'] = np.log(
            df['close_usd'] / df['close_usd'].shift(1)
        )
        df['realized_var'] = (
            df['log_return'].rolling(21).var() * 252
        )
        return df[['ts','datetime','date','close_usd',
                   'log_return','realized_var']]
    else:
        return df[['ts','datetime','close_usd']]
    

def get_orderbook_df(data_dir, days, expiries) -> pd.DataFrame:
    """
    Extract top of book (best bid, best ask) for all contracts
    across all days. Returns raw tick-by-tick data — 
    downsampling and filtering happens in the notebook.
    
    Parameters:
        data_dir : str   path to folder containing tar.gz files
        days     : range e.g. range(1, 16)
        expiries : list  e.g. ['260529', '260626', '260731', '260925']
    
    Returns DataFrame with columns:
        ts            int      Unix ms
        datetime      datetime
        expiry        str      e.g. '260626'
        strike        int      strike in USD
        option_type   str      'C' or 'P'
        best_bid_btc  float    best bid in BTC
        best_ask_btc  float    best ask in BTC
        mid_btc       float    mid price in BTC
    """
    all_records = []
    
    for day in days:
        date_str = f'2026-05-{day:02d}'
        gz_path = os.path.join(
            data_dir,
            f'BTC-USD-optionchain-L2orderbook-400lv-{date_str}.tar.gz'
        )
        
        if not os.path.exists(gz_path):
            print(f"Missing: {gz_path}")
            continue
        
        print(f"Processing {date_str}...")
        day_records = 0
        
        with tarfile.open(gz_path, 'r:gz') as tar:
            all_files = tar.getnames()
            
            # filter to requested expiries only
            relevant = []
            for filepath in all_files:
                try:
                    expiry, strike, opt_type = _parse_filename(filepath)
                    if expiry in expiries:
                        relevant.append((filepath, expiry, strike, opt_type))
                except:
                    continue
            
            for filepath, expiry, strike, opt_type in relevant:
                try:
                    f = tar.extractfile(filepath)
                    records = _get_snapshots_only(f)
                    
                    for rec in records:
                        rec['expiry'] = expiry
                        rec['strike'] = strike
                        rec['option_type'] = opt_type
                    
                    all_records.extend(records)
                    day_records += len(records)
                    
                except Exception as e:
                    continue
        
        print(f"  {date_str}: {day_records} records across {len(relevant)} contracts")
    
    if not all_records:
        raise ValueError("No top of book data extracted")
    
    df = pd.DataFrame(all_records)
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.sort_values(['ts','expiry','strike']).reset_index(drop=True)
    
    return df[['ts','datetime','expiry','strike','option_type',
               'best_bid_btc','best_ask_btc','mid_btc']]

def get_funding_df(filepath) -> pd.DataFrame:
    """
    Load BTC-USD-SWAP funding rate history from CSV.
    
    Parameters:
        filepath : str  path to funding rate CSV file
    
    Returns DataFrame with columns:
        ts                int      Unix ms
        datetime          datetime
        funding_rate      float    8-hourly funding rate
        funding_rate_ann  float    annualized funding rate
    """
    df = pd.read_csv(filepath)
    
    df = df.rename(columns={'funding_time': 'ts'})
    df['ts'] = df['ts'].astype(int)
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    
    # annualize — 3 funding periods per day, 365 days
    df['funding_rate_ann'] = df['funding_rate'] * 3 * 365
    
    df = df[['ts', 'datetime', 'funding_rate', 'funding_rate_ann']]
    df = df.sort_values('ts').reset_index(drop=True)
    
    return df

def get_funding_df(data_dir, days) -> pd.DataFrame:
    """
    Load BTC-USD-SWAP funding rates from daily zip files.
    Filters to BTC-USD-SWAP only.
    
    Parameters:
        data_dir : str   path to folder containing zip files
        days     : range e.g. range(1, 16)
    
    Returns DataFrame with columns:
        ts                int
        datetime          datetime
        funding_rate      float    8-hourly rate
        funding_rate_ann  float    annualized rate
    """
    import zipfile
    
    all_records = []
    
    for day in days:
        date_str = f'2026-05-{day:02d}'
        zip_path = os.path.join(
            data_dir,
            f'allswap-fundingrates-{date_str}.zip'
        )
        
        if not os.path.exists(zip_path):
            print(f"Missing: {zip_path}")
            continue
        
        with zipfile.ZipFile(zip_path, 'r') as z:
            # get csv filename inside zip
            csv_files = [f for f in z.namelist() if f.endswith('.csv')]
            
            for csv_file in csv_files:
                with z.open(csv_file) as f:
                    df = pd.read_csv(f)
                    
                    # filter to BTC-USD-SWAP only
                    df = df[df['instrument_name'] == 'BTC-USD-SWAP']
                    
                    if not df.empty:
                        all_records.append(df)
    
    if not all_records:
        raise ValueError("No funding rate data found")
    
    df = pd.concat(all_records, ignore_index=True)
    df = df.rename(columns={'funding_time': 'ts'})
    df['ts'] = df['ts'].astype(int)
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df['funding_rate_ann'] = df['funding_rate'] * 3 * 365
    df = df.sort_values('ts').reset_index(drop=True)
    
    return df[['ts', 'datetime', 'funding_rate', 'funding_rate_ann']]


def get_trades_df(filepath, expiry=None, 
                  strike=None, contract_type=None) -> pd.DataFrame:
    """
    Load trade data from CSV.
    If expiry/strike/contract_type provided — filter to chosen contract.
    If not provided — return all trades.
    
    Returns DataFrame with columns:
        ts, datetime, trade_id, side, price_btc, size, instrument_name
    """
    df = pd.read_csv(filepath)
    df['ts'] = df['created_time'].astype(int)
    df['datetime'] = pd.to_datetime(df['ts'], unit='ms')
    df = df.rename(columns={'price': 'price_btc'})
    df = df.sort_values('ts').reset_index(drop=True)
    
    if expiry and strike and contract_type:
        instrument = f'BTC-USD-{expiry}-{int(strike)}-{contract_type}'
        df = df[df['instrument_name'] == instrument].copy()
    
    return df[['ts','datetime','trade_id','side',
               'price_btc','size','instrument_name']]