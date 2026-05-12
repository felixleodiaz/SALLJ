# Calculates Wheeler & Hendon (2004) indices intrinsically from model data
# to find when the MJO is active

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import intake
import dask
from dask.distributed import Client, LocalCluster
from xmip.preprocessing import combined_preprocessing

SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
DATA_DIR    = PROJECT_DIR / 'local_data' / 'MJO'

LAT_S, LAT_N = -15.0, 15.0
HIST_MEMBERS = ['r1i1p1f1', 'r2i1p1f1', 'r3i1p1f1', 'r4i1p1f1', 'r5i1p1f1']

# define functions for processing

def preprocess_and_load(ds, member_id, time_slice):
    """
    Subsets the tropics, and computes the meridional average, then loads into memory.
    Returns variables OLR, U850, and U200 as dataset.
    """
    ds_tropics = ds.sel(lat=slice(LAT_S, LAT_N), time=time_slice)
    
    # Meridional mean (weighted by cosine of latitude)
    weights = np.cos(np.deg2rad(ds_tropics.lat))
    ds_mean = ds_tropics.weighted(weights).mean(dim='lat')
    
    # We load into memory here because the resulting array (time, lon) is very small (~10MB)
    # and doing rolling means / EOFs is much faster in-memory than via Dask chunks.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        olr  = ds_mean['rlut'].load()
        u850 = ds_mean['ua'].sel(plev=85000, method='nearest').load()
        # Note: CMIP6 ua top-level is often 25000 or 20000; W&H uses 200hPa, but 250hPa is acceptable if 200 is missing.
        u200 = ds_mean['ua'].sel(plev=20000, method='nearest').load() 
    
    return xr.Dataset({'olr': olr, 'u850': u850, 'u200': u200})

def compute_wh2004_anomalies(da):
    """
    Calculate anomilies using 120 day rolling mean
    """
    # 1. Remove day-of-year mean
    clim = da.groupby('time.dayofyear').mean('time')
    anom = da.groupby('time.dayofyear') - clim
    
    # 2. Remove 120-day trailing mean (interannual/ENSO filter)
    # Pandas rolling is robust for this continuous daily data
    rolling_mean = anom.to_series().unstack('lon').rolling(window=120, min_periods=1).mean()
    rolling_mean_xr = xr.DataArray(rolling_mean, dims=['time', 'lon'], coords=[anom.time, anom.lon])
    
    # Final anomaly
    intraseasonal_anom = anom - rolling_mean_xr
    
    return intraseasonal_anom

def calculate_mjo_index(ds_mem, member_name, experiment_name):
    """
    Calculates combined EOFs, PCs, Amplitude, and Phase.
    """
    print(f"  Computing anomalies for {member_name} ({experiment_name})...")
    olr_anom  = compute_wh2004_anomalies(ds_mem['olr'])
    u850_anom = compute_wh2004_anomalies(ds_mem['u850'])
    u200_anom = compute_wh2004_anomalies(ds_mem['u200'])
    
    # W&H 2004 Normalization: divide by the global variance of each anomaly field
    olr_norm  = olr_anom / olr_anom.std()
    u850_norm = u850_anom / u850_anom.std()
    u200_norm = u200_anom / u200_anom.std()
    
    # Combine along the longitude axis (Feature space: 3 * n_lon)
    combined = np.concatenate([olr_norm.values, u850_norm.values, u200_norm.values], axis=1)
    
    # Drop rows with NaNs (first ~120 days might have rolling mean edge cases depending on min_periods)
    valid_idx = ~np.isnan(combined).any(axis=1)
    combined_valid = combined[valid_idx]
    time_valid = ds_mem.time.values[valid_idx]
    
    print("  Computing SVD/EOFs...")
    # SVD for EOFs
    U, S, Vt = np.linalg.svd(combined_valid, full_matrices=False)
    
    # Extract Principal Components (PCs)
    # Standardize PCs so they have unit variance (RMM index convention)
    PC1 = U[:, 0] * S[0]
    PC2 = U[:, 1] * S[1]
    PC1 = PC1 / np.std(PC1)
    PC2 = PC2 / np.std(PC2)
    
    # Calculate Amplitude and Phase
    # Note: the exact sign/order of EOF1 and EOF2 is arbitrary in SVD.
    # W&H Phase space: Phase 1 starts with negative PC1, negative PC2. 
    # arctan2 handles the signs automatically to give an angle, which we divide into 8 bins.
    angle = np.arctan2(PC2, PC1) * (180 / np.pi)
    angle = np.where(angle < 0, angle + 360, angle) # 0 to 360
    
    # Map degrees to 8 phases (45 degrees each, offset by 22.5 to center them)
    # This formula creates standard 1-8 phase assignment:
    phase = np.floor(((angle + 22.5) % 360) / 45) + 1
    
    amplitude = np.sqrt(PC1**2 + PC2**2)
    
    df = pd.DataFrame({
        'time': time_valid,
        'member': member_name,
        'experiment': experiment_name,
        'rmm1': PC1,
        'rmm2': PC2,
        'phase': phase.astype(int),
        'amplitude': amplitude
    })
    
    return df

# processing block

if __name__ == '__main__':
    os.makedirs(DATA_DIR, exist_ok=True)
    
    print("Initializing Dask cluster")
    n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))
    cluster = LocalCluster(n_workers=n_cpus, threads_per_worker=1, memory_limit='4GB', dashboard_address=':0')
    client = Client(cluster)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    
    # Query for historical
    query_hist = dict(
        source_id='MRI-ESM2-0',
        table_id='day',
        experiment_id='historical',
        variable_id=['rlut', 'ua'],
        member_id=HIST_MEMBERS,
    )
    # Query for ssp585
    query_ssp = dict(
        source_id='MRI-ESM2-0',
        table_id='day',
        experiment_id='ssp585',
        variable_id=['rlut', 'ua'],
        member_id='r1i1p1f1',
    )
    
    cat_hist = col.search(**query_hist)
    cat_ssp  = col.search(**query_ssp)
    
    cache_path = '/scratch/fld1/cmip_cache'
    storage_options = {'filecache': {'cache_storage': cache_path, 'target_protocol': 'gs'}}
    z_kwargs = {'consolidated': True, 'decode_times': True}
    
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        hist_dict = cat_hist.to_dataset_dict(zarr_kwargs=z_kwargs, storage_options=storage_options, preprocessing=combined_preprocessing)
        ssp_dict  = cat_ssp.to_dataset_dict(zarr_kwargs=z_kwargs, storage_options=storage_options, preprocessing=combined_preprocessing)
    
    hist_ds = hist_dict['CMIP.MRI.MRI-ESM2-0.historical.day.gn']
    ssp_ds  = ssp_dict['CMIP.MRI.MRI-ESM2-0.ssp585.day.gn']
    
    mjo_dfs = []
    
    # Process Historical Members
    for member in HIST_MEMBERS:
        if member not in hist_ds.member_id.values:
            continue
        print(f"\nProcessing MJO for historical {member}...")
        ds_mem = preprocess_and_load(hist_ds.sel(member_id=member), member, slice('1979-01-01', '2014-12-31'))
        df = calculate_mjo_index(ds_mem, member, 'historical')
        mjo_dfs.append(df)
        
    # Process SSP585
    print("\nProcessing MJO for ssp585 r1i1p1f1...")
    ds_mem = preprocess_and_load(ssp_ds.sel(member_id='r1i1p1f1'), 'r1i1p1f1', slice('2015-01-01', '2100-12-31'))
    df = calculate_mjo_index(ds_mem, 'r1i1p1f1', 'ssp585')
    mjo_dfs.append(df)
    
    final_mjo_df = pd.concat(mjo_dfs, ignore_index=True)
    out_path = DATA_DIR / 'model_mjo_index.csv'
    final_mjo_df.to_csv(out_path, index=False)
    
    print(f"\nFinished! Model MJO index saved to: {out_path}")
    print(final_mjo_df.head())
    
    client.close()