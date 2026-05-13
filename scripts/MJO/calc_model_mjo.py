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

def compute_wh2004_anomalies(da, detrend=False):
    """
    Calculate anomalies using 120 day rolling mean.
    If detrend=True, removes the linear secular trend over the time dimension 
    prior to calculating the Day-Of-Year climatology.
    """
    if detrend:
        # Fit a 1st degree polynomial (linear trend) along time and subtract it
        trend = da.polyfit(dim='time', deg=1)
        fit = xr.polyval(da['time'], trend.polyfit_coefficients)
        da = da - fit
        
    # 1. Remove day-of-year mean
    clim = da.groupby('time.dayofyear').mean('time')
    anom = da.groupby('time.dayofyear') - clim
    
    # fix dates to string before converting to a series
    if isinstance(da.indexes['time'], xr.CFTimeIndex):
        da['time'] = da.indexes['time'].to_datetimeindex()

    # 2. Remove 120-day trailing mean (interannual/ENSO filter)
    # Pandas rolling is robust for this continuous daily data
    rolling_mean = anom.to_series().unstack('lon').rolling(window=120, min_periods=1).mean()
    rolling_mean_xr = xr.DataArray(rolling_mean, dims=['time', 'lon'], coords=[anom.time, anom.lon])
    
    # Final anomaly
    intraseasonal_anom = anom - rolling_mean_xr
    
    return intraseasonal_anom

def calculate_mjo_index(ds_mem, member_name, experiment_name, detrend=False):
    """
    Calculates combined EOFs, PCs, Amplitude, and Phase.
    """
    print(f"  Computing anomalies for {member_name} ({experiment_name})...")
    if detrend:
        print("  Detrending active to remove secular background state trends...")
        
    olr_anom  = compute_wh2004_anomalies(ds_mem['olr'], detrend=detrend)
    u850_anom = compute_wh2004_anomalies(ds_mem['u850'], detrend=detrend)
    u200_anom = compute_wh2004_anomalies(ds_mem['u200'], detrend=detrend)
    
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
    
    # =========================================================
    # EOF ALIGNMENT & SIGN CONVENTION CHECK
    # Enforce Phase 1 = Indian Ocean, Phase 3 = Maritime Continent
    # =========================================================
    lon = ds_mem.lon.values
    n_lon = len(lon)
    
    # Isolate OLR loadings for EOF1 and EOF2
    eof1_olr = Vt[0, :n_lon]
    eof2_olr = Vt[1, :n_lon]
    
    io_idx = (lon >= 60) & (lon <= 90)     # Indian Ocean
    mc_idx = (lon >= 110) & (lon <= 140)   # Maritime Continent
    
    # Calculate spatial mean loadings for both EOFs
    eof1_io_load = eof1_olr[io_idx].mean()
    eof2_io_load = eof2_olr[io_idx].mean()
    eof1_mc_load = eof1_olr[mc_idx].mean()
    eof2_mc_load = eof2_olr[mc_idx].mean()
    
    # 1. Swap check: Ensure EOF1 captures IO and EOF2 captures MC
    if abs(eof2_io_load) > abs(eof1_io_load):
        print("  Swapping EOF1/EOF2 to align with standard spatial modes (EOF1=IO, EOF2=MC).")
        U[:, [0, 1]] = U[:, [1, 0]]
        S[[0, 1]] = S[[1, 0]]
        Vt[[0, 1], :] = Vt[[1, 0], :]
        
        eof1_olr = Vt[0, :n_lon]
        eof2_olr = Vt[1, :n_lon]
        eof1_io_load = eof1_olr[io_idx].mean()
        eof2_mc_load = eof2_olr[mc_idx].mean()

    # Extract Principal Components (PCs)
    PC1 = U[:, 0] * S[0]
    PC2 = U[:, 1] * S[1]
    
    PC1 = PC1 / np.std(PC1)
    PC2 = PC2 / np.std(PC2)
    
    # 2. Sign check: Ensure PC1 > 0 means negative OLR in IO (Phase 1)
    if eof1_io_load > 0:
        print("  Flipping PC1 sign to ensure Phase 1 = enhanced IO convection.")
        PC1 = -PC1
        Vt[0, :] = -Vt[0, :]
        
    # 3. Sign check: Ensure PC2 > 0 means negative OLR in MC (Phase 3)
    if eof2_mc_load > 0:
        print("  Flipping PC2 sign to ensure Phase 3 = enhanced MC convection.")
        PC2 = -PC2
        Vt[1, :] = -Vt[1, :]
    # =========================================================

    # Calculate Amplitude and Phase
    angle = np.arctan2(PC2, PC1) * (180 / np.pi)
    angle = np.where(angle < 0, angle + 360, angle) # 0 to 360
    
    # Map degrees to 8 phases
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
    
    # Process Historical Members (No detrending needed, stationary climate assumed)
    for member in HIST_MEMBERS:
        if member not in hist_ds.member_id.values:
            continue
        print(f"\nProcessing MJO for historical {member}...")
        ds_mem = preprocess_and_load(hist_ds.sel(member_id=member), member, slice('1979-01-01', '2014-12-31'))
        df = calculate_mjo_index(ds_mem, member, 'historical', detrend=False)
        mjo_dfs.append(df)
        
    # Process SSP585 (Detrending activated to handle 85-year warming trend)
    print("\nProcessing MJO for ssp585 r1i1p1f1...")
    ds_mem = preprocess_and_load(ssp_ds.sel(member_id='r1i1p1f1'), 'r1i1p1f1', slice('2015-01-01', '2100-12-31'))
    df = calculate_mjo_index(ds_mem, 'r1i1p1f1', 'ssp585', detrend=True)
    mjo_dfs.append(df)
    
    final_mjo_df = pd.concat(mjo_dfs, ignore_index=True)
    out_path = DATA_DIR / 'model_mjo_index.csv'
    final_mjo_df.to_csv(out_path, index=False)
    
    print(f"\nFinished! Model MJO index saved to: {out_path}")
    print(final_mjo_df.head())
    
    client.close()