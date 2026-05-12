# SALLJ detection script
# We use the method from Wang & Fu (2004)

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
from matplotlib import pyplot as plt
import seaborn as sns


SCRIPT_DIR  = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent.parent
FIG_DIR     = PROJECT_DIR / 'figures' / 'MJO'
DATA_DIR    = PROJECT_DIR / 'local_data' / 'MJO'

# Create output directories if they don't exist yet.
# exist_ok=True means this is safe to call even if they already exist.

# 1: constants and parameters
# Wang & Fu LLJ domain: 15–25°S, 55–65°W but converted to 0→360

LAT_S, LAT_N = -25.0, -15.0
LON_W, LON_E = 295.0, 305.0
PLEV_850     = 85000.0

# RMM index only available from 1974; use 1979 to align with ERA reanalysis
# and avoid early-record quality issues in the RMM dataset.

HIST_START = '1979-01-01'
HIST_END   = '2014-12-31'
SSP_START  = '2015-01-01'
SSP_END    = '2100-12-31'

# define helper functions

def compute_llj_index(ds_full, lat_s, lat_n, lon_w, lon_e,
                    plev_850, time_start, time_end, member_id):
    """
    Extract 850-hPa meridional wind over the LLJ domain, compute
    area-weighted mean, return as a named daily pd.Series.
    """
    ds = ds_full.sel(
        time=slice(time_start, time_end),
        lat=slice(lat_s, lat_n),
        lon=slice(lon_w, lon_e),
    )
    va = ds['va'].sel(plev=plev_850, method='nearest')

    # compute weights and load, domain is small so this is fast

    weights = np.cos(np.deg2rad(va.lat)).broadcast_like(va.isel(time=0))
    llj_xr  = va.weighted(weights).mean(dim=['lat', 'lon'])

    series = (
        llj_xr
        .reset_coords(drop=True)
        .squeeze()
        .load()
        .to_series()
    )
    series.index = pd.DatetimeIndex(series.index).normalize()
    series.name  = member_id
    return series

def remove_seasonal_cycle(df, clim):
    """Subtract calendar-month climatology from each column."""
    anom = df.copy()
    for col in df.columns:
        anom[col] = df[col] - df.index.map(lambda d: clim[d.month])
    return anom

def flag_active(series, thresholds):
    flags = pd.Series(False, index=series.index)
    for month, thresh in thresholds.items():
        mask = series.index.month == month
        flags[mask] = series[mask] <= thresh
    return flags

# 2: setup dask client
if __name__ == '__main__':

    os.makedirs(FIG_DIR,  exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Script dir : {SCRIPT_DIR}")
    print(f"Project dir: {PROJECT_DIR}")
    print(f"Figure dir : {FIG_DIR}")
    print(f"Data dir   : {DATA_DIR}")

    print("Initializing Dask cluster")
    n_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', 4))

    cluster = LocalCluster(

        n_workers=n_cpus,
        threads_per_worker=1,
        memory_limit='4GB', 
        dashboard_address=':0' 
    )

    client = Client(cluster)
    print(client)
    print(f"Dashboard: {client.dashboard_link}\n")

    # 3: setup catalog query

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        col = intake.open_esm_datastore(
            "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
        )

    # using multiple historical members

    HIST_MEMBERS = ['r1i1p1f1', 'r2i1p1f1', 'r3i1p1f1', 'r4i1p1f1', 'r5i1p1f1']

    query_hist = dict(
        source_id='MRI-ESM2-0',
        table_id='day',
        experiment_id='historical',
        variable_id=['va', 'ua', 'zg'],
        member_id=HIST_MEMBERS,
    )

    # ssp585

    query_ssp = dict(
        source_id='MRI-ESM2-0',
        table_id='day',
        experiment_id='ssp585',
        variable_id=['va', 'ua', 'zg'],
        member_id='r1i1p1f1',
    )

    cat_hist = col.search(**query_hist)
    cat_ssp  = col.search(**query_ssp)

    print("Historical entries found:")
    print(cat_hist.df[['experiment_id', 'member_id']].drop_duplicates().to_string())
    print("\nSSP585 entries found:")
    print(cat_ssp.df[['experiment_id', 'member_id']].drop_duplicates().to_string())

    # 4: load in data

    cache_path = '/scratch/fld1/cmip_cache'
    storage_options = {
        'filecache': {
            'cache_storage': cache_path,
            'target_protocol': 'gs',
        }
    }
    z_kwargs = {'consolidated': True, 'decode_times': True}
    warnings.filterwarnings("ignore")

    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        hist_dict = cat_hist.to_dataset_dict(
            zarr_kwargs=z_kwargs,
            storage_options=storage_options,
            preprocessing=combined_preprocessing,
        )
        ssp_dict = cat_ssp.to_dataset_dict(
            zarr_kwargs=z_kwargs,
            storage_options=storage_options,
            preprocessing=combined_preprocessing,
        )

    print("\nHistorical dataset keys:")
    for k in hist_dict: print(" ", k)
    print("SSP585 dataset keys:")
    for k in ssp_dict:  print(" ", k)

    # 5: process each historical member to compute LLJ index

    hist_series_list = []

    hist_ds = hist_dict['CMIP.MRI.MRI-ESM2-0.historical.day.gn']

    for member in HIST_MEMBERS:
        if member not in hist_ds.member_id.values:
            print(f"WARNING: {member} not found in dataset, skipping.")
            continue
        print(f"Processing historical {member}...")
        ds_member = hist_ds.sel(member_id=member)
        s = compute_llj_index(
            ds_member, LAT_S, LAT_N, LON_W, LON_E,
            PLEV_850, HIST_START, HIST_END, member
        )
        hist_series_list.append(s)
        print(f"  {member}: {len(s)} days, mean={s.mean():.2f} m/s")

    # combine all members into one data frame where each column is one member

    hist_df = pd.concat(hist_series_list, axis=1)
    hist_df.index.name = 'time'

    print(f"\nHistorical LLJ DataFrame shape: {hist_df.shape}")
    print("(rows=days, cols=members)")

    # 6: process ssp585 (only one member)

    ssp_key = list(ssp_dict.keys())[0]
    print(f"\nProcessing ssp585 ({ssp_key})...")
    ssp_series = compute_llj_index(
        ssp_dict[ssp_key], LAT_S, LAT_N, LON_W, LON_E,
        PLEV_850, SSP_START, SSP_END, 'r1i1p1f1_ssp585'
    )
    ssp_df = ssp_series.to_frame()
    ssp_df.index.name = 'time'
    print(f"SSP585: {len(ssp_series)} days, mean={ssp_series.mean():.2f} m/s")

    # 7: compute LLJ anomaly (seasonal cycle removed)
    # Climatology computed from full historical period across all members,
    # then applied to both historical and ssp585.

    # stack all historical values for climatology

    hist_stack = hist_df.stack().reset_index()
    hist_stack.columns = ['time', 'member', 'llj_index']
    hist_stack['month'] = hist_stack['time'].dt.month

    monthly_clim = hist_stack.groupby('month')['llj_index'].mean()
    print("\nHistorical monthly climatology (m/s):")
    month_labels = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                    7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
    for m, v in monthly_clim.items():
        print(f"  {month_labels[m]}: {v:+.2f}")

    hist_anom = remove_seasonal_cycle(hist_df, monthly_clim)
    ssp_anom  = remove_seasonal_cycle(ssp_df,  monthly_clim)

    # 8: active LLJ detection (monthly 20th-percentile threshold)
    # Threshold computed from pooled historical data across all members.

    hist_stack['llj_anom'] = hist_stack.apply(
        lambda r: hist_anom.loc[r['time'], r['member']], axis=1
    )

    monthly_thresh = (
        hist_stack
        .groupby('month')['llj_index']
        .quantile(0.20)
    )
    print("\nMonthly 20th-percentile thresholds (m/s):")
    for m, v in monthly_thresh.items():
        print(f"  {month_labels[m]}: {v:+.2f}")

    # 9: build long format output (one row per member-day)
    # the compositing script expects: time | member | experiment | llj_index | llj_anom | active

    long_rows = []

    for member in hist_df.columns:
        s      = hist_df[member]
        s_anom = hist_anom[member]
        active = flag_active(s, monthly_thresh)
        for date in s.index:
            long_rows.append({
                'time':       date,
                'member':     member,
                'experiment': 'historical',
                'llj_index':  s[date],
                'llj_anom':   s_anom[date],
                'active':     active[date],
            })

    # add ssp585

    s_ssp      = ssp_df['r1i1p1f1_ssp585']
    s_ssp_anom = ssp_anom['r1i1p1f1_ssp585']
    active_ssp = flag_active(s_ssp, monthly_thresh)
    for date in s_ssp.index:
        long_rows.append({
            'time':       date,
            'member':     'r1i1p1f1',
            'experiment': 'ssp585',
            'llj_index':  s_ssp[date],
            'llj_anom':   s_ssp_anom[date],
            'active':     active_ssp[date],
        })

    out_df = pd.DataFrame(long_rows)
    out_df['time'] = pd.to_datetime(out_df['time'])
    out_df = out_df.sort_values(['experiment', 'member', 'time']).reset_index(drop=True)

    print(f"\nOutput DataFrame shape: {out_df.shape}")
    print(out_df.head(10).to_string())

    # 10: save everything to DATA_DIR

    out_df.to_csv(DATA_DIR / 'sallj_index_full.csv', index=False)
    print(f"\nSaved: {DATA_DIR / 'sallj_index_full.csv'}")

    monthly_clim.to_csv(DATA_DIR / 'sallj_monthly_clim.csv', header=['llj_clim'])
    print(f"Saved: {DATA_DIR / 'sallj_monthly_clim.csv'}")

    monthly_thresh.to_csv(DATA_DIR / 'sallj_monthly_thresh.csv', header=['llj_thresh'])
    print(f"Saved: {DATA_DIR / 'sallj_monthly_thresh.csv'}")

    # 11: diagnostic figures

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    hist_only = out_df[out_df['experiment'] == 'historical'].copy()
    hist_only['time'] = pd.to_datetime(hist_only['time'])

    members      = sorted(hist_only['member'].unique())
    n_members    = len(members)
    month_abbrev = ['J','F','M','A','M','J','J','A','S','O','N','D']

    # Figure A: full timeseries, one panel per member

    fig_a, axes_a = plt.subplots(
        n_members, 1,
        figsize=(18, 3 * n_members),
        sharex=True,
    )
    if n_members == 1:
        axes_a = [axes_a]

    fig_a.suptitle(
        'SALLJ Index — Full Historical Period (1979–2014)\n'
        'MRI-ESM2-0  |  Wang & Fu (2004) methodology  |  '
        'Monthly 20th-pctile threshold',
        fontsize=11, fontweight='bold', y=1.01
    )

    for ax, member in zip(axes_a, members):
        m_data  = hist_only[hist_only['member'] == member].set_index('time')
        idx     = m_data['llj_index']
        active  = m_data['active']

        thresh_line = pd.Series(
            [monthly_thresh[d.month] for d in idx.index],
            index=idx.index
        )

        ax.plot(idx.index, idx.values, lw=0.5, color='steelblue', alpha=0.85)
        ax.fill_between(
            idx.index, idx.values, 0,
            where=active.values, alpha=0.4, color='crimson',
            label='Active LLJ'
        )
        ax.plot(thresh_line.index, thresh_line.values,
                color='crimson', lw=0.9, ls='--', alpha=0.7,
                label='Monthly 20th pctile')
        ax.axhline(0, color='black', lw=0.5)

        ax.set_ylabel('v-wind\n850 hPa (m/s)', fontsize=8)
        ax.set_title(f'Member: {member}', fontsize=9, loc='left', pad=3)
        ax.grid(alpha=0.15)

        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

        if ax == axes_a[0]:
            ax.legend(fontsize=8, loc='lower left', ncol=2)

    axes_a[-1].set_xlabel('Year', fontsize=9)
    plt.tight_layout()
    plt.savefig(FIG_DIR / 'sallj_timeseries_all_members.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIG_DIR / 'sallj_timeseries_all_members.png'}")

    # Figure B: multi-member monthly climatology

    fig_b, axes_b = plt.subplots(1, 2, figsize=(14, 5))
    fig_b.suptitle(
        'SALLJ Monthly Climatology  |  MRI-ESM2-0  |  1979–2014\n'
        'Bar = ensemble mean across 5 members  |  Error bar = 1σ across members',
        fontsize=10, fontweight='bold'
    )

    member_monthly = (
        hist_only
        .assign(month=hist_only['time'].dt.month)
        .groupby(['member', 'month'])['llj_index']
        .mean()
        .unstack('member')
    )
    ens_mean = member_monthly.mean(axis=1)
    ens_std  = member_monthly.std(axis=1)

    ax_b1 = axes_b[0]
    ax_b1.bar(range(1, 13), ens_mean.values,
            yerr=ens_std.values,
            color='steelblue', alpha=0.8, edgecolor='white',
            capsize=4, error_kw={'elinewidth': 1.2})
    ax_b1.axhline(0, color='black', lw=0.6)
    ax_b1.set_xticks(range(1, 13))
    ax_b1.set_xticklabels(month_abbrev)
    ax_b1.set_ylabel('Mean LLJ Index (m/s)')
    ax_b1.set_title('Ensemble-Mean Monthly Climatology')
    ax_b1.grid(alpha=0.2, axis='y')

    for member in members:
        m_clim = member_monthly[member]
        ax_b1.plot(range(1, 13), m_clim.values,
                color='steelblue', lw=0.8, alpha=0.4)

    active_freq = (
        hist_only
        .assign(month=hist_only['time'].dt.month)
        .groupby(['member', 'month'])['active']
        .mean()
        .unstack('member')
    )
    freq_mean = active_freq.mean(axis=1)
    freq_std  = active_freq.std(axis=1)

    ax_b2 = axes_b[1]
    ax_b2.bar(range(1, 13), freq_mean.values * 100,
            yerr=freq_std.values * 100,
            color='crimson', alpha=0.7, edgecolor='white',
            capsize=4, error_kw={'elinewidth': 1.2})
    ax_b2.axhline(20, color='black', lw=0.8, ls='--',
                label='Expected 20% (threshold definition)')
    ax_b2.set_xticks(range(1, 13))
    ax_b2.set_xticklabels(month_abbrev)
    ax_b2.set_ylabel('Active LLJ Frequency (%)')
    ax_b2.set_title('Active LLJ Frequency by Month\n(should be ~20% by construction)')
    ax_b2.legend(fontsize=8)
    ax_b2.grid(alpha=0.2, axis='y')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'sallj_climatology_all_members.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIG_DIR / 'sallj_climatology_all_members.png'}")

    # Figure C: annual mean LLJ index over time (trend check)

    fig_c, ax_c = plt.subplots(figsize=(12, 4))
    fig_c.suptitle(
        'SALLJ Annual Mean LLJ Index  |  MRI-ESM2-0  |  1979–2014\n'
        'Thin lines = individual members  |  Thick line = ensemble mean',
        fontsize=10, fontweight='bold'
    )

    annual_all = (
        hist_only
        .assign(year=hist_only['time'].dt.year)
        .groupby(['member', 'year'])['llj_index']
        .mean()
        .unstack('member')
    )

    for member in members:
        ax_c.plot(annual_all.index, annual_all[member].values,
                lw=0.9, alpha=0.45, color='steelblue')

    ens_annual_mean = annual_all.mean(axis=1)
    ax_c.plot(annual_all.index, ens_annual_mean.values,
            lw=2.5, color='steelblue', label='Ensemble mean')

    years_num = annual_all.index.values - annual_all.index.values.mean()
    valid     = ~np.isnan(ens_annual_mean.values)
    coeffs    = np.polyfit(years_num[valid], ens_annual_mean.values[valid], 1)
    trend     = np.polyval(coeffs, years_num)
    ax_c.plot(annual_all.index, trend, lw=1.5, color='crimson', ls='--',
            label=f'Trend: {coeffs[0]:+.3f} m/s yr⁻¹')

    ax_c.axhline(0, color='black', lw=0.5)
    ax_c.set_xlabel('Year')
    ax_c.set_ylabel('Annual Mean LLJ Index (m/s)')
    ax_c.legend(fontsize=9)
    ax_c.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'sallj_annual_trend.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {FIG_DIR / 'sallj_annual_trend.png'}")

    # seasonal Meridional Wind cross section plots

    print("\nGenerating Seasonal Cross-Section Plots...")
    
    # target slice

    cross_lat = -20.5
    lon_slice = slice(270, 330)
    
    # Extract and subset the 3D 'va' data (averaging across historical members)
    va_hist_sub = hist_ds['va'].sel(lat=cross_lat, method='nearest').sel(lon=lon_slice).mean(dim='member_id')
    va_ssp_sub  = ssp_dict[ssp_key]['va'].sel(lat=cross_lat, method='nearest').sel(lon=lon_slice).squeeze()
    
    # Define the same seasons used in MJO.py
    seasons_dict = {
        'ALL': list(range(1, 13)),
        'DJF': [12, 1, 2],
        'MAM': [3, 4, 5],
        'JJA': [6, 7, 8],
        'SON': [9, 10, 11],
    }
    
    hist_seasons = {}
    ssp_seasons  = {}
    anom_seasons = {}
    
    # Compute the seasonal time-means utilizing Dask
    for s_name, s_months in seasons_dict.items():
        print(f"  Computing mean for {s_name}...")
        if s_name == 'ALL':
            h_mean = va_hist_sub.mean(dim='time').load()
            s_mean = va_ssp_sub.mean(dim='time').load()
        else:
            # Filter by month and compute mean
            h_mean = va_hist_sub.isel(time=va_hist_sub.time.dt.month.isin(s_months)).mean(dim='time').load()
            s_mean = va_ssp_sub.isel(time=va_ssp_sub.time.dt.month.isin(s_months)).mean(dim='time').load()
            
        # Convert Pascals to hPa for plotting
        h_mean = h_mean.assign_coords(plev=h_mean.plev / 100)
        s_mean = s_mean.assign_coords(plev=s_mean.plev / 100)
        
        hist_seasons[s_name] = h_mean
        ssp_seasons[s_name]  = s_mean
        anom_seasons[s_name] = s_mean - h_mean

    # Define a helper function to plot the 1x5 grid
    def plot_seasonal_cross_section(data_dict, title_prefix, filename, cmap='RdBu_r'):
        sns.set_style('white')
        fig, axes = plt.subplots(nrows=1, ncols=5, figsize=(22, 4.5), sharey=True, sharex=True)
        
        # Determine global max for a symmetric colorbar centered on 0
        vmax = max([abs(v.max().item()) for v in data_dict.values()])
        vmin = -vmax
        
        for ax, (s_name, data) in zip(axes, data_dict.items()):
            im = data.plot(ax=ax, add_colorbar=False, cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_ylim(1000, 300)
            ax.set_title(f'{s_name}', fontsize=12)
            ax.set_xlabel('Longitude (°E)')
            
            if ax == axes[0]:
                ax.set_ylabel('Pressure (hPa)')
            else:
                ax.set_ylabel('')
        
        # Position colorbar
        cbar_ax = fig.add_axes([0.91, 0.15, 0.015, 0.7]) 
        fig.colorbar(im, cax=cbar_ax, label=f'{title_prefix} Meridional Wind Speed (m/s)')
        
        # Use tight_layout to handle spacing, leaving room for the manual colorbar
        plt.tight_layout(rect=[0, 0, 0.9, 1]) 
        
        out_path = FIG_DIR / filename
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {out_path}")

    # Execute plotting
    print("Plotting historical cross-sections")
    plot_seasonal_cross_section(hist_seasons, "Historical (1979-2014)", "sallj_cross_section_hist.png")
    
    print("Plotting SSP585 cross-sections")
    plot_seasonal_cross_section(ssp_seasons, "SSP585 (2015-2100)", "sallj_cross_section_ssp585.png")
    
    print("Plotting Anomaly cross-sections")
    plot_seasonal_cross_section(anom_seasons, "SSP585 Anomalous", "sallj_cross_section_anom.png")

    # Wang and Fu reversal picture

    # =========================================================================
    # Figure 9: Wang & Fu (2004) Reversal Composites
    # =========================================================================
    print("\nGenerating Wang & Fu Reversal Plot (Figure 9)...")
    
    # We will use member r1i1p1f1 for the 3D fields to save memory
    mem = 'r1i1p1f1'
    
    # Isolate the July index for this specific member
    idx_jul = hist_df[mem][hist_df.index.month == 7].copy()
    
    # Define the spatial domain for Figure 9 (Equator to 50S, 120W to 20W)
    # CMIP6 longitudes are 0-360, so 120W -> 240, 20W -> 340
    lat_slice = slice(-50, 0)
    lon_slice = slice(240, 340) 
    
    ds_mem = hist_ds.sel(member_id=mem)
    va_850 = ds_mem['va'].sel(plev=85000, method='nearest').sel(lat=lat_slice, lon=lon_slice)
    ua_850 = ds_mem['ua'].sel(plev=85000, method='nearest').sel(lat=lat_slice, lon=lon_slice)
    zg_700 = ds_mem['zg'].sel(plev=70000, method='nearest').sel(lat=lat_slice, lon=lon_slice)
    
    # Calculate index variance for the regression denominator
    X = xr.DataArray(idx_jul.values, coords=[('time', idx_jul.index)])
    X_var = X.var(dim='time')
    
    lags = [-4, -2, 0] # Day -4, Day -2, Day 0
    
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(8, 12), sharex=True, sharey=True)
    
    for i, lag in enumerate(lags):
        print(f"  Computing regression for lag {lag}...")
        
        # Shift the index to get the corresponding lagged dates
        lag_dates = idx_jul.index + pd.Timedelta(days=lag)
        
        # Keep only dates valid in the dataset
        valid_mask = lag_dates.isin(va_850.time.values)
        valid_base_dates = idx_jul.index[valid_mask]
        valid_lag_dates = lag_dates[valid_mask]
        
        # Subset the index and compute anomalies
        X_valid = xr.DataArray(idx_jul.loc[valid_base_dates].values, coords=[('time', valid_lag_dates)])
        X_valid = X_valid - X_valid.mean(dim='time')
        
        # Subset the 3D fields and compute anomalies
        v_field = va_850.sel(time=valid_lag_dates)
        u_field = ua_850.sel(time=valid_lag_dates)
        z_field = zg_700.sel(time=valid_lag_dates)
        
        v_field = v_field - v_field.mean(dim='time')
        u_field = u_field - u_field.mean(dim='time')
        z_field = z_field - z_field.mean(dim='time')
        
        # Calculate Regression slope and scale to +5 m/s southerly index
        cov_v = (v_field * X_valid).mean(dim='time')
        cov_u = (u_field * X_valid).mean(dim='time')
        cov_z = (z_field * X_valid).mean(dim='time')
        
        v_reg = (cov_v / X_var) * 5.0
        u_reg = (cov_u / X_var) * 5.0
        z_reg = (cov_z / X_var) * 5.0
        
        # Push through Dask workers
        v_reg, u_reg, z_reg = dask.compute(v_reg, u_reg, z_reg)
        
        # Plotting
        ax = axes[i]
        lons_plot = z_reg.lon - 360 # Convert to -180 to 180 for standard display
        lats_plot = z_reg.lat
        
        # Color contour for 700-hPa height anomalies
        cf = ax.contourf(lons_plot, lats_plot, z_reg, levels=np.linspace(-30, 30, 13), cmap='RdBu_r', extend='both')
        
        # Subsample vectors for readable quiver
        skip = 2 
        q = ax.quiver(lons_plot[::skip], lats_plot[::skip], u_reg.values[::skip, ::skip], v_reg.values[::skip, ::skip], 
                      color='black', scale=40, width=0.003)
                      
        ax.set_title(f'Day {lag}', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=10)
        
        if i == 2: # Bottom panel configuration
            ax.set_xlabel('Longitude', fontsize=10)
            ax.quiverkey(q, X=0.9, Y=-0.15, U=5, label='5 m/s', labelpos='E')
            
    fig.subplots_adjust(right=0.85)
    cbar_ax = fig.add_axes([0.88, 0.15, 0.03, 0.7])
    fig.colorbar(cf, cax=cbar_ax, label='700-hPa Geopotential Height Anomaly (m)')
    
    plt.suptitle("Reversal Composites (Scaled to 5 m/s Southerly LLJ)\n850-hPa Winds (vectors) & 700-hPa Height (color)", 
                 y=0.95, fontsize=14, fontweight='bold')
                 
    out_path = FIG_DIR / 'wang_and_fu_fig9_reversal.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

    # plot MJO during reversal events

# =========================================================================
    # NEW PLOTTING BLOCK: Lagged Large-Scale Pressure (500-hPa Height)
    # =========================================================================
    print("\nGenerating Lagged Reversal Composites (Days -2 to +2)...")
    
    # We will use the first historical member
    mem = 'r1i1p1f1'
    idx_all = hist_df[mem].copy()
    
    # 1. Define Base Reversal Days (Day 0)
    REVERSAL_THRESH = -2.0
    reversal_dates = idx_all[idx_all <= REVERSAL_THRESH].index
    print(f"  Found {len(reversal_dates)} base reversal days.")
    
    # 2. Extract 500-hPa Geopotential Height
    lat_wide = slice(-80, 20)
    lon_wide = slice(150, 360) 
    
    zg_500 = hist_ds.sel(member_id=mem)['zg'].sel(
        plev=50000, method='nearest'
    ).sel(lat=lat_wide, lon=lon_wide)
    
    # 3. Compute Climatology once
    print("  Computing climatology through Dask (this may take a moment)...")
    # We load this once to speed up the loop below
    zg_clim = zg_500.mean(dim='time').load() 
    
    # 4. Define Lags and Setup Plot
    lags = [-2, -1, 0, 1, 2]
    lag_names = ['Day -2', 'Day -1', 'Day 0 (Reversal)', 'Day +1', 'Day +2']
    
    # Tall figure to accommodate 5 rows
    fig, axes = plt.subplots(nrows=5, ncols=1, figsize=(12, 22), sharex=True, sharey=True)
    import matplotlib.patches as patches
    
    for i, lag in enumerate(lags):
        print(f"  Computing composite for {lag_names[i]}...")
        
        # Shift the dates by 'lag' days
        shifted_dates = reversal_dates + pd.Timedelta(days=lag)
        
        # Filter for dates that actually exist in the 3D dataset
        valid_dates = shifted_dates[shifted_dates.isin(zg_500.time.values)]
        
        # Calculate Mean for this lag and get Anomaly
        zg_lag_mean = zg_500.sel(time=valid_dates).mean(dim='time')
        zg_anom = (zg_lag_mean - zg_clim).load() 
        
        # Plotting
        ax = axes[i]
        lons = zg_anom.lon
        lats = zg_anom.lat
        
        cf = ax.contourf(
            lons, lats, zg_anom, 
            levels=np.linspace(-60, 60, 13), 
            cmap='PuOr_r', 
            extend='both'
        )
        
        # Draw bounding box
        sallj_box = patches.Rectangle(
            (295, -25), 10, 10, linewidth=2, edgecolor='black', facecolor='none', linestyle='--'
        )
        ax.add_patch(sallj_box)
        
        # Only add the text to the top plot to avoid clutter
        if i == 0:
            ax.text(300, -13, 'SALLJ Domain', color='black', weight='bold', ha='center')
            
        ax.set_title(f'{lag_names[i]}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Latitude', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.6)

    # X-axis label only goes on the bottom plot
    axes[-1].set_xlabel('Longitude (°E)', fontsize=12)
    
    # Setup shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(cf, cax=cbar_ax)
    cbar.set_label('500-hPa Geopotential Height Anomaly (m)', fontsize=12)
    
    plt.suptitle(f'Evolution of 500-hPa Height Anomalies Around SALLJ Reversals', 
                 fontsize=18, fontweight='bold', y=0.92)
    
    out_path = FIG_DIR / 'reversal_evolution_zg_anom.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

    # =========================================================================
    # NEW PLOTTING BLOCK: Large-Scale Pressure (500-hPa Height) 
    # Active Northerly vs. Reversed Southerly SALLJ
    # =========================================================================
    print("\nGenerating Large-Scale SALLJ Composites (Active vs Reversed)...")
    
    # We will use the first historical member
    mem = 'r1i1p1f1'
    idx_all = hist_df[mem].copy()
    
    # 1. Define Active and Reversal Days
    ACTIVE_THRESH = 2.0
    REVERSAL_THRESH = -2.0
    
    active_dates = idx_all[idx_all >= ACTIVE_THRESH].index
    reversal_dates = idx_all[idx_all <= REVERSAL_THRESH].index
    
    print(f"  Found {len(active_dates)} active northerly days.")
    print(f"  Found {len(reversal_dates)} reversed southerly days.")
    
    # 2. Extract 500-hPa Geopotential Height
    lat_wide = slice(-80, 20)
    lon_wide = slice(150, 360) 
    
    zg_500 = hist_ds.sel(member_id=mem)['zg'].sel(
        plev=50000, method='nearest'
    ).sel(lat=lat_wide, lon=lon_wide)
    
    # 3. Filter for dates that exist in our loaded 3D dataset
    valid_active = active_dates[active_dates.isin(zg_500.time.values)]
    valid_reversals = reversal_dates[reversal_dates.isin(zg_500.time.values)]
    
    # 4. Calculate Climatological Mean vs. Composite Means
    print("  Computing means through Dask (this may take a moment)...")
    zg_clim = zg_500.mean(dim='time')
    zg_act_mean = zg_500.sel(time=valid_active).mean(dim='time')
    zg_rev_mean = zg_500.sel(time=valid_reversals).mean(dim='time')
    
    # The Anomaly isolates the specific pressure signals
    zg_anom_act = (zg_act_mean - zg_clim).load()
    zg_anom_rev = (zg_rev_mean - zg_clim).load() 
    
    # 5. Plotting the 2-panel figure
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(12, 10), sharex=True, sharey=True)
    
    lons = zg_anom_act.lon
    lats = zg_anom_act.lat
    
    # Panel 1: Active Northerly
    cf1 = axes[0].contourf(lons, lats, zg_anom_act, levels=np.linspace(-60, 60, 13), cmap='PuOr_r', extend='both')
    axes[0].set_title(f'Active Northerly SALLJ (LLJ $\geq$ {ACTIVE_THRESH} m/s)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Latitude', fontsize=12)
    
    # Panel 2: Reversed Southerly
    cf2 = axes[1].contourf(lons, lats, zg_anom_rev, levels=np.linspace(-60, 60, 13), cmap='PuOr_r', extend='both')
    axes[1].set_title(f'Reversed Southerly SALLJ (LLJ $\leq$ {REVERSAL_THRESH} m/s)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Latitude', fontsize=12)
    axes[1].set_xlabel('Longitude (°E)', fontsize=12)
    
    import matplotlib.patches as patches
    for ax in axes:
        # Draw a bounding box for the general SALLJ region
        sallj_box = patches.Rectangle((295, -25), 10, 10, linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
        ax.add_patch(sallj_box)
        ax.grid(True, linestyle=':', alpha=0.6)
    
    # Setup shared colorbar
    fig.subplots_adjust(right=0.88)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(cf1, cax=cbar_ax)
    cbar.set_label('500-hPa Geopotential Height Anomaly (m)', fontsize=12)
    
    out_path = FIG_DIR / 'sallj_active_vs_reversed_zg_anom.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

    client.close()
    print("\nDask client closed. Detection complete.")