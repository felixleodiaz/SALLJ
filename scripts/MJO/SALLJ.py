# SALLJ Detection  —  HPCC version
# Methodology: Wang & Fu (2004), J. Climate 17:1247-1262

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
FIG_DIR     = PROJECT_DIR / 'figures' / 'MJO'
DATA_DIR    = PROJECT_DIR / 'local_data' / 'MJO'

# Create output directories if they don't exist yet.
# exist_ok=True means this is safe to call even if they already exist.

os.makedirs(FIG_DIR,  exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print(f"Script dir : {SCRIPT_DIR}")
print(f"Project dir: {PROJECT_DIR}")
print(f"Figure dir : {FIG_DIR}")
print(f"Data dir   : {DATA_DIR}")

# 1: setup dask client

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

# 2: setup catalog query

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
    variable_id='va',
    member_id=HIST_MEMBERS,
)

# ssp585

query_ssp = dict(
    source_id='MRI-ESM2-0',
    table_id='day',
    experiment_id='ssp585',
    variable_id='va',
    member_id='r1i1p1f1',
)

cat_hist = col.search(**query_hist)
cat_ssp  = col.search(**query_ssp)

print("Historical entries found:")
print(cat_hist.df[['experiment_id', 'member_id']].drop_duplicates().to_string())
print("\nSSP585 entries found:")
print(cat_ssp.df[['experiment_id', 'member_id']].drop_duplicates().to_string())

# 3: load in data

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

# 4: constants and parameters
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

# 5: process each historical member to compute LLJ index

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

hist_series_list = []

for member in HIST_MEMBERS:

    # find the dataset key for this member
    key = [k for k in hist_dict if member in k]
    if not key:
        print(f"WARNING: {member} not found in loaded datasets, skipping.")
        continue
    ds = hist_dict[key[0]]
    print(f"Processing historical {member}...")
    s = compute_llj_index(
        ds, LAT_S, LAT_N, LON_W, LON_E,
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

def remove_seasonal_cycle(df, clim):
    """Subtract calendar-month climatology from each column."""
    anom = df.copy()
    for col in df.columns:
        anom[col] = df[col] - df.index.map(lambda d: clim[d.month])
    return anom

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

def flag_active(series, thresholds):
    flags = pd.Series(False, index=series.index)
    for month, thresh in thresholds.items():
        mask = series.index.month == month
        flags[mask] = series[mask] <= thresh
    return flags

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

client.close()
print("\nDask client closed. Detection complete.")