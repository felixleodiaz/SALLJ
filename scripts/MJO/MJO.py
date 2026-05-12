# MJO and SALLJ comparison
# Uses Wheeler & Hendon (2004) MJO index and Wang & Fu (2004) SALLJ index

import os
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from joblib import Parallel, delayed

# paths and directories

SCRIPT_DIR  = Path(__file__).resolve().parent          # SALLJ/scripts/MJO/
PROJECT_DIR = SCRIPT_DIR.parent.parent                 # SALLJ/
FIG_DIR     = PROJECT_DIR / 'figures' / 'MJO'
DATA_DIR    = PROJECT_DIR / 'local_data' / 'MJO'

N_BOOTSTRAP      = 5000
MAX_LAG          = 60
AMPLITUDE_THRESH = 1.0

SEASONS = {
    'ALL': list(range(1, 13)),
    'DJF': [12, 1, 2],
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
}

phases = range(1, 9)
lags   = range(0, MAX_LAG + 1)

phase_labels = {
    1: "Indian Ocean",
    2: "Indian Ocean",
    3: "Maritime Continent",
    4: "Maritime Continent",
    5: "West Pacific",
    6: "West Pacific",
    7: "West. Hem & Africa",
    8: "West. Hem & Africa",
}

# helper functions

def merge_with_mjo(llj_subset, mjo_subset, amplitude_thresh):
    """
    Join one member's LLJ series with THAT specific member's MJO index.
    """
    llj_indexed = llj_subset.set_index('time')[['llj_anom']]
    mjo_indexed = mjo_subset.set_index('time')[['phase', 'amplitude']]
    
    merged = llj_indexed.join(mjo_indexed, how='inner')
    merged = merged[merged['amplitude'] > amplitude_thresh].copy()
    return merged

def bootstrap_lag_cell(vals, pool, n_boot, rng_seed):
    """
    Compute mean, p-value, and 95% CI for one (phase, lag) cell.
    """
    n = len(vals)
    if n < 3:
        return dict(mean=np.nan, p=np.nan, ci_lo=np.nan, ci_hi=np.nan, n=n)

    rng      = np.random.default_rng(rng_seed)
    obs_mean = vals.mean()

    # null distribution
    null = np.array([
        rng.choice(pool, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    p = np.mean(np.abs(null) >= np.abs(obs_mean))

    # CI on observed mean
    boot_obs = np.array([
        rng.choice(vals, size=n, replace=True).mean()
        for _ in range(n_boot)
    ])
    return dict(
        mean  = obs_mean,
        p     = p,
        ci_lo = np.percentile(boot_obs, 2.5),
        ci_hi = np.percentile(boot_obs, 97.5),
        n     = n,
    )

def run_lag_composites(merged_all_members, llj_full_series, n_boot, max_lag, n_jobs):
    """
    Compute lag composites across all merged member data using the full series for lags.
    """
    lags_r = range(0, max_lag + 1)
    pool   = merged_all_members['llj_anom'].values   # null pool = active MJO days

    # pre-gather all values for each (phase, lag) cell
    cells = {}
    for phase in range(1, 9):
        phase_dates = merged_all_members[
            merged_all_members['phase'] == phase
        ].index.normalize()
        for lag in lags_r:
            lagged = phase_dates + pd.Timedelta(days=lag)
            valid  = lagged[lagged.isin(llj_full_series.index)]
            # look up from full series — not restricted to active MJO days
            cells[(phase, lag)] = llj_full_series.loc[valid].values

    # parallel bootstrap over all cells
    keys    = list(cells.keys())
    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(bootstrap_lag_cell)(cells[k], pool, n_boot, seed)
        for seed, k in enumerate(keys)
    )
    result_map = dict(zip(keys, results))

    # pack into arrays
    means = np.full((8, max_lag + 1), np.nan)
    pvals = np.full((8, max_lag + 1), np.nan)
    ci_lo = np.full((8, max_lag + 1), np.nan)
    ci_hi = np.full((8, max_lag + 1), np.nan)
    ns    = np.zeros((8, max_lag + 1), dtype=int)

    for (phase, lag), r in result_map.items():
        means[phase - 1, lag] = r['mean']
        pvals[phase - 1, lag] = r['p']
        ci_lo[phase - 1, lag] = r['ci_lo']
        ci_hi[phase - 1, lag] = r['ci_hi']
        ns[phase - 1, lag]    = r['n']

    return dict(means=means, pvals=pvals, ci_lo=ci_lo, ci_hi=ci_hi, ns=ns)

def save_lag_table(means, pvals, ci_lo, ci_hi, ns, max_lag, tag):
    """Save lag composite arrays as a tidy CSV."""
    rows = []
    for p in range(1, 9):
        for l in range(0, max_lag + 1):
            rows.append({
                'phase':   p,
                'lag':     l,
                'mean':    means[p-1, l],
                'p_value': pvals[p-1, l],
                'ci_low':  ci_lo[p-1, l],
                'ci_high': ci_hi[p-1, l],
                'n':       ns[p-1, l],
            })
    df = pd.DataFrame(rows)
    fpath = FIG_DIR / f'mjo_sallj_lag_table_{tag}.csv'
    df.to_csv(fpath, index=False)
    print(f"  Saved: {fpath}")
    return df

import numpy as np
import matplotlib.pyplot as plt

def plot_heatmap(means, pvals, max_lag, tag):
    # Note: I removed 'title_suffix' from the parameters since the title is gone.
    fig, ax = plt.subplots(figsize=(16, 6))
    
    vmax = np.nanmax(np.abs(means))
    vmax = max(vmax, 0.5)

    im = ax.imshow(
        means, aspect='auto', origin='upper',
        extent=[-0.5, max_lag + 0.5, 8.5, 0.5],
        cmap='RdBu', vmin=-vmax, vmax=vmax, interpolation='nearest',
    )
    
    # Colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("LLJ Anomaly (m/s)\nblue = stronger northerly jet", fontsize=10)

    # Stippling: p < 0.05 only (boxes removed)
    y05, x05 = np.where(pvals < 0.05)
    if len(x05):
        ax.scatter(x05, y05 + 1, marker='.', color='black',
                   s=25, zorder=5, label='p < 0.05')

    # Add crisp white space between grid cells using minor ticks
    ax.set_xticks(np.arange(-0.5, max_lag + 1.5, 1), minor=True)
    ax.set_yticks(np.arange(0.5, 9.5, 1), minor=True)
    ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False) # Hide minor tick marks
    
    # Remove outer black frame so the white grid bleeds to the edges
    for spine in ax.spines.values():
        spine.set_visible(False)

    # X-axis ticks: Display every 5 units
    x_ticks = list(range(0, max_lag + 1, 5))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"+{l}d" for l in x_ticks], fontsize=10)
    
    # Y-axis ticks
    ax.set_yticks(range(1, 9))
    ax.set_yticklabels(
        [f"Phase {p}  ({phase_labels[p]})" for p in range(1, 9)], fontsize=10
    )
    
    # Labels
    ax.set_xlabel("Lag (days after MJO phase)", fontsize=11)
    ax.set_ylabel("MJO Phase", fontsize=11)

    # Cleaned-up legend
    dot_h = plt.Line2D([0], [0], marker='.', color='black',
                       ls='none', ms=8, label='p < 0.05')
    ax.legend(handles=[dot_h], fontsize=10, loc='lower right', framealpha=0.9)

    plt.tight_layout()
    fpath = FIG_DIR / f'mjo_sallj_heatmap_{tag}.png'
    # Bumped DPI to 300 for a sharper, professional export
    plt.savefig(fpath, dpi=300, bbox_inches='tight') 
    plt.close()
    
    print(f"  Saved: {fpath}")


# Main Execution Block

if __name__ == '__main__':

    os.makedirs(FIG_DIR,  exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"Script dir : {SCRIPT_DIR}")
    print(f"Project dir: {PROJECT_DIR}")
    print(f"Figure dir : {FIG_DIR}")
    print(f"Data dir   : {DATA_DIR}")

    # Dynamically grab CPUs from SLURM, falling back to 32 if not found
    N_JOBS = int(os.environ.get('SLURM_CPUS_PER_TASK', 32))

    # 1. Load Model MJO index
    MJO_FILE = DATA_DIR / 'model_mjo_index.csv'

    if not MJO_FILE.exists():
        raise FileNotFoundError(
            f"Model MJO index not found at {MJO_FILE}\n"
            "Run calc_model_mjo.py first to generate this file."
        )

    model_mjo = pd.read_csv(MJO_FILE, parse_dates=['time'])
    model_mjo['time'] = pd.to_datetime(model_mjo['time']).dt.normalize()

    print(f"Model MJO data loaded: {len(model_mjo)} rows")

    # 2. Load LLJ data
    LLJ_FILE = DATA_DIR / 'sallj_index_full.csv'

    if not LLJ_FILE.exists():
        raise FileNotFoundError(
            f"LLJ index not found at {LLJ_FILE}\n"
            "Run SALLJ.py first to generate this file."
        )

    llj_full = pd.read_csv(LLJ_FILE, parse_dates=['time'])
    llj_full['time'] = pd.to_datetime(llj_full['time']).dt.normalize()

    print(f"LLJ data loaded: {len(llj_full)} rows")
    print(f"Members: {llj_full['member'].unique()}")
    print(f"Experiments: {llj_full['experiment'].unique()}")
    print(f"Date range: {llj_full['time'].min().date()} to {llj_full['time'].max().date()}")

    # 3. Historical Composites
    hist_llj = llj_full[llj_full['experiment'] == 'historical'].copy()

    print(f"\nHistorical LLJ rows: {len(hist_llj)}")
    print(f"Unique members: {hist_llj['member'].unique()}")

    for season_name, months in SEASONS.items():
        print(f"\n{'='*60}")
        print(f"  Season: {season_name}  (months {months})")
        print(f"{'='*60}")

        if season_name == 'ALL':
            subset = hist_llj.copy()
        else:
            subset = hist_llj[hist_llj['time'].dt.month.isin(months)].copy()

        member_merged = []
        for member in subset['member'].unique():
            # Get LLJ data for this member
            m_data_llj = subset[subset['member'] == member][['time', 'llj_anom']]
            
            # Get MJO data for this specific member and experiment
            m_data_mjo = model_mjo[
                (model_mjo['member'] == member) & 
                (model_mjo['experiment'] == 'historical')
            ]
            
            merged = merge_with_mjo(m_data_llj, m_data_mjo, AMPLITUDE_THRESH)
            if len(merged) > 0:
                merged['member'] = member
                member_merged.append(merged)

        if not member_merged:
            print(f"  No active MJO days found for {season_name}, skipping.")
            continue

        merged_all = pd.concat(member_merged)
        n_total    = sum(len(m) for m in member_merged)
        print(f"  Total active MJO member-days: {n_total}")
        print(f"  Phase distribution:\n{merged_all['phase'].value_counts().sort_index()}")

        print(f"  Running bootstrap (N={N_BOOTSTRAP}, {N_JOBS} parallel workers)")

        llj_full_for_lag = (
            subset[['time', 'llj_anom']]
            .set_index('time')['llj_anom']
        )
        llj_full_for_lag.index = llj_full_for_lag.index.normalize()

        results = run_lag_composites(merged_all, llj_full_for_lag, N_BOOTSTRAP, MAX_LAG, N_JOBS)

        tag = f"hist_{season_name}"
        save_lag_table(
            results['means'], results['pvals'],
            results['ci_lo'], results['ci_hi'],
            results['ns'], MAX_LAG, tag
        )
        plot_heatmap(results['means'], results['pvals'], MAX_LAG, tag)

        comp_rows = {}
        for p in range(1, 9):
            comp_rows[p] = {
                'mean':   results['means'][p-1, 0],
                'ci_lo':  results['ci_lo'][p-1, 0],
                'ci_hi':  results['ci_hi'][p-1, 0],
                'p':      results['pvals'][p-1, 0],
                'sig_05': results['pvals'][p-1, 0] < 0.05,
                'sig_10': results['pvals'][p-1, 0] < 0.10,
            }
        comp_df = pd.DataFrame(comp_rows).T
        plot_phase_barplot(comp_df, tag)

    # 4. SSP585 Composites
    print(f"\n{'='*60}")
    print("  SSP585 composite (r1i1p1f1, all seasons)")
    print(f"{'='*60}")

    ssp_llj = llj_full[llj_full['experiment'] == 'ssp585'].copy()

    m_data_llj = ssp_llj[['time', 'llj_anom']]
    m_data_mjo = model_mjo[
        (model_mjo['member'] == 'r1i1p1f1') & 
        (model_mjo['experiment'] == 'ssp585')
    ]
    
    ssp_merged = merge_with_mjo(m_data_llj, m_data_mjo, AMPLITUDE_THRESH)
    
    if len(ssp_merged) > 0:
        print(f"    SSP585 active MJO days: {len(ssp_merged)}")
        
        ssp_full_for_lag = (
            ssp_llj[['time', 'llj_anom']]
            .set_index('time')['llj_anom']
        )
        ssp_full_for_lag.index = ssp_full_for_lag.index.normalize()

        ssp_results = run_lag_composites(
            ssp_merged, ssp_full_for_lag, N_BOOTSTRAP, MAX_LAG, N_JOBS
        )
        
        save_lag_table(
            ssp_results['means'], ssp_results['pvals'],
            ssp_results['ci_lo'], ssp_results['ci_hi'],
            ssp_results['ns'], MAX_LAG, 'ssp585_ALL'
        )
        
        # No more arbitrary observational cutoff limit! 
        plot_heatmap(
            ssp_results['means'], ssp_results['pvals'], MAX_LAG, 'ssp585_ALL',
            title_suffix='  |  Intrinsic Model MJO'
        )
    else:
        print("  No SSP585 active MJO overlap found.")

    # summary
    print("\n\nSummary of output files:")
    for f in sorted(FIG_DIR.iterdir()):
        size_kb = f.stat().st_size / 1024
        print(f"  {str(f):<70s}  {size_kb:6.1f} kB")

    # =========================================================================
    # NEW PLOTTING BLOCK: SALLJ Cross-Sections by Active MJO Phase
    # =========================================================================
    print("\nGenerating SALLJ Meridional Wind Cross-Sections for Active MJO Phases...")
    
    import xarray as xr
    import intake
    import dask
    from xmip.preprocessing import combined_preprocessing
    
    # 1. Connect to CMIP6 Catalog to grab Daily Meridional Wind (va)
    url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
    col = intake.open_esm_datastore(url)
    
    # We use the first historical member to save memory
    mem = 'r1i1p1f1'
    query_va = dict(
        source_id='MRI-ESM2-0',
        table_id='day',
        experiment_id='historical',
        variable_id='va',
        member_id=mem,
    )
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cat_va = col.search(**query_va)
        
    z_kwargs = {'consolidated': True, 'decode_times': True}
    with dask.config.set(**{'array.slicing.split_large_chunks': True}):
        va_dict = cat_va.to_dataset_dict(zarr_kwargs=z_kwargs, preprocess=combined_preprocessing)
        
    va_ds = va_dict['CMIP.MRI.MRI-ESM2-0.historical.day.gn']
    
    # 2. Extract 3D va cross-section data (~20.5S lat, 280-320 lon)
    cross_lat = -20.5
    lon_slice = slice(280, 320)
    
    va_cross = va_ds['va'].sel(member_id=mem).sel(lat=cross_lat, method='nearest').sel(lon=lon_slice)
    
    # Align time coordinates to Pandas DatetimeIndex
    try:
        va_time = va_cross.indexes['time'].to_datetimeindex().normalize()
    except AttributeError:
        va_time = pd.to_datetime(va_cross.time.values).normalize()
    va_cross['time'] = va_time
    
    # Convert Pa to hPa for coordinates
    va_cross = va_cross.assign_coords(plev=va_cross.plev / 100)
    
    print("  Computing climatology...")
    va_clim = va_cross.mean(dim='time').load()
    
    # 3. Get Active MJO Dates for this member
    mjo_mem = model_mjo[(model_mjo['member'] == mem) & 
                        (model_mjo['experiment'] == 'historical') & 
                        (model_mjo['amplitude'] > AMPLITUDE_THRESH)]
    
    # 4. Plot an 8-panel composite (2 rows, 4 columns)
    fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(16, 8), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for phase in range(1, 9):
        print(f"  Computing va composite for Phase {phase}...")
        phase_dates = mjo_mem[mjo_mem['phase'] == phase]['time']
        
        # Filter for dates that exist in our loaded 3D dataset
        valid_dates = phase_dates[phase_dates.isin(va_cross.time.values)]
        
        # Calculate Phase Mean and subtract Climatology to get the Anomaly
        va_phase = va_cross.sel(time=valid_dates).mean(dim='time')
        va_anom = (va_phase - va_clim).load()
        
        ax = axes[phase-1]
        
        # Color contours for meridional wind anomalies
        # Setting vmax to 3.0 m/s to capture the anomaly range cleanly
        vmax = 3.0 
        cf = ax.contourf(
            va_anom.lon, va_anom.plev, va_anom, 
            levels=np.linspace(-vmax, vmax, 13), 
            cmap='RdBu_r', 
            extend='both'
        )
        
        # Invert Y-axis for standard atmospheric view
        ax.set_ylim(1000, 300) 
        ax.set_title(f'Phase {phase}: {phase_labels[phase]}', fontsize=12, fontweight='bold')
        
        # Format axes for the grid
        if phase in [1, 5]:
            ax.set_ylabel('Pressure (hPa)', fontsize=11)
        if phase >= 5:
            ax.set_xlabel('Longitude (°E)', fontsize=11)
            
    # Format and save
    plt.tight_layout(rect=[0, 0, 0.92, 0.95])
    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    fig.colorbar(cf, cax=cbar_ax, label='Meridional Wind Anomaly (m/s)')
    
    out_path = FIG_DIR / 'mjo_sallj_cross_section_phases.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {out_path}")

    print("\nCompositing complete.")