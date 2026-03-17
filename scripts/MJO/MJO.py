# MJO – SALLJ Compositing  —  HPCC version
# Wheeler & Hendon (2004) RMM × Wang & Fu (2004) LLJ index

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


SCRIPT_DIR  = Path(__file__).resolve().parent          # SALLJ/scripts/MJO/
PROJECT_DIR = SCRIPT_DIR.parent.parent                 # SALLJ/
FIG_DIR     = PROJECT_DIR / 'figures' / 'MJO'
DATA_DIR    = PROJECT_DIR / 'local_data' / 'MJO'

os.makedirs(FIG_DIR,  exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

print(f"Script dir : {SCRIPT_DIR}")
print(f"Project dir: {PROJECT_DIR}")
print(f"Figure dir : {FIG_DIR}")
print(f"Data dir   : {DATA_DIR}")

# configuration (reminder to self to match SLURM script)

N_BOOTSTRAP      = 5000
MAX_LAG          = 25
AMPLITUDE_THRESH = 1.0
N_JOBS           = 200

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

# load RMM index
# http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt
# in local_data/MJO/rmm_index.txt

RMM_FILE = DATA_DIR / 'rmm_index.txt'

if not RMM_FILE.exists():
    raise FileNotFoundError(
        f"RMM index not found at {RMM_FILE}\n"
        "Download from: http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt\n"
        f"and save to: {RMM_FILE}"
    )

rmm_rows = []
with open(RMM_FILE, 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        try:
            year = int(parts[0])
        except ValueError:
            continue
        rmm_rows.append({
            'year': year, 'month': int(parts[1]), 'day': int(parts[2]),
            'rmm1': float(parts[3]), 'rmm2': float(parts[4]),
            'phase': int(parts[5]), 'amplitude': float(parts[6]),
        })

rmm = pd.DataFrame(rmm_rows)
rmm['time'] = pd.to_datetime(rmm[['year', 'month', 'day']])
rmm = rmm.set_index('time').drop(columns=['year', 'month', 'day'])
rmm.index = rmm.index.normalize()

print(f"RMM loaded: {rmm.index[0].date()} to {rmm.index[-1].date()}")

# load LLJ data (written by SALLJ.py)

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

# helper functions for compositing

def merge_with_rmm(llj_subset, rmm, amplitude_thresh):
    """
    Join one member's LLJ series with RMM index and filter active MJO days.
    llj_subset must have columns: time, llj_anom.
    Returns merged DataFrame with RMM columns added.
    """
    llj_indexed = llj_subset.set_index('time')[['llj_anom']]
    merged = llj_indexed.join(rmm, how='inner')
    merged = merged[merged['amplitude'] > amplitude_thresh].copy()
    return merged


def bootstrap_lag_cell(vals, pool, n_boot, rng_seed):
    """
    Compute mean, p-value, and 95% CI for one (phase, lag) cell.
    Designed to be called in parallel via joblib.
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


def run_lag_composites(merged_all_members, n_boot, max_lag, n_jobs):
    """
    Compute lag composites across all merged member data.

    merged_all_members : concatenated merged DataFrames from all members,
                         must have columns llj_anom, phase, amplitude
    Returns dict with keys 'means', 'pvals', 'ci_lo', 'ci_hi', 'ns',
    each a (8, max_lag+1) array.
    """
    lags_r   = range(0, max_lag + 1)
    llj_idx  = merged_all_members.index
    llj_vals = merged_all_members['llj_anom']
    pool     = llj_vals.values

    # pre-gather all values for each (phase, lag) cell
    cells = {}
    for phase in range(1, 9):
        phase_dates = merged_all_members[
            merged_all_members['phase'] == phase
        ].index.normalize()
        for lag in lags_r:
            lagged = phase_dates + pd.Timedelta(days=lag)
            valid  = lagged[lagged.isin(llj_idx)]
            cells[(phase, lag)] = llj_vals.loc[valid].values

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


# plotting functions

def plot_heatmap(means, pvals, max_lag, tag, title_suffix=''):
    fig, ax = plt.subplots(figsize=(16, 6))

    vmax = np.nanmax(np.abs(means))
    vmax = max(vmax, 0.5)

    im = ax.imshow(
        means, aspect='auto', origin='upper',
        extent=[-0.5, max_lag + 0.5, 8.5, 0.5],
        cmap='RdBu', vmin=-vmax, vmax=vmax, interpolation='nearest',
    )
    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.85)
    cbar.set_label("LLJ Anomaly (m/s)\nblue = stronger northerly jet", fontsize=9)

    # stippling: p < 0.05
    y05, x05 = np.where(pvals < 0.05)
    if len(x05):
        ax.scatter(x05, y05 + 1, marker='.', color='black',
                   s=18, zorder=5, label='p < 0.05')

    # dashed box: p < 0.10
    for (pi, li) in zip(*np.where((pvals < 0.10) & (pvals >= 0.05))):
        ax.add_patch(plt.Rectangle(
            (li - 0.5, pi + 0.5), 1, 1,
            lw=1.2, edgecolor='black', facecolor='none',
            ls='--', zorder=5
        ))

    ax.set_xticks(range(0, max_lag + 1))
    ax.set_xticklabels([f"+{l}d" for l in range(0, max_lag + 1)], fontsize=8)
    ax.set_yticks(range(1, 9))
    ax.set_yticklabels(
        [f"Phase {p}  ({phase_labels[p]})" for p in range(1, 9)], fontsize=9
    )
    ax.set_xlabel("Lag (days after MJO phase)", fontsize=10)
    ax.set_ylabel("MJO Phase", fontsize=10)
    ax.set_title(
        f"Lag Composite: SALLJ Anomaly Response to MJO Forcing\n"
        f"MRI-ESM2-0  |  Active MJO amp > 1.0  |  Season: {tag}{title_suffix}\n"
        "Dots = p < 0.05,  dashed box = p < 0.10",
        fontsize=10, fontweight='bold'
    )
    for l in range(0, max_lag + 1, 5):
        ax.axvline(l - 0.5, color='white', lw=0.5, alpha=0.4)

    dot_h = plt.Line2D([0], [0], marker='.', color='black',
                        ls='none', ms=6, label='p < 0.05')
    box_h = mpatches.Patch(fc='none', ec='black', ls='--', label='p < 0.10')
    ax.legend(handles=[dot_h, box_h], fontsize=9,
              loc='lower right', framealpha=0.85)

    plt.tight_layout()
    fpath = FIG_DIR / f'mjo_sallj_heatmap_{tag}.png'
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fpath}")


def plot_phase_barplot(comp_df, tag):
    """Lag-0 barplot for a given season/experiment tag."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(1, 9)
    colors = [
        'crimson'   if comp_df.loc[p, 'sig_05'] else
        'salmon'    if comp_df.loc[p, 'sig_10'] else
        'steelblue'
        for p in range(1, 9)
    ]
    ax.bar(x, comp_df['mean'], color=colors, alpha=0.85,
           edgecolor='white', width=0.65, zorder=3)
    ax.errorbar(
        x, comp_df['mean'],
        yerr=[comp_df['mean'] - comp_df['ci_lo'],
              comp_df['ci_hi'] - comp_df['mean']],
        fmt='none', color='black', capsize=5, lw=1.5, zorder=4
    )
    ax.axhline(0, color='black', lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"Phase {p}\n{phase_labels[p]}" for p in range(1, 9)], fontsize=8
    )
    ax.set_ylabel("LLJ Anomaly (m/s)")
    ax.set_title(
        f"Lag-0 SALLJ Composite by MJO Phase  |  Season: {tag}\n"
        "error bars = 95% bootstrap CI",
        fontweight='bold'
    )
    ax.grid(alpha=0.25, axis='y', zorder=0)
    sig_h   = mpatches.Patch(color='crimson', alpha=0.85, label='p < 0.05')
    mar_h   = mpatches.Patch(color='salmon',  alpha=0.85, label='p < 0.10')
    ins_h   = mpatches.Patch(color='steelblue', alpha=0.85, label='p ≥ 0.10')
    ax.legend(handles=[sig_h, mar_h, ins_h], fontsize=9)
    plt.tight_layout()
    fpath = FIG_DIR / f'mjo_sallj_barplot_{tag}.png'
    plt.savefig(fpath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fpath}")


# historical composite (all seasons + seasonal subsets)

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
        m_data = subset[subset['member'] == member][['time', 'llj_anom']]
        merged = merge_with_rmm(m_data, rmm, AMPLITUDE_THRESH)
        if len(merged) > 0:
            member_merged.append(merged)

    if not member_merged:
        print(f"  No active MJO days found for {season_name}, skipping.")
        continue

    merged_all = pd.concat(member_merged)
    n_total    = sum(len(m) for m in member_merged)
    print(f"  Total active MJO member-days: {n_total}")
    print(f"  Phase distribution:\n{merged_all['phase'].value_counts().sort_index()}")

    print(f"  Running bootstrap (N={N_BOOTSTRAP}, {N_JOBS} parallel workers)")
    results = run_lag_composites(merged_all, N_BOOTSTRAP, MAX_LAG, N_JOBS)

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

# ssp585 composite (all seasons, but only one member and limited RMM overlap)

print(f"\n{'='*60}")
print("  SSP585 composite (r1i1p1f1, all seasons)")
print(f"{'='*60}")

ssp_llj = llj_full[llj_full['experiment'] == 'ssp585'].copy()

# RMM index only covers the observational period and for ssp585 there is no
# observed RMM. Two options:
#   A. Use only ssp585 dates that overlap with RMM (2015 to end of RMM record)
#   B. Use model-derived MJO index (requires additional processing)
# We use option A here as a first pass. This gives ~5-10 years depending on
# when the RMM file was last updated. Interpret with caution.

ssp_merged_list = []
m_data     = ssp_llj[['time', 'llj_anom']]
ssp_merged = merge_with_rmm(m_data, rmm, AMPLITUDE_THRESH)
if len(ssp_merged) > 0:
    ssp_merged_list.append(ssp_merged)
    print(f"    SSP585 x RMM overlap: {len(ssp_merged)} active MJO days")
    print("     NOTE: RMM overlap limited to observed period; interpret with caution.")

    ssp_results = run_lag_composites(
        pd.concat(ssp_merged_list), N_BOOTSTRAP, MAX_LAG, N_JOBS
    )
    save_lag_table(
        ssp_results['means'], ssp_results['pvals'],
        ssp_results['ci_lo'], ssp_results['ci_hi'],
        ssp_results['ns'], MAX_LAG, 'ssp585_ALL'
    )
    plot_heatmap(
        ssp_results['means'], ssp_results['pvals'], MAX_LAG, 'ssp585_ALL',
        title_suffix='\nCAUTION: RMM overlap period only (~2015 onward)'
    )
else:
    print("  No SSP585 x RMM overlap found (RMM may not extend past 2015).")
    print("  SSP585 compositing requires a model-derived MJO index.")

# summary

print("\n\nSummary of output files:")
for f in sorted(FIG_DIR.iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f"  {str(f):<70s}  {size_kb:6.1f} kB")

print("\nCompositing complete.")