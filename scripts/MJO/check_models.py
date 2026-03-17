# =============================================================================
# Catalog Diagnostic — find what is actually available for MRI-ESM2-0
# Run this before the main detection script to pick the right table_id.
# =============================================================================

import warnings
import pandas as pd
import intake

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    col = intake.open_esm_datastore(
        "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
    )

# =============================================================================
# STEP 1: Check what table_ids exist for MRI-ESM2-0 at all
# =============================================================================

all_mri = col.search(source_id='MRI-ESM2-0')
available_tables = all_mri.df['table_id'].unique()
print("All table_ids available for MRI-ESM2-0:")
print(sorted(available_tables))

# =============================================================================
# STEP 2: Check which of those tables have meridional wind (va)
# =============================================================================

va_mri = col.search(source_id='MRI-ESM2-0', variable_id='va')
print("\nTables containing 'va' for MRI-ESM2-0:")
print(
    va_mri.df[['table_id', 'experiment_id', 'member_id']]
    .drop_duplicates()
    .sort_values(['table_id', 'experiment_id'])
    .to_string(index=False)
)

# =============================================================================
# STEP 3: Check the preferred table hierarchy for this project
# =============================================================================
# Best → worst for LLJ / MJO work:
#
#   6hrPlev    6-hourly, pressure levels  — ideal; often missing
#   6hrPlevPt  6-hourly, pressure levels, instantaneous (point-in-time)
#              — nearly as good; common alternative to 6hrPlev
#   day        daily,    pressure levels  — sufficient for LLJ climatology
#              and Wang & Fu replication; MJO phase composites still work
#   Amon       monthly                   — too coarse for MJO work

preferred = ['6hrPlev', '6hrPlevPt', 'day', 'Amon']

best_table = None
for t in preferred:
    check = col.search(
        source_id='MRI-ESM2-0',
        variable_id='va',
        table_id=t,
        experiment_id=['historical', 'ssp585'],
    )
    n = len(check.df)
    print(f"\n  {t:12s}  →  {n} catalog entries found", end="")
    if n > 0 and best_table is None:
        best_table = t
        print("  ✓  USE THIS", end="")
    print()

print(f"\n>>> Recommended table_id for your query: '{best_table}'")

# =============================================================================
# STEP 4: Confirm member_ids available for the chosen table
# =============================================================================
if best_table:
    best_check = col.search(
        source_id='MRI-ESM2-0',
        variable_id='va',
        table_id=best_table,
        experiment_id=['historical', 'ssp585'],
    )
    print(f"\nAvailable member_ids in '{best_table}':")
    print(
        best_check.df[['experiment_id', 'member_id']]
        .drop_duplicates()
        .sort_values(['experiment_id', 'member_id'])
        .to_string(index=False)
    )

# =============================================================================
# STEP 5: Also check ua (zonal wind) — needed for cross-Andes flow analysis
# =============================================================================
ua_check = col.search(
    source_id='MRI-ESM2-0',
    variable_id='ua',
    table_id=best_table,
    experiment_id=['historical', 'ssp585'],
) if best_table else None

print(f"\n'ua' available in '{best_table}': "
      f"{'Yes' if ua_check and len(ua_check.df) > 0 else 'No'}")