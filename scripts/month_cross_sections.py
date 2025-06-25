# import libraries

import seaborn as sns
import xarray as xr
import numpy as np
import pandas as pd
import intake
import dask
import regionmask
import warnings
from pathlib import Path
from tqdm import tqdm
from xmip.preprocessing import combined_preprocessing, replace_x_y_nominal_lat_lon
import matplotlib.pyplot as plt

# link to data

url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
col = intake.open_esm_datastore(url)

# load some monthly data from the GFDL-CM4 4-K warming experiment

query = dict(experiment_id='amip-p4K',
             variable_id=['ua', 'va'],
             source_id=['GFDL-CM4'],
             table_id='Amon'
            )

cat = col.search(**query)
print(cat.df['source_id'].unique())

# load data into dictionary with keys constructed as 'activity_id.institution_id.source_id.experiment_id.table_id.grid_label'

z_kwargs = {'consolidated': True, 'decode_times':True}

with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    dset_dict = cat.to_dataset_dict(zarr_kwargs=z_kwargs, preprocess=combined_preprocessing)


# get GFDL-AM4 4K warming experiment

ds = dset_dict['CFMIP.NOAA-GFDL.GFDL-CM4.amip-p4K.Amon.gr1']

# calculate month datasets

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

monthly_data = {}
for i, mon in enumerate(months, start=1):
    data = ds.sel(time=ds.time.dt.month == i)
    monthly_data[mon] = data.mean(dim = 'time')

# plot

titles = [f'{mon} Meridional Wind' for mon in months]

all_data = ds.va.sel(x=slice(280, 320), y=-20.5)
global_min = all_data.min().item() # compute global min and max
global_max = all_data.max().item()

fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8), sharey=True)
axes = axes.flatten()

im = None
for i, (ax, mon) in enumerate(zip(axes, months)):
    data = all_data[i].assign_coords(plev=ds['plev'] / 100)
    im = data.plot(ax=ax, vmin=global_min, vmax=global_max, add_colorbar=False)
    ax.set_ylim(1000, 400)
    ax.set_title(titles[i])
    ax.set_xlabel('Longitude (°E)')
    if i % 4 == 0:
        ax.set_ylabel('Pressure (hPa)')
    else:
        ax.set_ylabel('')

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax, label='Meridional Wind (units)')  # Adjust label as needed

fig.suptitle(
    'South American Low-Level Jet Wind Profiles\nMonthly Averages 1979–2012 with +4K Forcing',
    fontsize=16,
    y=1.03
)

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar and title
plt.savefig('../figures/MonthAverages.png', dpi=300)
plt.show()