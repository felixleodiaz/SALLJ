# import libraries

import seaborn as sns
import xarray as xr
import numpy as np
import pandas as pd
import intake
import dask
from xmip.preprocessing import combined_preprocessing
import matplotlib.pyplot as plt

# link to data

url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
col = intake.open_esm_datastore(url)

# load some monthly data from the GFDL-CM4 4-K warming experiment

query = dict(experiment_id =['amip', 'amip-p4K', 'amip-m4K'],
             variable_id=['ua', 'va'],
             source_id=['GFDL-CM4'],
             table_id='Amon'
)

cat = col.search(**query)

# load data into dictionary with keys constructed as 'activity_id.institution_id.source_id.experiment_id.table_id.grid_label'

z_kwargs = {'consolidated': True, 'decode_times':True}

with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    dset_dict = cat.to_dataset_dict(zarr_kwargs=z_kwargs, preprocess=combined_preprocessing)


# get GFDL-AM4 datasets

dsh = dset_dict['CMIP.NOAA-GFDL.GFDL-CM4.amip.Amon.gr1']
dsw = dset_dict['CFMIP.NOAA-GFDL.GFDL-CM4.amip-p4K.Amon.gr1']
dsc = dset_dict['CFMIP.NOAA-GFDL.GFDL-CM4.amip-m4K.Amon.gr1']


# calculate month datasets and anomolies

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

month_hist = {}
month_warm = {}
month_cool = {}
warm_anom = {}
cool_anom = {}

for i, mon in enumerate(months, start=1):
    data_hist = dsh.sel(time=dsh.time.dt.month == i)
    month_hist[mon] = data_hist.mean(dim = 'time')

    data_warm = dsw.sel(time=dsw.time.dt.month == i)
    month_warm[mon] = data_warm.mean(dim = 'time')

    data_cool = dsc.sel(time=dsc.time.dt.month == i)
    month_cool[mon] = data_cool.mean(dim = 'time')

    warm_anom[mon] = month_warm[mon] - month_hist[mon]
    cool_anom[mon] = month_cool[mon] - month_hist[mon]


# plot historical

all_data = dsh.va.sel(x=slice(280, 320), y=-20.5)
global_min = all_data.min() # compute global min and max
global_max = all_data.max()

sns.set_style('white')
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8), sharey=True, sharex=True)
axes = axes.flatten()

for i, (ax, mon) in enumerate(zip(axes, months)):
    im = month_hist[mon].va.sel(x=slice(280, 320), y=-20.5).assign_coords(plev=dsh['plev'] / 100).plot(ax=ax, add_colorbar=False)
    ax.set_ylim(1000, 400)
    ax.set_title(f'{mon} Meridional Wind')
    xlab = 'Longitude (°E)' if i // 4 ==2 else ''
    ylab = 'Pressure (hPa)' if i % 4 == 0 else ''
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax, label='Average Meridional Wind Speed (m/s)')  # Adjust label as needed

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar and title
plt.savefig('../figures/MonthAverageCrossSectionsHistorical.png', dpi=300)
plt.show()

# plot warming

all_data = dsw.va.sel(x=slice(280, 320), y=-20.5)
global_min = all_data.min() # compute global min and max
global_max = all_data.max()

sns.set_style('white')
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8), sharey=True, sharex=True)
axes = axes.flatten()

for i, (ax, mon) in enumerate(zip(axes, months)):
    im = month_warm[mon].va.sel(x=slice(280, 320), y=-20.5).assign_coords(plev=dsw['plev'] / 100).plot(ax=ax, add_colorbar=False)
    ax.set_ylim(1000, 400)
    ax.set_title(f'{mon} Meridional Wind')
    xlab = 'Longitude (°E)' if i // 4 ==2 else ''
    ylab = 'Pressure (hPa)' if i % 4 == 0 else ''
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax, label='Average Meridional Wind Speed (m/s)')  # Adjust label as needed

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar and title
plt.savefig('../figures/MonthAverageCrossSectionsUnderWarming.png', dpi=300)
plt.show()

# plot cooling

all_data = dsc.va.sel(x=slice(280, 320), y=-20.5)
global_min = all_data.min() # compute global min and max
global_max = all_data.max()

sns.set_style('white')
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8), sharey=True, sharex=True)
axes = axes.flatten()

for i, (ax, mon) in enumerate(zip(axes, months)):
    im = month_cool[mon].va.sel(x=slice(280, 320), y=-20.5).assign_coords(plev=dsc['plev'] / 100).plot(ax=ax, add_colorbar=False)
    ax.set_ylim(1000, 400)
    ax.set_title(f'{mon} Meridional Wind')
    xlab = 'Longitude (°E)' if i // 4 ==2 else ''
    ylab = 'Pressure (hPa)' if i % 4 == 0 else ''
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax, label='Average Meridional Wind Speed (m/s)')  # Adjust label as needed

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar and title
plt.savefig('../figures/MonthAverageCrossSectionsUnderCooling.png', dpi=300)
plt.show()

# plot warming anomoly

global_min = -4  # set global min and max
global_max = 4

sns.set_style('white')
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8), sharey=True, sharex=True)
axes = axes.flatten()

for i, (ax, mon) in enumerate(zip(axes, months)):
    im = warm_anom[mon].va.sel(x=slice(280, 320), y=-20.5).assign_coords(plev=dsh['plev'] / 100).plot(ax=ax, add_colorbar=False)
    ax.set_ylim(1000, 400)
    ax.set_title(f'{mon} Anomoly')
    xlab = 'Longitude (°E)' if i // 4 ==2 else ''
    ylab = 'Pressure (hPa)' if i % 4 == 0 else ''
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax, label='Average Meridional Wind Speed (m/s)')  # Adjust label as needed

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar and title
plt.savefig('../figures/MonthAverageCrossSections.png', dpi=300)
plt.show()

# plot cooling anomoly

global_min = -4 # set global min and max
global_max = 4

sns.set_style('white')
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(12, 8), sharey=True, sharex=True)
axes = axes.flatten()

for i, (ax, mon) in enumerate(zip(axes, months)):
    im = cool_anom[mon].va.sel(x=slice(280, 320), y=-20.5).assign_coords(plev=dsh['plev'] / 100).plot(ax=ax, add_colorbar=False)
    ax.set_ylim(1000, 400)
    ax.set_title(f'{mon} Anomonly')
    xlab = 'Longitude (°E)' if i // 4 ==2 else ''
    ylab = 'Pressure (hPa)' if i % 4 == 0 else ''
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)

cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
fig.colorbar(im, cax=cbar_ax, label='Average Meridional Wind Speed (m/s)')  # Adjust label as needed

plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space for colorbar and title
plt.savefig('../figures/MonthAverageCrossSections.png', dpi=300)
plt.show()
