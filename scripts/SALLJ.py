# libraries

import seaborn as sns
import xarray as xr
import numpy as np
import pandas as pd
import intake
import dask
import warnings
from tqdm import tqdm
from xmip.preprocessing import combined_preprocessing
import matplotlib.pyplot as plt

# data

warnings.filterwarnings("ignore")
url = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
col = intake.open_esm_datastore(url)

# set up the local filecache path for faster reading (~5 min w 50 cores and 50 GB memory)

cache_path = '/scratch/fld1/cmip_cache'

storage_options = {
    'filecache': {
        'cache_storage': cache_path,
        'target_protocol': 'gs',
    }
}

# load 6-hourly data with meridional and zonal wind, air temperature, surface pressure, and specific humidity

query = dict(source_id='MRI-ESM2-0',
             table_id='6hrLev',
             experiment_id=['historical', 'ssp585'],
             variable_id=['va', 'ua', 'ta']
)

cat = col.search(**query)

# load SA data into dictionary. keys: 'activity_id.institution_id.source_id.experiment_id.table_id.grid_label'

warnings.filterwarnings("ignore")

z_kwargs = {'consolidated': True, 'decode_times':True}

with dask.config.set(**{'array.slicing.split_large_chunks': True}):
    dset_dict = cat.to_dataset_dict(
        zarr_kwargs=z_kwargs,
        storage_options=storage_options,
        preprocess=lambda ds: ds.sel(lon=slice(260, 335), lat=slice(-60, 10))
)

# pull SSP data from dictionary

ds = dset_dict['ScenarioMIP.MRI.MRI-ESM2-0.ssp585.6hrLev.gn']

# function to find LLJs

def find_jets(x):

    '''input 4D array output 1D array of dicts of LLJ information'''

    # step one: core speed
     
    # code here
    
    # step two: checks

    # code here

    # step three: create dict

    # code here

    # step four: apply to xarray

    # code here

# assign total wind vector field and find LLJs

ds = (ds
    .assign(pWind=lambda x: np.sqrt(x['ua']**2 + x['va']**2))
    .assign() #this should then call the function for finding LLJs
    .squeeze()
    .compute())
