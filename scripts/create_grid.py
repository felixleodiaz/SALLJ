# import libraries

import xarray as xr
import numpy as np

# create grid w regional bounds here (south american LLJ)

xmin, xmax, ymin, ymax = -105+360, -90+360, 25, 50
latitudes = np.arange(ymin, ymax + 1, 0.5)
longitudes = np.arange(xmin, xmax + 1, 0.5)

# create empty dataset with specified dimensions

ds = xr.Dataset(
    {
        "foo": (["y", "x"], np.empty((len(latitudes), len(longitudes))))
    },
    coords={
        "y": latitudes,
        "x": longitudes
    }
)

ds.to_netcdf('../data/grid.nc')