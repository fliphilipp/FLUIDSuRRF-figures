import os
# os.environ["GDAL_DATA"] = "/home/parndt/anaconda3/envs/geo_py37/share/gdal"
# os.environ["PROJ_LIB"] = "/home/parndt/anaconda3/envs/geo_py37/share/proj"
import h5py
import math
import zipfile
import traceback
import shapely
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle
from cmcrameri import cm as cmc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import Image, display
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from sklearn.neighbors import KDTree
from scipy.stats import binned_statistic
from scipy.stats import pearsonr
from scipy.signal import find_peaks
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

import sys
sys.path.append('../utils/')
from lakeanalysis.utils import dictobj, convert_time_to_string, read_melt_lake_h5

def getstats(dfsel, verb=False):
    dfsel = dfsel.reset_index(drop=True)
    dfsel.loc[dfsel.conf<0.5, 'depth'] = np.nan
    diffs = (dfsel.manual/1.336 - dfsel.depth) # divide by refractive index for depth
    diffs[(dfsel.manual==0) | (dfsel.depth==0)] = np.nan
    diffs = diffs[~np.isnan(diffs)]
    bias = np.mean(diffs)
    std = np.std(diffs)
    mae = np.mean(np.abs(diffs))
    rmse = np.sqrt(np.mean(diffs**2))
    sel = (~np.isnan(dfsel.depth)) & (~np.isnan(dfsel.manual))
    correl = pearsonr(dfsel.manual[sel], dfsel.depth[sel]/1.336).statistic
    if verb:
        print('- mean diff:', bias)
        print('- std diff:', std)
        print('- MAE:', mae)
        print('- RMSE:', rmse)
        print('- correl:', correl)
    return bias, std, mae, rmse, correl

# depth_files_melling = [
#     'data/validation/melling/257.csv',
#     'data/validation/melling/963.csv',
#     'data/validation/melling/1116.csv',
#     'data/validation/melling/4788.csv',
#     'data/validation/melling/5289.csv'
# ]
# gtxs_melling = ['gt2l', 'gt2l', 'gt1l', 'gt2l', 'gt1l']
# dates_melling = [
#     '2020-07-06',
#     '2020-07-17',
#     '2020-07-06',
#     '2020-07-06',
#     '2019-07-16'
# ]
# granules_melling = [
#     'ATL03_20200706005932_01630805_006_01.h5',
#     'ATL03_20200717114945_03380803_006_01.h5',
#     'ATL03_20200706005932_01630805_006_01.h5',
#     'ATL03_20200706005932_01630805_006_01.h5',
#     'ATL03_20190716051841_02770403_006_02.h5'
# ]
# renames = {
#     'Latitude': 'lat',
#     'Longitude': 'lon',
#     'Distance along the transect (m)': 'xatc',
#     'Depth': 'manual'
# }
# dfs = []
# for i,fn in enumerate(depth_files_melling):
#     thisdf = pd.read_csv(fn).rename(columns=renames)
#     thisdf['pond'] = 5+i
#     thisdf['granule_id'] = granules_melling[i]
#     thisdf['gtx'] = gtxs_melling[i]
#     thisdf.manual *= 1.333 # refractive index correction
#     dfs.append(thisdf)
# df_melling = pd.concat(dfs)
# df_melling['source'] = 'Melling_Cryosphere_2024'

# df_fricker = pd.read_csv('data/validation/2021_paper_depths.csv').rename(columns={'Unnamed: 0': 'lat'})
# df_fricker['source'] = 'Fricker_GRL_2021'
# df_fricker['granule_id'] = 'ATL03_20190102184312_00810210_006_02.h5'
# df_fricker['gtx'] = 'gt2l'

# vars_select = ['pond', 'lat', 'manual', 'granule_id', 'gtx', 'source']

# dfm = pd.concat((df_fricker[vars_select], df_melling[vars_select]))
# dfm.to_csv('data/validation/manual_estimates.csv', index=False)

# for i, ilake in enumerate(np.unique(df_melling.pond)):
#     dfi = df_melling[df_melling.pond==ilake]
#     latmin = dfi.lat.min()
#     latmax = dfi.lat.max()
#     latrange = latmax-latmin
#     lonmin = dfi.lon.min()
#     lonmax = dfi.lon.max()
#     lonrange = lonmax-lonmin
#     fac = 2
#     xmin = lonmin - fac*lonrange
#     xmax = lonmax + fac*lonrange
#     ymin = latmin - fac*latrange
#     ymax = latmax + fac*latrange
#     poly = orient(Polygon([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)]), sign=1.0)
#     out_fn = 'geojsons/melling2024_lake%d.geojson' % (i+1)
#     gpd.GeoSeries(poly).set_crs('EPSG:4326').to_file(out_fn, driver='GeoJSON')
#     print(out_fn, xmin, xmax, ymin, ymax)