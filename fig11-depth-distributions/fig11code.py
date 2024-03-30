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
from matplotlib.patches import Patch
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
from sklearn.neighbors import KDTree
from scipy.stats import binned_statistic
from scipy.signal import find_peaks
from scipy.stats import linregress

import sys
sys.path.append('../utils/')
from lakeanalysis.utils import dictobj, convert_time_to_string, read_melt_lake_h5

def update_lake_stats(minconf=0.3, mindepth=0.5):
    out_path_csv = '../data/lakestats_methods_paper.csv'
    df = pd.read_csv(out_path_csv)
    df_lakes = df.copy()
    maxdepths = []
    for i in range(len(df_lakes)):
        print('reading file %5i / %5i' % (i+1, len(df_lakes)), end='\r')
        fn = df_lakes.iloc[i].file_name
        lk = dictobj(read_melt_lake_h5(fn))
        dfd = lk.depth_data.copy()
        
        isdepth = dfd.depth>0
        bed = dfd.h_fit_bed
        bed[~isdepth] = np.nan
        bed[dfd.conf < minconf] = np.nan
        bed[dfd.depth < mindepth] = np.nan
        surf = np.ones_like(dfd.xatc) * lk.surface_elevation
        surf[~isdepth] = np.nan
        bed_only = bed[(~np.isnan(surf)) & (~np.isnan(bed))]
    
        if len(bed_only) == 0:
            maxdepths.append(np.nan)
        else:
            y_low = np.percentile(bed_only, 5)
            y_up = lk.surface_elevation
            ref_index = 1.336
            max_depth = (y_up - y_low) / ref_index
            maxdepths.append(max_depth)

    df_lakes['depth_95th_pctl'] = maxdepths
    df_lakes.to_csv('data/lakestats_methods_paper_depth_update.csv', index=False)
    return df_lakes

def style_legend_titles_by_removing_handles(leg: matplotlib.legend.Legend) -> None:
    for col in leg._legend_handle_box.get_children():
        row = col.get_children()
        new_children: list[plt.Artist] = []
        for hpacker in row:
            if not isinstance(hpacker, matplotlib.offsetbox.HPacker):
                new_children.append(hpacker)
                continue
            drawing_area, text_area = hpacker.get_children()
            handle_artists = drawing_area.get_children()
            if not all(a.get_visible() for a in handle_artists):
                new_children.append(text_area)
            else:
                new_children.append(hpacker)
        col._children = new_children

def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])