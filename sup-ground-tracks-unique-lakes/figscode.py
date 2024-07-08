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
from scipy.signal import find_peaks
import ee
import requests
from datetime import datetime 
from datetime import timedelta
from datetime import timezone
import rasterio as rio
from rasterio import plot as rioplot
from rasterio import warp
from shapely import wkt
from collections import defaultdict
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import matplotlib.patheffects as path_effects
from shapely.geometry import Polygon

import sys
sys.path.append('../utils/')
from lakeanalysis.utils import dictobj, convert_time_to_string, read_melt_lake_h5

#####################################################################
class HandlerLinesVertical(HandlerTuple):
    def create_artists(self, legend, orig_handle,
                   xdescent, ydescent, width, height, fontsize,
                   trans):
        ndivide = len(orig_handle)
        ydescent = height/float(ndivide+1)
        a_list = []
        for i, handle in enumerate(orig_handle):
            # y = -height/2 + (height / float(ndivide)) * i -ydescent
            y = -height + 2*i*ydescent
            line = plt.Line2D(np.array([0,1])*width, [-y,-y])
            line.update_from(handle)
            # line.set_marker(None)
            point = plt.Line2D(np.array([.5])*width, [-y])
            point.update_from(handle)
            for artist in [line, point]:
                artist.set_transform(trans)
            a_list.extend([line,point])
        return a_list


#####################################################################
def add_graticule(gdf, ax_img, meridians_locs=['bottom','right'], parallels_locs=['top','left'], fontsz=8):
    from lakeanalysis.curve_intersect import intersection
    xl = ax_img.get_xlim()
    yl = ax_img.get_ylim()
    minx = xl[0]
    miny = yl[0]
    maxx = xl[1]
    maxy = yl[1]
    bounds = [minx, miny, maxx, maxy]
    latlon_bbox = warp.transform(gdf.crs, {'init': 'epsg:4326'}, 
                                 [bounds[i] for i in [0,2,2,0,0]], 
                                 [bounds[i] for i in [1,1,3,3,1]])
    min_lat = np.min(latlon_bbox[1])
    max_lat = np.max(latlon_bbox[1])
    min_lon = np.min(latlon_bbox[0])
    max_lon = np.max(latlon_bbox[0])
    latdiff = max_lat-min_lat
    londiff = max_lon-min_lon
    diffs = np.array([0.0001, 0.0002, 0.00025, 0.0004, 0.0005,
                      0.001, 0.002, 0.0025, 0.004, 0.005, 
                      0.01, 0.02, 0.025, 0.04, 0.05, 0.1, 0.2, 0.25, 0.4, 0.5, 1, 2])
    latstep = np.min(diffs[diffs>latdiff/8])
    lonstep = np.min(diffs[diffs>londiff/8])
    minlat = np.floor(min_lat/latstep)*latstep
    maxlat = np.ceil(max_lat/latstep)*latstep
    minlon = np.floor(min_lon/lonstep)*lonstep
    maxlon = np.ceil(max_lon/lonstep)*lonstep

    # plot meridians and parallels
    xl = ax_img.get_xlim()
    yl = ax_img.get_ylim()
    meridians = np.arange(minlon,maxlon, step=lonstep)
    parallels = np.arange(minlat,maxlat, step=latstep)
    latseq = np.linspace(minlat,maxlat,200)
    lonseq = np.linspace(minlon,maxlon,200)
    gridcol = 'k'
    gridls = ':'
    gridlw = 0.5
    topline = [[xl[0],xl[1]],[yl[1],yl[1]]]
    bottomline = [[xl[0],xl[1]],[yl[0],yl[0]]]
    leftline = [[xl[0],xl[0]],[yl[0],yl[1]]]
    rightline = [[xl[1],xl[1]],[yl[0],yl[1]]]
    addleft = r'$\rightarrow$'
    addright = r'$\leftarrow$'
    for me in meridians:
        gr_trans = warp.transform({'init': 'epsg:4326'},gdf.crs,me*np.ones_like(latseq),latseq)
        deglab = '%.10g°E' % me if me >= 0 else '%.10g°W' % -me
        rot = np.arctan2(gr_trans[1][-1] - gr_trans[1][0], gr_trans[0][-1] - gr_trans[0][0]) * 180 / np.pi
        if 'bottom' in meridians_locs:
            ha = 'right' if rot>0 else 'left'
            deglab_ = deglab+addleft if rot>0 else addright+deglab
            intx,inty = intersection(bottomline[0], bottomline[1], gr_trans[0], gr_trans[1])
            if len(intx) > 0:
                intx, inty = intx[0], inty[0]
                ax_img.text(intx, inty, deglab_, fontsize=fontsz, color='gray',verticalalignment='center',horizontalalignment=ha,
                        rotation=rot, rotation_mode='anchor')
        if 'top' in meridians_locs:
            ha = 'left' if rot>0 else 'right'
            deglab_ = addright+deglab if rot>0 else deglab+addleft
            intx,inty = intersection(topline[0], topline[1], gr_trans[0], gr_trans[1])
            if len(intx) > 0:
                intx, inty = intx[0], inty[0]
                ax_img.text(intx, inty, deglab_, fontsize=fontsz, color='gray',verticalalignment='center',horizontalalignment=ha,
                        rotation=rot, rotation_mode='anchor')
        if 'right' in meridians_locs:
            intx,inty = intersection(rightline[0], rightline[1], gr_trans[0], gr_trans[1])
            if len(intx) > 0:
                intx, inty = intx[0], inty[0]
                ax_img.text(intx, inty, addright+deglab, fontsize=fontsz, color='gray',verticalalignment='center',horizontalalignment='left',
                        rotation=rot, rotation_mode='anchor')
        if 'left' in meridians_locs:
            intx,inty = intersection(leftline[0], leftline[1], gr_trans[0], gr_trans[1])
            if len(intx) > 0:
                intx, inty = intx[0], inty[0]
                ax_img.text(intx, inty, deglab+addleft, fontsize=fontsz, color='gray',verticalalignment='center',horizontalalignment='right',
                        rotation=rot, rotation_mode='anchor')
        
        thislw = gridlw
        ax_img.plot(gr_trans[0],gr_trans[1],c=gridcol,ls=gridls,lw=thislw,alpha=0.5)
    for pa in parallels:
        gr_trans = warp.transform({'init': 'epsg:4326'},gdf.crs,lonseq,pa*np.ones_like(lonseq))
        thislw = gridlw
        deglab = '%.10g°N' % pa if pa >= 0 else '%.10g°S' % -pa
        rot = np.arctan2(gr_trans[1][-1] - gr_trans[1][0], gr_trans[0][-1] - gr_trans[0][0]) * 180 / np.pi
        if 'left' in parallels_locs:
            intx,inty = intersection(leftline[0], leftline[1], gr_trans[0], gr_trans[1])
            if len(intx) > 0:
                intx, inty = intx[0], inty[0]
                ax_img.text(intx, inty, deglab+addleft, fontsize=fontsz, color='gray',verticalalignment='center',horizontalalignment='right',
                           rotation=rot, rotation_mode='anchor')
        if 'right' in parallels_locs:
            intx,inty = intersection(rightline[0], rightline[1], gr_trans[0], gr_trans[1])
            if len(intx) > 0:
                intx, inty = intx[0], inty[0]
                ax_img.text(intx, inty, addright+deglab, fontsize=fontsz, color='gray',verticalalignment='center',horizontalalignment='left',
                           rotation=rot, rotation_mode='anchor')
        if 'top' in parallels_locs:
            ha = 'left' if rot>0 else 'right'
            deglab_ = addright+deglab if rot>0 else deglab+addleft
            intx,inty = intersection(topline[0], topline[1], gr_trans[0], gr_trans[1])
            if len(intx) > 0:
                intx, inty = intx[0], inty[0]
                ax_img.text(intx, inty, deglab_, fontsize=fontsz, color='gray',verticalalignment='center',horizontalalignment=ha,
                           rotation=rot, rotation_mode='anchor')
        if 'bottom' in parallels_locs:
            ha = 'right' if rot>0 else 'left'
            deglab_ = deglab+addleft if rot>0 else addright+deglab
            intx,inty = intersection(bottomline[0], bottomline[1], gr_trans[0], gr_trans[1])
            if len(intx) > 0:
                intx, inty = intx[0], inty[0]
                ax_img.text(intx, inty, deglab_, fontsize=fontsz, color='gray',verticalalignment='center',horizontalalignment=ha,
                           rotation=rot, rotation_mode='anchor')
        
        ax_img.plot(gr_trans[0],gr_trans[1],c=gridcol,ls=gridls,lw=thislw,alpha=0.5)
        ax_img.set_xlim(xl)
        ax_img.set_ylim(yl)


#####################################################################
def find_unique_lakes(gti):

    # Union-Find (Disjoint-Set) data structure implementation
    class UnionFind:
        def __init__(self):
            self.parent = {}
            self.rank = {}
    
        def find(self, x):
            if self.parent[x] != x:
                self.parent[x] = self.find(self.parent[x])
            return self.parent[x]
    
        def union(self, x, y):
            rootX = self.find(x)
            rootY = self.find(y)
    
            if rootX != rootY:
                if self.rank[rootX] > self.rank[rootY]:
                    self.parent[rootY] = rootX
                elif self.rank[rootX] < self.rank[rootY]:
                    self.parent[rootX] = rootY
                else:
                    self.parent[rootY] = rootX
                    self.rank[rootX] += 1
    
        def add(self, x):
            if x not in self.parent:
                self.parent[x] = x
                self.rank[x] = 0
    
    # Initialize union-find
    uf = UnionFind()
    
    # Step 1: Create a mapping from number to list of indices
    num_to_indices = defaultdict(set)
    for idx, numbers in gti.items():
        for number in numbers:
            num_to_indices[number].add(idx)
    
    # Step 2: Union indices that share the same numbers
    for indices in num_to_indices.values():
        indices = list(indices)
        first = indices[0]
        uf.add(first)
        for idx in indices[1:]:
            uf.add(idx)
            uf.union(first, idx)
    
    # Step 3: Collect the groups
    groups = defaultdict(list)
    for idx in gti.index:
        uf.add(idx)  # Ensure all indices, including those with empty lists, are added
        root = uf.find(idx)
        groups[root].append(idx)
    
    # Step 4: Create idx_list for each group
    grouped_idx_lists = []
    for group_indices in groups.values():
        idx_list = set()
        for idx in group_indices:
            idx_list.update(gti[idx])
        grouped_idx_lists.append(list(idx_list))
    
    # Output the groups and their idx_lists
    return pd.DataFrame({'IDs_is2': list(groups.values()), 'IDs_extent': grouped_idx_lists}).reset_index(names='ID_unique_lake')
