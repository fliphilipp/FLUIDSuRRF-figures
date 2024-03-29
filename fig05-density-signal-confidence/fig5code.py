import os
import h5py
import pickle
import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import datetime
import time
import traceback
from IPython.display import Image, display
from cmcrameri import cm as cmc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from sklearn.neighbors import KDTree

import sys
sys.path.append('../utils/')
from lakeanalysis.utils import dictobj, convert_time_to_string, read_melt_lake_h5
from lakeanalysis.nsidc import download_is2, read_atl03

# not needed here but for reference, can play around with parameters
def get_density(file, segment_length=140, signal_width=0.35, aspect=30, K_phot=10, h_signal=None, frac_noise=0.05):
    lk = dictobj(read_melt_lake_h5(file))
    df = lk.photon_data.sort_values(by='xatc')
    df.is_afterpulse = df.prob_afterpulse > np.random.uniform(0,1,len(df))
    df = df[~df.is_afterpulse].copy()
    n_segs = int(np.ceil((df.xatc.max()-df.xatc.min())/segment_length))
    len_seg = (df.xatc.max()-df.xatc.min())/n_segs
    edges = np.arange(df.xatc.min(), df.xatc.max()+len_seg/2, len_seg)
    df['snr'] = 0.0
    for i in range(len(edges)-1):
        selector_segment = (df.xatc>=edges[i]) & (df.xatc<edges[i+1])
        dfseg = df[selector_segment].copy()
        xmin = edges[i]
        xmax = edges[i+1]
        if h_signal:
            dfseg_nosurface = dfseg[np.abs(dfseg.h-h_signal) > signal_width]
        else:
            dfseg_nosurface = dfseg[np.abs(dfseg.h-dfseg.rmean) > signal_width]
        nphot_bckgrd = len(dfseg_nosurface.h)
    
        # radius of a circle in which we expect to find one noise photon
        telem_h = dfseg_nosurface.h.max()-dfseg_nosurface.h.min()
        h_noise = telem_h-signal_width*2
        wid_noise = (xmax-xmin)/aspect
        area = h_noise*wid_noise/nphot_bckgrd
        fac=3
        wid = np.sqrt(fac*frac_noise*(K_phot+1)*area/np.pi)
    
        # buffer segment for density calculation
        selector_buffer = (df.xatc >= (dfseg.xatc.min()-aspect*wid)) & (df.xatc <= (dfseg.xatc.max()+aspect*wid))
        dfseg_buffer = df[selector_buffer].copy()
    
        # normalize xatc to be regularly spaced and scaled by the aspect parameter
        xmin_buff = dfseg_buffer.xatc.min()
        xmax_buff = dfseg_buffer.xatc.max()
        nphot_buff = len(dfseg_buffer.xatc)
        xnorm = np.linspace(xmin_buff, xmax_buff, nphot_buff) / aspect
    
        # KD tree query distances
        Xn = np.array(np.transpose(np.vstack((xnorm, dfseg_buffer['h']))))
        kdt = KDTree(Xn)
        idx, dist = kdt.query_radius(Xn, r=wid, count_only=False, return_distance=True,sort_results=True)
        density = (np.array([np.sum(1-np.abs(x/wid)) if (len(x)<(K_phot+1)) 
                   else np.sum(1-np.abs(x[:K_phot+1]/wid))
                   for x in dist]) - 1) / K_phot
    
        # get densities only for segment
        selector_segment_only = (np.array(dfseg_buffer.xatc)>=edges[i]) & (np.array(dfseg_buffer.xatc)<edges[i+1])
        densities = np.array(density[selector_segment_only])
        df.loc[selector_segment, 'snr'] = densities
        
    df = df.sort_values(by='snr').reset_index(drop=True)

    try:
        with h5py.File(file, 'r+') as f:
            if 'photon_data' in f.keys():
                del f['photon_data']
            comp="gzip"
            photdat = f.create_group('photon_data')
            cols = list(df.keys())
            for col in cols:
                photdat.create_dataset(col, data=df[col], compression=comp)
            
    except:
        print('WARNING: Densities could not be written to file!')
        traceback.print_exc()
        
    return df

def print_lake_info(fn, description='', print_imagery_info=True):
    lk = dictobj(read_melt_lake_h5(fn))
    keys = vars(lk).keys()
    print('\nLAKE INFO: %s' % description)
    print('  granule_id:            %s' % lk.granule_id)
    print('  RGT:                   %s' % lk.rgt)
    print('  GTX:                   %s' % lk.gtx.upper())
    print('  beam:                  %s (%s)' % (lk.beam_number, lk.beam_strength))
    print('  acquisition time:      %s' % lk.time_utc)
    print('  center location:       (%s, %s)' % (lk.lon_str, lk.lat_str))
    print('  ice sheet:             %s' % lk.ice_sheet)
    print('  melt season:           %s' % lk.melt_season)
    
    if ('imagery_info' in keys) and (print_imagery_info):
        print('  IMAGERY INFO:')
        print('    product ID:                     %s' % lk.imagery_info['product_id'])
        print('    acquisition time imagery:       %s' % lk.imagery_info['time_imagery'])
        print('    acquisition time ICESat-2:      %s' % lk.imagery_info['time_icesat2'])
        print('    time difference from ICESat-2:  %s (%s)' % (lk.imagery_info['time_diff_from_icesat2'],lk.imagery_info['time_diff_string']))
    print('')

def download_full_atl03_for_lake(lk):
    df = lk.photon_data
    latmin = df.lat.min()
    latmax = df.lat.max()
    lonmin = df.lon.min()
    lonmax = df.lon.max()
    thisdate = lk.time_utc[:10]
    rgt = lk.rgt
    gtx = lk.gtx
    
    # download ATL03 data from NSIDC (if they're not already there)
    atl03_dir = 'data'
    fn_atl03 = atl03_dir + '/' + lk.granule_id
    if not os.path.isfile(fn_atl03):
        bbox = [lonmin, latmin, lonmax, latmax]
        granule_list = download_is2(start_date=thisdate, end_date=thisdate, rgt=rgt, boundbox=bbox, output_dir='data')
        for gran in granule_list:
            if lk.granule_id[:30] in gran:
                thisfile = atl03_dir + '/processed_' + gran
                break
        os.rename(thisfile, fn_atl03)
    return fn_atl03