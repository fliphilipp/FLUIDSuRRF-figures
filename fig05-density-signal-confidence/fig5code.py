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

def get_signal(df, n_iter=100, h_thresh=[200,3], xatc_win=[500,7]):
    number_iterations = n_iter
    elevation_threshold_start = h_thresh[0]
    elevation_threshold_end = h_thresh[1]
    xatc_window_width_start = xatc_win[0]
    xatc_window_width_end = xatc_win[1]
    
    df.xatc -= df.xatc.min()
    df['is_signal'] = True
    df['rmean'] = np.nan
    
    def get_params(start,end,number):
        return [start * ((end / start)**(1/(number-1))) ** it for it in range(number)]
    
    h_diff_thresholds = get_params(start=elevation_threshold_start, end=elevation_threshold_end, number=number_iterations)
    window_sizes = np.int32(np.round(get_params(start=xatc_window_width_start, end=xatc_window_width_end, number=number_iterations)))

    for i in range(number_iterations):
        rmean = df.h[df.is_signal].rolling(3*window_sizes[i],center=True,min_periods=5,win_type='gaussian').mean(
            std=int(np.ceil(window_sizes[i]/2)))
        df['rmean'] = np.interp(df.xatc, df.xatc[df.is_signal], rmean)
        df['is_signal'] = np.abs(df.rmean-df.h) < h_diff_thresholds[i]
        
    rmean = df.h[df.is_signal].rolling(5*window_sizes[i],center=True,min_periods=5,win_type='gaussian').mean(std=window_sizes[-1])
    df['rmean'] = np.interp(df.xatc, df.xatc[df.is_signal], rmean)
    
    return df
    

# function for robust (iterative) nonparametric regression (to fit surface and bed of lake)
def robust_npreg(df_fit, n_iter=10, poly_degree=1, len_xatc_min=5, n_points=[100,30], 
    resolutions=[5,1], stds=[20,6], full=False, init=None):
    
    h_list = []
    x_list = []
    n_phots = np.linspace(n_points[0], n_points[1], n_iter)
    resols = np.linspace(resolutions[0], resolutions[1], n_iter)
    n_stds = np.hstack((np.linspace(stds[0], stds[1], n_iter-1), stds[1]))
    minx = df_fit.xatc.min()
    maxx = df_fit.xatc.max()

    # take into account initial guess, if specified (needs to be dataframe with columns 'xatc' and 'h')
    if (init is not None) and (len(init) > 0): 
        range_vweight = 10.0
        df_fit['heights_fit'] = np.interp(df_fit.xatc, init.xatc, init.h, left=np.nan, right=np.nan)
        vert_weight = (1.0 - np.clip((np.abs(df_fit.h-df_fit.heights_fit)/range_vweight),0,1)**3 )**3
        vert_weight[np.isnan(vert_weight)] = 0.01
        df_fit['vert_weight'] = vert_weight
    else: 
        df_fit['vert_weight'] = 1.0
    
    for it in range(n_iter):
        
        n_phot = n_phots[it]
        res = resols[it]
        n_std = n_stds[it]
        evaldf = pd.DataFrame(np.arange(minx,maxx+res,step=res),columns=['xatc'])
        h_arr = np.full_like(evaldf.xatc,fill_value=np.nan)
        stdev_arr = np.full_like(evaldf.xatc,fill_value=np.nan)
        df_fit_nnz = df_fit[df_fit.vert_weight > 1e-3].copy()

        # for every point at which to evaluate local fit
        for i,x in enumerate(evaldf.xatc):
            
            # look for the closest n_phot photons around the center point for local polynomial fit
            idx_closest_photon = np.argmin(np.abs(np.array(df_fit_nnz.xatc - x)))
            n_phot_each_side = int(np.ceil(n_phot / 2))
            idx_start = np.clip(idx_closest_photon - n_phot_each_side, 0, None)
            idx_end = np.clip(idx_closest_photon + n_phot_each_side +1, None, len(df_fit_nnz)-1)
            xatc_start = df_fit_nnz.iloc[idx_start].xatc
            xatc_end = df_fit_nnz.iloc[idx_end].xatc
            len_xatc = xatc_end - xatc_start

            # if the fit for n_phot does not span at least len_xatc_min, then make the segment longer
            if len_xatc < len_xatc_min: 
                xstart = x - len_xatc_min/2
                xend = x + len_xatc_min/2
                idx_start = np.min((int(np.clip(np.argmin(np.abs(np.array(df_fit_nnz.xatc - xstart))), 0, None)), idx_start))
                idx_end = np.max((int(np.clip(np.argmin(np.abs(np.array(df_fit_nnz.xatc - xend))), None, len(df_fit_nnz)-1)), idx_end))

            # make a data frame with the data for the fit
            dfi = df_fit_nnz.iloc[idx_start:idx_end].copy()

            # tricube weights for xatc distance from evaluation point
            maxdist = np.nanmax(np.abs(dfi.xatc - x))
            dfi['weights'] = (1.0-(np.abs(dfi.xatc-x)/(1.00001*maxdist))**3)**3

            # also weight by the SNR values and the vertical distance from previous fit 
            dfi.weights *= dfi.density
            if (init is not None) | (it > 0):  # vertical weights are only available after first iteration or with initial guess
                dfi.weights *= dfi.vert_weight

            # do the polynomial fit
            try: 
                reg_model = np.poly1d(np.polyfit(dfi.xatc, dfi.h, poly_degree, w=dfi.weights))
                h_arr[i] = reg_model(x)
                stdev_arr[i] = np.average(np.abs(dfi.h - reg_model(dfi.xatc)), weights=dfi.weights) # use weighted mean absolute error
            except:  # if polynomial fit does not converge, use a weighted average
                h_arr[i] = np.average(dfi.h,weights=dfi.weights)
                stdev_arr[i] = np.average(np.abs(dfi.h - h_arr[i]), weights=dfi.weights) # use weighted mean absolute error
            
        evaldf['h_fit'] = h_arr
        evaldf['stdev'] = stdev_arr
        
        # interpolate the fit and residual MAE to the photon-level data
        df_fit['heights_fit'] = np.interp(df_fit.xatc, evaldf.xatc, evaldf.h_fit, left=-9999, right=-9999)
        df_fit['std_fit'] = np.interp(df_fit.xatc, evaldf.xatc, evaldf.stdev)

        # compute tricube weights for the vertical distance for the next iteration
        width_vweight = np.clip(n_std*df_fit.std_fit,0.0, 10.0)
        df_fit['vert_weight'] = (1.0 - np.clip((np.abs(df_fit.h-df_fit.heights_fit)/width_vweight),0,1)**3 )**3
        df_fit.loc[df_fit.heights_fit == -9999, 'vert_weight'] = 0.0 # give small non-zero weight for leading and trailing photons
        
        if full:
            h_list.append(h_arr)
            x_list.append(evaldf.xatc)

        # print('iteration %i / %i' % (it+1, n_iter), end='\r')

    if full:
        return evaldf, df_fit, x_list, h_list
    else:
        return evaldf, df_fit

# for applying density estimation (not needed here but for reference)
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