import os
# os.environ["GDAL_DATA"] = "/Users/parndt/anaconda3/envs/eeicelakes-env/share/gdal"
# os.environ["PROJ_LIB"] = "/Users/parndt/anaconda3/envs/eeicelakes-env/share/proj"
# os.environ["PROJ_DATA"] = "/Users/parndt/anaconda3/envs/eeicelakes-env/share/proj"
import ee
import h5py
import math
import datetime
import requests
import traceback
import shapely
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime 
from datetime import timedelta
import rasterio as rio
from rasterio import plot as rioplot
from rasterio import warp
import matplotlib
import matplotlib.pylab as plt
from matplotlib.patches import Rectangle
from cmcrameri import cm as cmc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import Image, display

import sys
sys.path.append('../utils/')
from lakeanalysis.utils import dictobj, convert_time_to_string, read_melt_lake_h5

import warnings
warnings.filterwarnings("ignore")

#-------------------------------------------------------------------------------------
# this one is to add the extra surrf visualization data to a lake file if it is missing
def surrf_add_fit_weights(self, file, ext_input=None, final_resolution=5.0):

    # function for robust (iterative) nonparametric regression (to fit surface and bed of lake)
    def robust_npreg(df_fit, ext, n_iter=10, poly_degree=1, len_xatc_min=100, n_points=[200,50], 
        resolutions=[30,5], stds=[20,6], ext_buffer=250.0, full=False, init=None):
        
        h_list = []
        x_list = []
        n_phots = np.linspace(n_points[0], n_points[1], n_iter)
        resols = np.linspace(resolutions[0], resolutions[1], n_iter)
        n_stds = np.hstack((np.linspace(stds[0], stds[1], n_iter-1), stds[1]))
        minx = np.min(np.array(ext))
        maxx = np.max(np.array(ext))
    
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
            evaldf = pd.DataFrame(np.arange(minx-ext_buffer,maxx+ext_buffer+res,step=res),columns=['xatc'])
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
                if it < (n_iter-1):  # ignore SNR values in the last pass
                    dfi.weights *= dfi.snr
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
            width_vweight = np.clip(n_std*df_fit.std_fit,0.5, 10.0)
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

    # get the relevant data (photon-level dataframe, water surface elevation estimate, extent estimate)
    df = self.photon_data.copy()
    h_surf = self.surface_elevation
    if ext_input:
        ext = ext_input
    else:
        ext = self.surface_extent_detection
    init_guess_bed = pd.DataFrame(self.detection_2nd_returns)
    df.sort_values(by='xatc', inplace=True, ignore_index=True)

    if 'sat_ratio' in df.keys(): 
        df['is_afterpulse'] = df.prob_afterpulse>np.random.uniform(0,1,len(df))

    # fit the surface elevation only to photons just around and above the estimated water surface elevation
    df['in_extent'] = False
    for extseg in ext:
        df.loc[(df.xatc >= extseg[0]) & (df.xatc <= extseg[1]),'in_extent'] = True
    # change below h_surf-0.4
    # surffit_selector = (((df.h > (h_surf-0.4)) | (~df.in_extent)) & (df.snr > 0.5)) | ((df.h > (h_surf-0.3)) & (df.h < (h_surf+0.3)))
    surffit_selector = (((df.h > (h_surf-0.4)) | (~df.in_extent)) & (df.snr > 0.5))
    df_tofit_surf = df[surffit_selector].copy()
    evaldf_surf, df_fit_surf = robust_npreg(df_tofit_surf , ext, n_iter=10, poly_degree=1, len_xatc_min=20,
                                            n_points=[300,100], resolutions=[20,final_resolution], stds=[10,4], ext_buffer=250.0)

    # re-calculate water surface elevation based on fit
    hist_res = 0.001
    hist_smoothing = 0.05
    bins = np.arange(h_surf-1, h_surf+1+hist_res, hist_res)
    mids = bins[:-1] + np.diff(bins)
    histvals = np.histogram(evaldf_surf.h_fit, bins=bins)[0]
    hist_smooth = pd.Series(histvals).rolling(window=int(np.ceil(hist_smoothing/hist_res)),center=True, min_periods=1).mean()
    surf_elev = mids[np.argmax(hist_smooth)]

    # set the probability of photons being surface / water
    df['prob_surf'] = 0
    df.loc[df_fit_surf.index, 'prob_surf'] = df_fit_surf.vert_weight
    df['is_signal'] = df.prob_surf > 0.0
    df['surf_fit'] = np.interp(df.xatc, evaldf_surf.xatc, evaldf_surf.h_fit, left=np.nan, right=np.nan)
    width_water = 0.25
    df['is_water'] = (np.abs(df.surf_fit - surf_elev) < width_water) & ((df.h - df.surf_fit) > (-width_water))
    
    # get data frame with the water surface removed, and set a minimum for SNR, except for afterpulses
    df_nosurf = df[(~df.is_water) & (df.h < (surf_elev + 30)) & (df.h > (surf_elev - 50))].copy()
    
    # discard heavily saturated PMT ionization afterpulses (rarely an issue)
    if 'sat_ratio' in df_nosurf.keys(): 
        df_nosurf = df_nosurf[(df_nosurf.sat_ratio < 3.5) | ((surf_elev - df_nosurf.h) < 12.0)]
        df_nosurf.loc[df_nosurf.is_afterpulse, 'snr'] = 0.0
    
    # get an initial guess for the nonparametric regression fit to the lake bed (from secondary peaks during detection stage)
    init_guess_bed = init_guess_bed[(init_guess_bed.prom > 0.5) & (init_guess_bed.h < (surf_elev-2.0))]
    init_guess_surf = pd.DataFrame({'xatc': evaldf_surf.xatc, 'h': evaldf_surf.h_fit})
    init_guess_surf = init_guess_surf[init_guess_surf.h > (surf_elev+width_water)]
    init_guess = pd.concat((init_guess_bed, init_guess_surf), ignore_index=True).sort_values(by='xatc')
    init_guess_hsmooth = init_guess.h.rolling(window=5, center=True, min_periods=1).mean()
    is_bed = init_guess.h < surf_elev
    init_guess.loc[is_bed, 'h'] = init_guess_hsmooth[is_bed]
    
    # reduce the snr between the lake surface and initial guess, to mitigate the effect of subsurface scattering
    # (very occasionally, this can remove signal)
    if len(init_guess.h) > 0:
        df_nosurf['init_guess'] = np.interp(df_nosurf.xatc, init_guess.xatc, init_guess.h, left=np.nan, right=np.nan)
    else:
        df_nosurf['init_guess'] = np.ones_like(df_nosurf.xatc) * (surf_elev - 2.0)
    reduce_snr = (df_nosurf.h > (df_nosurf.init_guess + 1.0)) & (df_nosurf.h < surf_elev)
    df_nosurf['reduce_snr_factor'] = 1.0
    reduce_snr_factor = 1.0 - 1.5*((df_nosurf.h[reduce_snr] - (df_nosurf.init_guess[reduce_snr] + 1.0)) / 
                                ((surf_elev-width_water) - (df_nosurf[reduce_snr].init_guess + 1.0)))
    reduce_snr_factor = np.clip(reduce_snr_factor, 0, 1)
    df_nosurf.loc[reduce_snr, 'reduce_snr_factor'] = reduce_snr_factor
    df_nosurf.loc[reduce_snr, 'snr'] *= df_nosurf.reduce_snr_factor
    df_tofit_bed = df_nosurf.copy()
    df_nosurf = df_nosurf[df_nosurf.snr > 0].copy()
    
    # fit lakebed surface 
    npts = [100,50] if self.beam_strength=='weak' else [200,100]
    evaldf, df_fit_bed, xv, hv = robust_npreg(df_nosurf, ext, n_iter=20, poly_degree=1, len_xatc_min=100,
                                              n_points=npts, resolutions=[20,final_resolution], stds=[10,3], 
                                              ext_buffer=200.0, full=True, init=init_guess)

    # add probability of being lake bed for each photon
    df['prob_bed'] = 0
    df.loc[df_fit_bed.index, 'prob_bed'] = df_fit_bed.vert_weight
    df.loc[df.prob_bed>0.0,'is_signal'] = True
    df.loc[df.h > surf_elev, 'prob_bed'] = 0
    df.prob_bed /= df.prob_bed.max()

    # compile the results from surface and bed fitting into one data frame
    evaldf['h_fit_surf'] = np.interp(evaldf.xatc, evaldf_surf.xatc, evaldf_surf.h_fit)
    evaldf['stdev_surf'] = np.interp(evaldf.xatc, evaldf_surf.xatc, evaldf_surf.stdev)
    evaldf['is_water'] = np.abs(evaldf.h_fit_surf - surf_elev) < width_water
    
    df['bed_fit'] = np.interp(df.xatc, evaldf.xatc, evaldf.h_fit, left=np.nan, right=np.nan)
    df.loc[df.bed_fit > surf_elev,'prob_surf'] = 0.0

    # estimate the quality of the signal 
    std_range = 2.0  # calculate the photon density within this amount of residual MAEs for the bed density
    qual_smooth = 20  # along-track meters for smoothing the the quality measure 
    evaldf['lower'] = evaldf.h_fit-std_range*evaldf.stdev  # lower threshold for bed photon density 
    evaldf['upper'] = evaldf.h_fit+std_range*evaldf.stdev  # uppper threshold for bed photon density / lower threshold for lake interior 
    evaldf['hrange_bed'] = evaldf.upper - evaldf.lower  # the elevation range over which to calculate bed photon density
    evaldf['hrange_int'] = np.clip((surf_elev - evaldf.upper) * 0.5 , 0.5, None) # the elevation range over which to calculate interior photon density
    evaldf['upper_int'] = evaldf.h_fit + evaldf.hrange_bed/2 + evaldf.hrange_int # uppper threshold for lake interior photon density

    # initialize photon counts per depth measurement point, and get photon data frame with afterpulses removed
    num_bed = np.zeros_like(evaldf.xatc)
    num_interior = np.zeros_like(evaldf.xatc)
    df_nnz = df.copy()
    if 'is_afterpulse' in df_nnz.keys(): 
        df_nnz = df_nnz[~df_nnz.is_afterpulse]

    # loop through measurement points and count photons in the lake bed and lake interior ranges
    for i in range(len(evaldf)):
        vals = evaldf.iloc[i]
        in_xatc = (df_nnz.xatc > (vals.xatc-final_resolution)) & (df_nnz.xatc < (vals.xatc+final_resolution))
        in_range_bed = in_xatc & (df_nnz.h > vals.lower) & (df_nnz.h < vals.upper)
        in_range_interior = in_xatc & (df_nnz.h > vals.upper) & (df_nnz.h < vals.upper_int)
        num_bed[i] = np.sum(in_range_bed)
        num_interior[i] = np.sum(in_range_interior)

    # calculate the density ratio weight between the lake bed and the lake interior for each point
    # is zero if bed density is less or equal to interior density
    # approaches 1 as bed density becomes >> interior density
    # is 0 if there are no bed photons
    # is 1 if there are bed photons, but no interior photons
    evaldf['nph_bed'] = num_bed
    evaldf['nph_int'] = num_interior
    evaldf.loc[evaldf.nph_bed == 0, 'nph_bed'] = np.nan
    evaldf['density_ratio'] = 1 - np.clip((evaldf.nph_int / evaldf.hrange_int)/(evaldf.nph_bed / (evaldf.hrange_bed)), 0, 1)
    evaldf.loc[evaldf.h_fit > surf_elev,'density_ratio'] = 1.0 
    evaldf.loc[evaldf.nph_bed.isna(), 'density_ratio'] = 0.0

    # get the width ratio weight 
    # becomes 0 when the bed fit range includes the surface
    # becomes 1 when the interior range is at least as large as the lake bed fit range
    width_ratio = np.clip((evaldf.h_fit_surf+std_range*evaldf.stdev_surf - evaldf.upper),0,None) / (1.0*evaldf.hrange_bed)
    width_ratio[evaldf.h_fit > surf_elev] = 1.0
    evaldf['width_ratio'] = np.clip(width_ratio, 0, 1)
    
    # smooth out the weights a little 
    wdw = int(np.ceil(qual_smooth/final_resolution))
    evaldf['density_ratio'] = evaldf.density_ratio.rolling(window=3*wdw, win_type='gaussian', min_periods=1, center=True).mean(std=wdw/2)
    evaldf['width_ratio'] = evaldf.width_ratio.rolling(window=3*wdw, win_type='gaussian', min_periods=1, center=True).mean(std=wdw/2)

    # get the confidence in the measurement as the product between the two
    evaldf['conf'] = evaldf.density_ratio * evaldf.width_ratio

    # calculate the water depth
    evaldf['depth'] = np.clip(surf_elev - evaldf.h_fit, 0, None) / 1.336
    evaldf.loc[(~evaldf.is_water) & (evaldf.depth > 0.0), 'conf'] = 0.0
    evaldf.loc[~evaldf.is_water, 'depth'] = 0.0

    # multiply probability of bed by condifence in measurement
    df.prob_bed *= np.interp(df.xatc, evaldf.xatc, evaldf.conf, left=0.0, right=0.0)


    # get the overall lake quality
    df_bed = evaldf[(evaldf.h_fit < surf_elev) & (evaldf.h_fit < evaldf.h_fit_surf)].copy()
    nbins = 300
    counts = np.zeros((len(df_bed), nbins))
    
    for i in range(len(df_bed)):
        vals = df_bed.iloc[i]
        in_xatc = (df_nnz.xatc > (vals.xatc-final_resolution/2)) & (df_nnz.xatc < (vals.xatc+final_resolution/2))
        thisdf = df_nnz[in_xatc]
        # bins = np.linspace(vals.h_fit-vals.depth, surf_elev+vals.depth, nbins+1)
        hrng = vals.h_fit_surf - vals.h_fit
        bins = np.linspace(vals.h_fit-hrng, vals.h_fit_surf+hrng, nbins+1)
        hist = np.histogram(thisdf.h, bins=bins)[0]
        counts[i,:] = hist
    
    scaled_hist = np.sum(counts, axis=0)
    scaled_smooth = pd.Series(scaled_hist).rolling(window=int(nbins/10), win_type='gaussian', min_periods=1, center=True).mean(std=nbins/100)
    df_dens = pd.DataFrame({'x': np.linspace(-1,2,nbins), 'n': scaled_smooth})
    n_bedpeak = np.interp(0.0, df_dens.x, df_dens.n)
    df_dens_int = df_dens[(df_dens.x > 0) & (df_dens.x < 1)].copy()
    # n_saddle = np.min(df_dens_int.n)
    n_saddle = np.mean(df_dens_int.n[df_dens_int.n < np.percentile(df_dens_int.n, 25)])
    depth_quality = np.clip(n_bedpeak / n_saddle - 2.0, 0, None)

    evaldf['h_fit_bed'] = evaldf.h_fit
    evaldf['std_bed'] = evaldf.stdev
    evaldf['std_surf'] = evaldf.stdev_surf
    df['xatc_10m'] = np.round(df.xatc, -1)
    df_spatial = df[['lat', 'lon', 'xatc', 'xatc_10m']][df.is_signal].groupby('xatc_10m').median()
    evaldf['lat'] = np.interp(evaldf.xatc, df_spatial.xatc, df_spatial.lat, right=np.nan, left=np.nan)
    evaldf['lon'] = np.interp(evaldf.xatc, df_spatial.xatc, df_spatial.lon, right=np.nan, left=np.nan)
    evaldf = evaldf[~evaldf.lat.isna()]
    evaldf = evaldf[~evaldf.lon.isna()]

    self.photon_data = df[list(np.unique(list(self.photon_data.keys()) + ['prob_surf', 'prob_bed', 'is_signal', 'is_afterpulse']))]

    # self.photon_data['prob_surf'] = df.prob_surf
    # self.photon_data['prob_bed'] = df.prob_bed
    # self.photon_data['is_signal'] = df.is_signal
    # self.photon_data['is_afterpulse'] = df.is_afterpulse
    self.depth_data = evaldf[['xatc', 'lat', 'lon', 'depth', 'conf', 'h_fit_surf', 'h_fit_bed', 'std_surf', 'std_bed']].copy()
    self.surface_elevation = surf_elev
    self.lake_quality = depth_quality
    self.max_depth = evaldf.depth[evaldf.conf>0.0].max()

    df_tofit_surf = df_tofit_surf[~df_tofit_surf.is_afterpulse].copy()
    df_tofit_bed = df_tofit_bed[(~df_tofit_bed.is_afterpulse) & (df_tofit_bed.snr>0)].copy()

    try:
        with h5py.File(file, 'r+') as f:
            comp="gzip"
            if 'init_guess' in f.keys():
                del f['init_guess']
            initdat = f.create_group('init_guess')
            initdat.create_dataset('xatc', data=np.array(init_guess.xatc), compression=comp)
            initdat.create_dataset('h', data=np.array(init_guess.h), compression=comp)

            if 'ph_tofit_surf' in f.keys():
                del f['ph_tofit_surf']
            surfdat = f.create_group('ph_tofit_surf')
            surfdat.create_dataset('xatc', data=np.array(df_tofit_surf.xatc), compression=comp)
            surfdat.create_dataset('h', data=np.array(df_tofit_surf.h), compression=comp)
            surfdat.create_dataset('snr', data=np.array(df_tofit_surf.snr), compression=comp)

            if 'ph_tofit_bed' in f.keys():
                del f['ph_tofit_bed']
            beddat = f.create_group('ph_tofit_bed')
            beddat.create_dataset('xatc', data=np.array(df_tofit_bed.xatc), compression=comp)
            beddat.create_dataset('h', data=np.array(df_tofit_bed.h), compression=comp)
            beddat.create_dataset('snr', data=np.array(df_tofit_bed.snr), compression=comp)

    except:
        print('WARNING: SuRRF fit photons could not be written to file!')
        traceback.print_exc()

    return self, df_tofit_surf, df_tofit_bed, init_guess, df_dens


def print_lake_info(fn, description='', print_imagery_info=True):
    lk = dictobj(read_melt_lake_h5(fn))
    keys = vars(lk).keys()
    print('\nLAKE INFO: %s' % description)
    print('  granule_id:            %s' % lk.granule_id)
    print('  RGT:                   %s' % lk.rgt)
    print('  GTX:                   %s' % lk.gtx.upper())
    print('  beam:                  %s (%s)' % (lk.beam_number, lk.beam_strength))
    print('  acquisition time:      %s' % lk.date_time)
    print('  center location:       (%s, %s)' % (lk.lon_str, lk.lat_str))
    print('  ice sheet:             %s' % lk.ice_sheet)
    print('  melt season:           %s' % lk.melt_season)
    print('  SuRRF lake quality:    %.2f' % lk.lake_quality)
    print('  surface_elevation:     %.2f m' % lk.surface_elevation)
    print('  maximum water depth:   %.2f m' % lk.max_depth)
    print('  water surface length:  %.2f km' % lk.len_surf_km)
    
    if ('imagery_info' in keys) and (print_imagery_info):
        print('  IMAGERY INFO:')
        print('    product ID:                     %s' % lk.imagery_info['product_id'])
        print('    acquisition time imagery:       %s' % lk.imagery_info['time_imagery'])
        print('    acquisition time ICESat-2:      %s' % lk.imagery_info['time_icesat2'])
        print('    time difference from ICESat-2:  %s (%s)' % (lk.imagery_info['time_diff_from_icesat2'],lk.imagery_info['time_diff_string']))
        print('    mean cloud probability:         %.1f %%' % lk.imagery_info['mean_cloud_probability'])
    print('')