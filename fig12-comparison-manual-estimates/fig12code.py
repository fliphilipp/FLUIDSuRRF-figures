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
from collections import defaultdict
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import matplotlib.patheffects as path_effects

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
            y = -(height/2+ydescent/2) + 2*i*ydescent
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

#####################################################################
def surrf_correction(self, final_resolution=5.0, ext_buffer=200.0, correct=True, correct_extent=True, hrange_water=0.3,
                     footprint_diameter=11, photon_precision_time=800e-12, reduce_snr_below_init=True):

    # function for robust (iterative) nonparametric regression (to fit surface and bed of lake)
    def robust_npreg(df_fit, ext, n_iter=10, poly_degree=1, len_xatc_min=100, n_points=[300,100], 
        resolutions=[30,5], stds=[20,6], ext_buffer=250.0, full=False, init=None, vweight_start=10):
        
        h_list = []
        x_list = []
        n_phots = np.linspace(n_points[0], n_points[1], n_iter)
        resols = np.linspace(resolutions[0], resolutions[1], n_iter)
        n_stds = np.hstack((np.linspace(stds[0], stds[1], n_iter-1), stds[1]))
        minx = np.min(np.array(ext))
        maxx = np.max(np.array(ext))
    
        # take into account initial guess, if specified (needs to be dataframe with columns 'xatc' and 'h')
        if (init is not None) and (len(init) > 0): 
            range_vweight = vweight_start
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
            width_vweight = np.clip(n_std*df_fit.std_fit,0.0, 10.0)
            if 'min_vertrange' in df_fit.keys():
                width_vweight = pd.concat((width_vweight, df_fit.min_vertrange),axis=1).max(axis=1)
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

    # remove scattering data if correct is true and lake already has depth data
    # this allows the function to be used for the initial surrf guess as well as the correction
    iscorrection = correct and hasattr(self, 'depth_data')
    if iscorrection:
        print('applying SuRRF scattering correction.')
        width_water = hrange_water
        vweight_start_surf = 3
        vweight_start_bed = 3
        dfd = self.depth_data.copy()
        hsurf = self.surface_elevation
        h_surf = self.surface_elevation # used two different names, too lazy to change right now...

        if correct_extent:
            xbounds = list(intersection(dfd.xatc, dfd.h_fit_bed, dfd.xatc, np.ones_like(dfd.h_fit_bed)*hsurf)[0])
            smaller = dfd.xatc - xbounds[0]
            smaller = smaller[smaller<0]
            idx = len(smaller-1)
            if dfd.h_fit_bed.iloc[idx] < dfd.h_fit_bed.iloc[idx+1]:
                xbounds = xbounds[1:]
            if (len(xbounds) % 2) != 0:
                xbounds = xbounds[:-1]
            ext = [xbounds[iseg:iseg+2] for iseg in np.arange(0,len(xbounds),2)]
        else:
            ext = self.surface_extent_detection
        if len(ext) == 0:
            return None

        #footprint_diameter = 11 # icesat-2 footprint diameter in meters
        ref_idx = 1.336 # refractive index for the speed of light in 0 degree C freshwater
        #photon_precision_time = 800e-12 # Single-photon time-of-flight precision: 800 ps
        speed_of_light = 299792458 # m/s in vacuum
        def get_footprint_hrange(x):
            footprint_border = np.asarray([x.xatc - footprint_diameter/2, x.xatc + footprint_diameter/2])
            return np.abs(np.diff(np.interp(footprint_border, dfd.xatc, dfd.h_fit_bed))[0])
        hrange_footprint_atl03 = dfd.apply(get_footprint_hrange, axis=1)
        hrange_footprint_depth = hrange_footprint_atl03 / ref_idx
        uncert_atl03 = speed_of_light * photon_precision_time / 2
        uncert_depth = uncert_atl03 / ref_idx
        
        total_spread_atl03 = hrange_footprint_atl03 + uncert_atl03
        total_spread_depth = hrange_footprint_depth + uncert_depth
        upper_lim_atl03 = dfd.h_fit_bed + total_spread_atl03 / 2
        lower_lim_atl03 = dfd.h_fit_bed - total_spread_atl03 / 2
        dfd['upper_lim_atl03'] = upper_lim_atl03
        dfd['lower_lim_atl03'] = lower_lim_atl03

        df = self.photon_data.copy()
        df['min_vertrange'] = np.interp(df.xatc, dfd.xatc, total_spread_atl03/2)
        df['lower_lim'] = np.interp(df.xatc, dfd.xatc, lower_lim_atl03)
        df['correction_keep'] = df.h >= df.lower_lim
        df['bedfit'] = np.interp(df.xatc, dfd.xatc, dfd.h_fit_bed)
        diff = np.abs(np.clip(df.h - df.bedfit, None, 0))
        wd = df.bedfit - df.lower_lim
        df['reduce_snr_correction'] = (1- np.clip(np.abs(diff)/(1.00001*wd),0,1)**2)**3
        #df['reduce_snr_correction'] = (1- np.clip(np.abs(diff)/(1.00001*wd),0,1)**2)**5 * (1- np.clip(np.abs(diff)/(1.00001*wd),0,1))
        wd_shallow = (df.bedfit - df.lower_lim) / 5
        shallow_thresh = 1.0
        is_shallow = (hsurf - df.bedfit) < shallow_thresh
        df.loc[is_shallow, 'reduce_snr_correction'] = (1- np.clip(np.abs(diff[is_shallow])/(1.00001*wd_shallow),0,1)**2)**3

        # use old estimate as initial guess, rename column to match FLUID bathymetry peak labels
        init_guess = dfd[['xatc', 'h_fit_bed']].rename(columns={'h_fit_bed': 'h'})

    else:
        # get the relevant data (photon-level dataframe, water surface elevation estimate, extent estimate)
        print('applying surrf from raw data, no scattering correction.')
        width_water = 0.35
        vweight_start_surf = 10
        vweight_start_bed = 10
        ext = self.surface_extent_detection
        if len(ext) == 0:
            return None
        df = self.photon_data.copy()
        h_surf = self.surface_elevation
        
        init_guess_bed = pd.DataFrame(self.detection_2nd_returns)
        
    df.sort_values(by='xatc', inplace=True, ignore_index=True)

    # if ('sat_ratio' in df.keys()) & (not (correct and hasattr(self, 'depth_data'))): 
    #     df['is_afterpulse'] = df.prob_afterpulse>np.random.uniform(0,1,len(df))
    if 'sat_ratio' in df.keys(): 
        df['is_afterpulse'] = df.prob_afterpulse>np.random.uniform(0,1,len(df))

    # fit the surface elevation only to photons just around and above the estimated water surface elevation
    ext_buffer_meters = 30
    df['in_extent'] = False
    df['in_extent_buffer'] = False
    for extseg in ext:
        df.loc[(df.xatc >= extseg[0]) & (df.xatc <= extseg[1]),'in_extent'] = True
    ext_buffer_water = [[xt[0]-ext_buffer_meters, xt[1]+ext_buffer_meters] for xt in ext]
    for extseg in ext_buffer_water:
        df.loc[(df.xatc >= extseg[0]) & (df.xatc <= extseg[1]),'in_extent_buffer'] = True
    surffit_selector = (((df.h > (h_surf-0.2)) | (~df.in_extent_buffer)) & (df.snr > 0.2))  |  ((df.h > (h_surf-0.15)) & (df.h < (h_surf+0.15)))
    df_fit = df[surffit_selector].copy()
    if 'reduce_snr_correction' in df_fit.keys():
        df_fit.snr *= df_fit['reduce_snr_correction']
    min_len = 5 if iscorrection else 20
    npts = [100, 50] if iscorrection else [300,100]
    start_resolution = 5 if iscorrection else 20
    evaldf_surf, df_fit_surf = robust_npreg(df_fit, ext, n_iter=10, poly_degree=1, len_xatc_min=min_len, vweight_start=vweight_start_surf, 
                                            n_points=npts, resolutions=[start_resolution,final_resolution], stds=[10,4], ext_buffer=ext_buffer+10.0)

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

    in_ext_and_water_range = (df['in_extent']  |  ((df['in_extent_buffer']) & (np.abs(df.surf_fit - surf_elev) < 0.275)))
    df['is_water'] =  in_ext_and_water_range  &  ((((df.h - surf_elev) > (-width_water)) & (df.snr < 0.75)) | (np.abs(df.h - surf_elev) < width_water))
    
    # get data frame with the water surface removed, and set a minimum for SNR, except for afterpulses
    df_nosurf = df[(~df.is_water) & (df.h < (surf_elev + 30)) & (df.h > (surf_elev - 50))].copy()
    min_snr = 0.2
    df_nosurf['snr'] = min_snr + (1.0-min_snr)*df_nosurf.snr

    # if correcting for scattering, take out the bottom scattering photons again
    if not reduce_snr_below_init:
        df_nosurf['reduce_snr_correction'] = 1
    if 'reduce_snr_correction' in df_nosurf.keys():
            df_nosurf.snr *= df_nosurf['reduce_snr_correction']
    
    # discard heavily saturated PMT ionization afterpulses (rarely an issue)
    if 'sat_ratio' in df_nosurf.keys(): 
        df_nosurf = df_nosurf[(df_nosurf.sat_ratio < 3.5) | ((surf_elev - df_nosurf.h) < 12.0)]
        df_nosurf.loc[df_nosurf.is_afterpulse, 'snr'] = 0.0

    if iscorrection:
        df_nosurf['init_guess'] = np.interp(df_nosurf.xatc, dfd.xatc, dfd.h_fit_bed, left=np.nan, right=np.nan)
    else:
        # get an initial guess for the nonparametric regression fit to the lake bed (from secondary peaks during detection stage)
        init_guess_bed = init_guess_bed[(init_guess_bed.prom > 0.3) & (init_guess_bed.h < (surf_elev-2.0))]
        init_guess_surf = pd.DataFrame({'xatc': evaldf_surf.xatc, 'h': evaldf_surf.h_fit})
        init_guess_surf = init_guess_surf[init_guess_surf.h > (surf_elev+width_water)]
        init_guess = pd.concat((init_guess_bed, init_guess_surf), ignore_index=True).sort_values(by='xatc')
        init_guess_hsmooth = init_guess.h.rolling(window=5, center=True, min_periods=1).mean()
        is_bed = init_guess.h < surf_elev
        init_guess.loc[is_bed, 'h'] = init_guess_hsmooth[is_bed]
        
        if len(init_guess.h) > 0:
            df_nosurf['init_guess'] = np.interp(df_nosurf.xatc, init_guess.xatc, init_guess.h, left=np.nan, right=np.nan)
        else:
            df_nosurf['init_guess'] = np.ones_like(df_nosurf.xatc) * (surf_elev - 2.0)

    # re-calculate densities
    df_nosurf = df_nosurf[df_nosurf.snr > 0].copy()
    df_new = df_nosurf.copy()
    df_new = get_signal(df_new, n_iter=100, h_thresh=[30,0.5], xatc_win=[40,5])
    df_new = get_density(df_new, segment_length=20, signal_width=0.2, aspect=30, K_phot=30, frac_noise=0.05)
    df_nosurf['snr'] = df_new.density

    # if correcting for scattering, take out the bottom scattering photons again
    if 'reduce_snr_correction' in df_nosurf.keys():
            df_nosurf.snr *= df_nosurf['reduce_snr_correction']

    # reduce the snr between the lake surface and initial guess, to mitigate the effect of subsurface scattering
    # (very occasionally, this can remove signal)
    reduce_snr_bedbuffer = df_nosurf.min_vertrange if 'min_vertrange' in df_nosurf.keys() else 1.0
    reduce_overshoot = 1.0 if 'min_vertrange' in df_nosurf.keys() else 1.5
    reduce_snr = (df_nosurf.h > (df_nosurf.init_guess + reduce_snr_bedbuffer)) & (df_nosurf.h < surf_elev)
    df_nosurf['reduce_snr_factor'] = 1.0
    reduce_snr_factor = 1.0 - reduce_overshoot*((df_nosurf.h[reduce_snr] - (df_nosurf.init_guess[reduce_snr] + reduce_snr_bedbuffer)) / 
                                (surf_elev - (df_nosurf[reduce_snr].init_guess + reduce_snr_bedbuffer)))
    reduce_snr_factor = np.clip(reduce_snr_factor, 0, 1)
    df_nosurf.loc[reduce_snr, 'reduce_snr_factor'] = reduce_snr_factor
    df_nosurf.loc[reduce_snr, 'snr'] *= df_nosurf.reduce_snr_factor
    df_nosurf = df_nosurf[df_nosurf.snr > 0].copy()

    start_resolution = 5 if iscorrection else 20
    len_xatc_min = 50 if iscorrection else 100
    stds = [7, 4] if iscorrection else [10, 3]
    # weakphots = [50,30] if iscorrection else [100,50]
    # strongphots = [100,50] if iscorrection else [200,100]
    weakphots = [100,50]
    strongphots = [200,100]
    
    # fit lakebed surface 
    npts = weakphots if self.beam_strength=='weak' else strongphots
    evaldf, df_fit_bed = robust_npreg(df_nosurf, ext, n_iter=20, poly_degree=3, len_xatc_min=len_xatc_min,
                                              n_points=npts, resolutions=[start_resolution,final_resolution], stds=stds, 
                                              ext_buffer=ext_buffer, full=False, init=init_guess, vweight_start=vweight_start_bed
                                     )

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
    qual_smooth = 40  # along-track meters for smoothing the the quality measure 
    if iscorrection:
        evaldf['upper_init'] = np.interp(evaldf.xatc, dfd.xatc, dfd.upper_lim_atl03)
        hrange = evaldf.upper_init - evaldf.h_fit
        depth_est_atl03 = np.clip(surf_elev - evaldf.h_fit, 0, None)
        evaldf['hrange'] = np.clip(hrange * depth_est_atl03/2, 0.15, 2.0)
        evaldf['lower'] = evaldf.h_fit - evaldf.hrange
        evaldf['upper'] = evaldf.h_fit + evaldf.hrange
        # evaldf['lower_init'] = np.interp(evaldf.xatc, dfd.xatc, dfd.lower_lim_atl03)
        # evaldf['upper'] = np.interp(evaldf.xatc, dfd.xatc, dfd.upper_lim_atl03)
        # notshallow = (surf_elev - evaldf.h_fit) > 3.5
        # evaldf.loc[notshallow, 'lower'] = evaldf.h_fit - (evaldf.h_fit-evaldf.lower) * 2.5
        # evaldf.loc[notshallow, 'upper'] = evaldf.h_fit + (evaldf.upper - evaldf.h_fit) * 2.5
    else:
        evaldf['lower'] = evaldf.h_fit-std_range*evaldf.stdev  # lower threshold for bed photon density 
        evaldf['upper'] = evaldf.h_fit+std_range*evaldf.stdev  # uppper threshold for bed photon density / lower threshold for lake interior 
    evaldf['hrange_bed'] = evaldf.upper - evaldf.lower  # the elevation range over which to calculate bed photon density
    evaldf['hrange_int'] = np.clip((surf_elev - evaldf.upper) * 0.5 , 0.5, None) # the elevation range over which to calculate interior photon density
    evaldf['upper_int'] = evaldf.h_fit + evaldf.hrange_bed/2 + evaldf.hrange_int # upper threshold for lake interior photon density

    # initialize photon counts per depth measurement point, and get photon data frame with afterpulses removed
    num_bed = np.zeros_like(evaldf.xatc)
    num_interior = np.zeros_like(evaldf.xatc)
    if iscorrection:
        df_nnz = df_nosurf.copy()
    else:
        df_nnz = df.copy()
    if 'is_afterpulse' in df_nnz.keys(): 
        df_nnz = df_nnz[~df_nnz.is_afterpulse]

    # loop through measurement points and count photons in the lake bed and lake interior ranges
    for i in range(len(evaldf)):
        vals = evaldf.iloc[i]
        in_xatc = (df_nnz.xatc > (vals.xatc-final_resolution)) & (df_nnz.xatc < (vals.xatc+final_resolution))
        in_range_bed = in_xatc & (df_nnz.h > vals.lower) & (df_nnz.h < vals.upper)
        in_range_interior = in_xatc & (df_nnz.h > vals.upper) & (df_nnz.h < vals.upper_int)
        if iscorrection:
            # for correction take (now likely more accurate SNR values into account)
            num_bed[i] = df_nnz.snr[in_range_bed].sum()
            num_interior[i] = df_nnz.snr[in_range_interior].sum()
        else:
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
    evaldf['density_ratio'] = 1 - np.clip((evaldf.nph_int / evaldf.hrange_int)/(evaldf.nph_bed / evaldf.hrange_bed), 0, 1)
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
    depth_quality = np.clip(n_bedpeak / n_saddle - 2, 0, None)
    depth_quality_sort = np.clip(n_bedpeak / n_saddle, 0, None)

    evaldf['h_fit_bed'] = evaldf.h_fit
    evaldf['std_bed'] = evaldf.stdev
    evaldf['std_surf'] = evaldf.stdev_surf
    df['xatc_10m'] = np.round(df.xatc, -1)
    df_spatial = df[['lat', 'lon', 'xatc', 'xatc_10m']][df.is_signal].groupby('xatc_10m').median()
    evaldf['lat'] = np.interp(evaldf.xatc, df_spatial.xatc, df_spatial.lat, right=np.nan, left=np.nan)
    evaldf['lon'] = np.interp(evaldf.xatc, df_spatial.xatc, df_spatial.lon, right=np.nan, left=np.nan)
    evaldf = evaldf[~evaldf.lat.isna()]
    evaldf = evaldf[~evaldf.lon.isna()]

    # self.photon_data['prob_surf'] = df.prob_surf
    # self.photon_data['prob_bed'] = df.prob_bed
    # self.photon_data['is_signal'] = df.is_signal
    # self.photon_data['is_afterpulse'] = df.is_afterpulse
    hfitsurf_all = evaldf.h_fit_surf.copy()
    prob_adjust_elev = 1 - np.interp(df.xatc, evaldf.xatc, np.clip((np.abs(hfitsurf_all-surf_elev)-0.03)/0.2, 0, 1))
    leftx = df.xatc - evaldf[evaldf.depth > 0].xatc.min()
    rightx = evaldf[evaldf.depth > 0].xatc.max() - df.xatc
    prob_adjust_sides = np.clip(np.min(np.vstack((leftx, rightx)), axis=0)/50, 0, 1)
    prob_adjust = prob_adjust_elev * prob_adjust_sides
    df.prob_surf *= prob_adjust
    df.prob_bed *= prob_adjust

    hfitsurf_all = evaldf.h_fit_surf.copy()
    prob_adjust_elev = 1 - np.clip((np.abs(hfitsurf_all-surf_elev)-0.03)/0.2, 0, 1)
    prob_adjust_elev[evaldf.depth <= 0] = 1.0
    evaldf.conf *= prob_adjust_elev

    # save results to lake object
    self.photon_data = df[list(set(list(self.photon_data.keys()) + ['prob_surf', 'prob_bed', 'is_signal', 'is_afterpulse'
                                                                   ]))].sort_values(by='xatc').reset_index(drop=True)
    self.depth_data = evaldf[['xatc', 'lat', 'lon', 'depth', 'conf', 'h_fit_surf', 'h_fit_bed', 'std_surf', 'std_bed']].copy()
    self.surface_elevation = surf_elev
    self.lake_quality = depth_quality
    depth_quality_sort = 0.0 if np.isnan(depth_quality_sort) else depth_quality_sort
    self.quality_sort = depth_quality_sort
    hqdepth = evaldf.depth[(evaldf.conf>0.3) & (np.abs(evaldf.h_fit_surf - surf_elev) < 0.25) & (evaldf.depth >= 0.0)]
    self.max_depth = hqdepth.max() if len(hqdepth) > 0 else 0.0
    if correct and hasattr(self, 'depth_data'):
        self.corrected = True
    else:
        self.corrected = False

    df_depth = evaldf[['xatc', 'lat', 'lon', 'depth', 'conf', 'h_fit_surf', 'h_fit_bed', 'std_surf', 'std_bed']].copy()
    return df_depth, df_fit, df_nosurf, init_guess, ext, evaldf

def _rect_inter_inner(x1, x2):
    n1 = x1.shape[0]-1
    n2 = x2.shape[0]-1
    X1 = np.c_[x1[:-1], x1[1:]]
    X2 = np.c_[x2[:-1], x2[1:]]
    S1 = np.tile(X1.min(axis=1), (n2, 1)).T
    S2 = np.tile(X2.max(axis=1), (n1, 1))
    S3 = np.tile(X1.max(axis=1), (n2, 1)).T
    S4 = np.tile(X2.min(axis=1), (n1, 1))
    return S1, S2, S3, S4


def _rectangle_intersection_(x1, y1, x2, y2):
    S1, S2, S3, S4 = _rect_inter_inner(x1, x2)
    S5, S6, S7, S8 = _rect_inter_inner(y1, y2)

    C1 = np.less_equal(S1, S2)
    C2 = np.greater_equal(S3, S4)
    C3 = np.less_equal(S5, S6)
    C4 = np.greater_equal(S7, S8)

    ii, jj = np.nonzero(C1 & C2 & C3 & C4)
    return ii, jj


def intersection(x1, y1, x2, y2):

    x1 = np.asarray(x1)
    x2 = np.asarray(x2)
    y1 = np.asarray(y1)
    y2 = np.asarray(y2)

    ii, jj = _rectangle_intersection_(x1, y1, x2, y2)
    n = len(ii)

    dxy1 = np.diff(np.c_[x1, y1], axis=0)
    dxy2 = np.diff(np.c_[x2, y2], axis=0)

    T = np.zeros((4, n))
    AA = np.zeros((4, 4, n))
    AA[0:2, 2, :] = -1
    AA[2:4, 3, :] = -1
    AA[0::2, 0, :] = dxy1[ii, :].T
    AA[1::2, 1, :] = dxy2[jj, :].T

    BB = np.zeros((4, n))
    BB[0, :] = -x1[ii].ravel()
    BB[1, :] = -x2[jj].ravel()
    BB[2, :] = -y1[ii].ravel()
    BB[3, :] = -y2[jj].ravel()

    for i in range(n):
        try:
            T[:, i] = np.linalg.solve(AA[:, :, i], BB[:, i])
        except:
            T[:, i] = np.Inf

    in_range = (T[0, :] >= 0) & (T[1, :] >= 0) & (
        T[0, :] <= 1) & (T[1, :] <= 1)

    xy0 = T[2:, in_range]
    xy0 = xy0.T
    return xy0[:, 0], xy0[:, 1]

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

def get_density(df, segment_length=20, signal_width=0.1, aspect=30, K_phot=20, frac_noise=0.05):
    n_segs = int(np.ceil((df.xatc.max()-df.xatc.min())/segment_length))
    len_seg = (df.xatc.max()-df.xatc.min())/n_segs
    edges = np.arange(df.xatc.min(), df.xatc.max()+len_seg/2, len_seg)
    df['density'] = 0.0
    for i in range(len(edges)-1):
        if 'is_afterpulse' in df.keys():
            selector_segment = (df.xatc>=edges[i]) & (df.xatc<edges[i+1]) & (~df.is_afterpulse)
        else:
            selector_segment = (df.xatc>=edges[i]) & (df.xatc<edges[i+1])

        if np.sum(selector_segment) > 0:
            dfseg = df[selector_segment].copy()
            dfseg_nosurface = dfseg[np.abs(dfseg.h-dfseg.rmean) > signal_width/2]
            nphot_bckgrd = len(dfseg_nosurface.h)
        
            # radius of a circle in which we expect to find one noise photon
            telem_h = np.nanmax((dfseg_nosurface.h.max()-dfseg_nosurface.h.min(), 15.0))
            h_noise = telem_h-signal_width
            wid_noise = segment_length/aspect
            area = h_noise*wid_noise/(nphot_bckgrd+1)
            fac = 3
            wid = np.sqrt(fac*frac_noise*(K_phot+1)*area/np.pi)
        
            # buffer segment for density calculation
            selector_buffer = (df.xatc >= (dfseg.xatc.min()-aspect*wid)) & (df.xatc <= (dfseg.xatc.max()+aspect*wid))
            dfseg_buffer = df[selector_buffer].copy()
        
            # normalize xatc to be regularly spaced and scaled by the aspect parameter
            nphot_buff = len(dfseg_buffer.xatc)
            xnorm = np.linspace(0, segment_length+2*wid*aspect, nphot_buff) / aspect
        
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
            if np.max(densities) != 0:
                densities /= np.max(densities)
            df.loc[selector_segment, 'density'] = densities
        
    return df

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