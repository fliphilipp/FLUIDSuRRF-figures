import os
# os.environ["GDAL_DATA"] = "/Users/parndt/anaconda3/envs/eeicelakes-env/share/gdal"
# os.environ["PROJ_LIB"] = "/Users/parndt/anaconda3/envs/eeicelakes-env/share/proj"
# os.environ["PROJ_DATA"] = "/Users/parndt/anaconda3/envs/eeicelakes-env/share/proj"
import ee
import h5py
import pickle
import math
import requests
import traceback
import shapely
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import geopandas as gpd
from datetime import datetime 
from datetime import timedelta
from datetime import timezone
import time
import rasterio as rio
from rasterio import plot as rioplot
from rasterio import warp
from IPython.display import Image, display
from cmcrameri import cm as cmc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from sklearn.neighbors import KernelDensity
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from scipy.signal import find_peaks
from scipy.stats import binned_statistic

import sys
sys.path.append('../utils/')
from lakeanalysis.utils import dictobj, convert_time_to_string, read_melt_lake_h5
from lakeanalysis.nsidc import download_is2, read_atl03

def run_bathymetry_checks(file):
    lk = dictobj(read_melt_lake_h5(file))
    dfm = lk.mframe_data
    df = lk.photon_data
    
    
    has_bathy_list = []
    elev_2ndpeaks_list = []
    prominences_list = []
    subpeaks_xatc_list = []
    qualities_list = []
    
    lns = []
    for i in range(len(dfm)-1):
        ln = (dfm.iloc[i].xatc_max + dfm.iloc[i+1].xatc_min) / 2
        lns.append(ln)
    
    bounds = [df.xatc.min()] + lns + [df.xatc.max()]
    n_subsegs = lk.n_subsegs_per_mframe
    
    for i in range(len(dfm)):
        is_lake, elev_2ndpeaks, prominences, subpeaks_xatc, yl, qualities = lake_check(imf=i, xmin=bounds[i], xmax=bounds[i+1],n_subsegs=n_subsegs,
                                                                                       df=df, dfm=dfm)
        has_bathy_list.append(is_lake)
        elev_2ndpeaks_list.append(elev_2ndpeaks)
        prominences_list.append(prominences)
        subpeaks_xatc_list.append(subpeaks_xatc)
        qualities_list.append(qualities)
    elev_2ndpeaks_list = [item for row in elev_2ndpeaks_list for item in row]
    prominences_list = [item for row in prominences_list for item in row]
    subpeaks_xatc_list = [item for row in subpeaks_xatc_list for item in row]
    dfm['passes_bathymetry_check'] = has_bathy_list
    dfm[['q_%d' % j for j in np.arange(1,5)] + ['q_s']] = np.array(qualities_list)
    dfm = dfm.reset_index(drop=False)
    cols = ['mframe', 'dt', 'is_flat', 'passes_bathymetry_check', 'lat', 'lon', 'n_phot', 'peak', 'xatc', 'xatc_max', 'xatc_min', 
            'q_1', 'q_2', 'q_3', 'q_4', 'q_s']
    dfm = dfm[cols]
    
    try:
        with h5py.File(file, 'r+') as f:
            comp="gzip"
            if 'mframe_data' in f.keys():
                del f['mframe_data']
            mfdat = f.create_group('mframe_data')
            for col in cols:
                mfdat.create_dataset(col, data=dfm[col], compression=comp)
            if 'detection_2nd_returns' in f.keys():
                del f['detection_2nd_returns']
            scnds = f.create_group('detection_2nd_returns')
            scnds.create_dataset('h', data=np.array(elev_2ndpeaks_list), compression=comp)
            scnds.create_dataset('xatc', data=np.array(subpeaks_xatc_list), compression=comp)
            scnds.create_dataset('prom', data=np.array(prominences_list), compression=comp)
            
    except:
        print('WARNING: Bathymetry check results could not be written to file!')
        traceback.print_exc()

    return has_bathy_list, {'xatc': subpeaks_xatc_list, 'h': elev_2ndpeaks_list, 'prom': prominences_list}

def lake_check(imf, xmin, xmax, n_subsegs, df, dfm, axs=None, ax_color='k', plot=False, yl=None):

    mf = dfm.iloc[imf].name
    dfseg = df[df.mframe==mf].copy()
    # dfseg.xatc -= xmin

    bin_height_coarse=0.2
    bin_height_fine=0.01
    smoothing_histogram=0.1
    buffer=2.0
    width_surf=0.12
    width_buff=0.35
    rel_dens_upper_thresh=5
    rel_dens_lower_thresh=2
    min_phot=30
    min_snr_surface=10
    min_snr_vs_all_above=100
    
    promininece_threshold = 0.1
    bins_coarse1 = np.arange(start=dfseg.h.min(), stop=dfseg.h.max(), step=bin_height_coarse)
    hist_mid1 = bins_coarse1[:-1] + 0.5 * bin_height_coarse
    broad_hist = np.array(pd.Series(np.histogram(dfseg.h, bins=bins_coarse1)[0]).rolling(window=10, center=True, min_periods=1, win_type='gaussian').mean(std=2.5))
    broad_hist /= np.max(broad_hist)
    peaks, peak_props = find_peaks(broad_hist, height=promininece_threshold, distance=1.0, prominence=promininece_threshold)
    peak_hs = hist_mid1[peaks]
    if len(peaks) > 1:
        peak_proms = peak_props['prominences']
        idx_2highest = np.flip(np.argsort(peak_proms))[:2]
        pks_h = np.sort(peak_hs[idx_2highest])
        peak_loc1 = np.max(pks_h)
    else:
        peak_loc1 = hist_mid1[np.argmax(broad_hist)]
    
    # decrease bin width and find finer peak
    bins_coarse2 = np.arange(start=peak_loc1-buffer, stop=peak_loc1+buffer, step=bin_height_fine)
    hist_mid2 = bins_coarse2[:-1] + 0.5 * bin_height_fine
    hist = np.histogram(dfseg.h, bins=bins_coarse2)
    window_size = int(smoothing_histogram/bin_height_fine)
    hist_vals = hist[0] / np.max(hist[0])
    hist_vals_smoothed = np.array(pd.Series(hist_vals).rolling(window=window_size*3, center=True, min_periods=1, win_type='gaussian').mean(std=window_size/2))
    hist_vals_smoothed /= np.max(hist_vals_smoothed)
    peak_loc2 = hist_mid2[np.argmax(hist_vals_smoothed)]

    subsegs = np.linspace(xmin, xmax, n_subsegs+1)
    subsegwidth = subsegs[1] - subsegs[0]
    nphot = len(dfseg)
    bin_height_snr = 0.1
    bin_height_counts = 0.01
    smoothing_length = 0.5
    smoothing_length_counts = 1.0
    buffer=4.0
    dh_signal=0.3
    window_size_sub = int(smoothing_length/bin_height_snr)
    window_size_sub_counts = int(smoothing_length_counts/bin_height_counts)

    n_2nd_returns = 0
    prominences = []
    elev_2ndpeaks = []
    subpeaks_xatc = []

    hist_smooth = []
    mids_list = []
    just_counts = []
    just_snr = []
    for subsegstart in subsegs[:-1]:

        subsegend = subsegstart + subsegwidth
        selector_subseg = ((dfseg.xatc > subsegstart) & (dfseg.xatc < subsegend))
        dfsubseg = dfseg[selector_subseg].copy()

        # ---> if the pulses are highly saturated don't check for peaks lower than 13 meters depths 
        # (then photomultiplier tube ionization effects become a problem)
        # this is a bit of a dirty fix, but for lake detection it's probably better to throw out some highly
        # saturated data, and almost all lakes that actually have a signal deeper than 13m will most likely
        # have a very strong signal near their edges, so they should still be detected
        avg_saturation = np.nanmean(dfsubseg.sat_ratio)
        maxdepth_2nd_return = 50.0 if avg_saturation < 3.5 else 13.0

        # avoid looking for peaks when there's no / very little data
        if len(dfsubseg > 5):

            # get the median of the snr values in each bin
            bins_subseg_snr = np.arange(start=np.max((dfsubseg.h.min()-3.0,peak_loc2-maxdepth_2nd_return)), stop=peak_loc2+2*buffer, step=bin_height_snr)
            mid_subseg_snr = bins_subseg_snr[:-1] + 0.5 * bin_height_snr
            bins_subseg_counts = np.arange(start=np.max((dfsubseg.h.min()-3.0,peak_loc2-maxdepth_2nd_return)), stop=peak_loc2+2*buffer, step=bin_height_counts)
            mid_subseg_counts = bins_subseg_counts[:-1] + 0.5 * bin_height_counts
            try:
                snrstats = binned_statistic(dfsubseg.h, dfsubseg.snr, statistic='median', bins=bins_subseg_snr)
            except ValueError:  #raised if empty
                pass
            snr_median = snrstats[0]
            snr_median[np.isnan(snr_median)] = 0
            snr_vals_smoothed = np.array(pd.Series(snr_median).rolling(window=window_size_sub*3,
                                        center=True, min_periods=1, win_type='gaussian').mean(std=window_size_sub/2))
            if len(snr_vals_smoothed) < 1:
                break
            if np.max(snr_vals_smoothed) == 0:
                break
                
            snr_vals_smoothed /= np.nanmax(snr_vals_smoothed)

            # take histogram binning values into account, but clip surface peak to second highest peak height
            subhist, subhist_edges = np.histogram(dfsubseg.h, bins=bins_subseg_counts)
            subhist_smoothed = np.array(pd.Series(subhist).rolling(window=window_size_sub_counts*3, 
                                center=True, min_periods=1, win_type='gaussian').mean(std=window_size_sub_counts/2))

            # if can't find two peaks, clip to the max of the histogram that's smoothed after removing surface photons
            # get the histogram without the surface counts set to zero
            subhist_nosurface = subhist.copy()
            subhist_nosurface[(mid_subseg_counts < (peak_loc2+dh_signal)) & (mid_subseg_counts > (peak_loc2-dh_signal))] = 0
            subhist_nosurface_smoothed = np.array(pd.Series(subhist_nosurface).rolling(window=window_size_sub_counts*3, 
                                         center=True, min_periods=1, win_type='gaussian').mean(std=window_size_sub_counts/2))
            if len(subhist_nosurface_smoothed) < 1:
                break
            if np.max(subhist_nosurface_smoothed) == 0:
                break
            subhist_max = subhist_nosurface_smoothed.max()
            subhist_smoothed = np.clip(subhist_smoothed, 0, subhist_max)
                
            if np.max(subhist_smoothed) == 0:
                break
            subhist_smoothed /= np.max(subhist_smoothed)

            # combine histogram and snr values to find peaks
            snr_vals_smoothed = np.interp(mid_subseg_counts, mid_subseg_snr, snr_vals_smoothed)
            
            snr_hist_smoothed = subhist_smoothed * snr_vals_smoothed
            
            hist_smooth.append(snr_hist_smoothed)
            mids_list.append(mid_subseg_counts)
            just_snr.append(snr_vals_smoothed)
            just_counts.append(subhist_smoothed)
            
            peaks, peak_props = find_peaks(snr_hist_smoothed, height=0.1, distance=int(0.5/bin_height_snr), prominence=0.1)

            if len(peaks) >= 2: 
                has_surf_peak = np.min(np.abs(peak_loc2 - mid_subseg_counts[peaks])) < 0.4
                if has_surf_peak: 
                    idx_surfpeak = np.argmin(np.abs(peak_loc2 - mid_subseg_counts[peaks]))
                    peak_props['prominences'][idx_surfpeak] = 0

                    # classify as second peak only if prominence is larger than $(prominence_threshold)
                    prominence_secondpeak = np.max(peak_props['prominences'])
                    prominence_threshold = 0.1
                    if prominence_secondpeak > prominence_threshold:

                        idx_2ndreturn = np.argmax(peak_props['prominences'])
                        secondpeak_h = mid_subseg_counts[peaks[idx_2ndreturn]]

                        # classify as second peak only if elevation is 0.6m lower than main peak (surface) 
                        # and higher than 50m below surface
                        if (secondpeak_h < (peak_loc2-0.5)) & (secondpeak_h > (peak_loc2-50.0)):
                            secondpeak_xtac = subsegstart + subsegwidth/2
                            n_2nd_returns += 1
                            prominences.append(prominence_secondpeak)
                            elev_2ndpeaks.append(secondpeak_h)
                            subpeaks_xatc.append(secondpeak_xtac)

    # keep only second returns that are 3 m or closer to the next one on either side 
    # (helps filter out random noise, but might in rare cases suppress a signal)
    maxdiff = 20.0
    if len(elev_2ndpeaks) > 0:
        if len(elev_2ndpeaks) > 2: # if there's at least 3 second returns, compare elevations and remove two-sided outliers
            diffs = np.abs(np.diff(np.array(elev_2ndpeaks)))
            right_diffs = np.array(list(diffs) + [np.abs(elev_2ndpeaks[-3]-elev_2ndpeaks[-1])])
            left_diffs = np.array([np.abs(elev_2ndpeaks[2]-elev_2ndpeaks[0])] + list(diffs))
            to_keep = (right_diffs < maxdiff) | (left_diffs < maxdiff)

        # a-change
        # just consider elevation difference if there's only two, 
        # keep if only one (won't make it as a lake segment but keep for later...)
        elif len(elev_2ndpeaks) == 2:
            to_keep = [False, False]
        elif len(elev_2ndpeaks) == 1:
            to_keep = [False]

        n_2nd_returns = np.sum(to_keep)
        elev_2ndpeaks = np.array(elev_2ndpeaks)[to_keep]
        prominences = np.array(prominences)[to_keep]
        subpeaks_xatc = np.array(subpeaks_xatc)[to_keep]

    # get the second return qualities
    minqual = 0.1
    min_ratio_2nd_returns = 0.25
    quality_summary = 0.0
    range_penalty = 0.0
    alignment_penalty = 0.0
    length_penalty = 0.0
    quality_secondreturns = 0.0
    quality_pass = 'No'

    ratio_2nd_returns = len(elev_2ndpeaks) / n_subsegs
    # ________________________________________________________ 
    if (len(elev_2ndpeaks) > 2) & (ratio_2nd_returns > min_ratio_2nd_returns):
        h_range = np.max(elev_2ndpeaks) - np.min(elev_2ndpeaks)
        diffs = np.diff(elev_2ndpeaks)
        dirchange = np.abs(np.diff(np.sign(diffs))) > 1
        total_distance = 0.0
        for i,changed in enumerate(dirchange):
            if changed: total_distance += np.min((np.abs(diffs)[i], np.abs(diffs)[i+1]))
        frac_2ndreturns = len(elev_2ndpeaks) / n_subsegs
        alignment_penalty = np.clip(np.clip(h_range, 0.5*n_subsegs, None) / (total_distance + np.clip(h_range, 0.5*n_subsegs, None)), 0, 1)
        range_penalty = np.clip(1/math.log(np.clip(h_range,1.1,None),5), 0, 1)
        length_penalty = frac_2ndreturns**1.5
        quality_secondreturns = np.clip(np.mean(prominences) * ((np.clip(2*frac_2ndreturns, 1, None)-1)*2+1), 0, 1)
        quality_summary = alignment_penalty * length_penalty * range_penalty * quality_secondreturns

    if quality_summary > minqual: #& (yspread < max_yspread):
        quality_pass = 'Yes'

    is_lake = quality_summary > minqual
    if not yl:
        yl = [peak_loc2 - 12, peak_loc2 + 2]
    if plot:
        if not axs: 
            fig, axs = plt.subplots(figsize=[6, 8], dpi=100, ncols=2, sharey=True)

        subseg_cols = tuple([list(x) for x in cmc.batlow(np.linspace(0,1,n_subsegs))])

        ##############################################################################################
        ##############################################################################################
        box_alph = 0.2
        ax = axs[0]
        xl = [0, xmax-xmin]
        xatc = dfseg.xatc - xmin
        dfsnr = dfseg.sort_values(by='snr')
        scatt = ax.scatter(dfsnr.xatc-xmin, dfsnr.h, s=5, alpha=1, c=dfsnr.snr, edgecolors='none', cmap=cmc.lajolla, vmin=0, vmax=1, zorder=500)
        ipks = []
        for i,xstart in enumerate(subsegs[:-1]):
            xend = subsegs[i+1]
            ax.fill_between(np.array([xstart, xend])-xmin, yl[0], yl[1], color=subseg_cols[i], alpha=box_alph)
            is_inside_bounds = (subpeaks_xatc>xstart) & (subpeaks_xatc<xend)
            if np.sum(is_inside_bounds) == 1:
                j = np.argmax(is_inside_bounds)
                ipks.append(i)
                ax.plot(np.array([xstart, xend])-xmin, [elev_2ndpeaks[j]]*2, color=subseg_cols[i], ls='-', zorder=1000)
                ax.plot(np.array([xstart, xend])-xmin, [elev_2ndpeaks[j]]*2, color='k', ls='-', lw=0.3, zorder=1001)
        edgecols = [subseg_cols[i] for i in ipks]

        ax.scatter(np.array(subpeaks_xatc)-xmin, elev_2ndpeaks, s=np.array(prominences)*30, 
                   edgecolors='k', facecolors=edgecols, linewidth=0.5, zorder=1000)
            
        ax.plot(xl, [peak_loc2]*2, 'r-', zorder=1500, lw=0.7)
        ax.set_xlim(xl)
        ax.set_ylim(yl)
        # ax.set_ylabel('elevation (m)')
        ax.set_xlabel('along-track distance (m)')
        set_axis_color(ax, ax_color)
        
        ##############################################################################################
        ax = axs[1]
        xl = [0,1.15]
        psurf, = ax.plot(xl, [peak_loc2]*2, 'r-', lw=0.7, label='surface signal peak')
        for i,curve in enumerate(hist_smooth):
            ax.plot(curve, mids_list[i], color=subseg_cols[i], label='signal confidence\nbinned statistic',alpha=1.0, lw=0.5)
        ax.scatter(prominences, elev_2ndpeaks, s=np.array(prominences)*30, edgecolors='k', facecolors=edgecols, linewidth=0.5, zorder=1000)
        n_scatter_pts = np.min((len(elev_2ndpeaks),4))
        scatt_idx = list(np.linspace(0, len(elev_2ndpeaks)-1, n_scatter_pts).astype(np.int32))
        ppeaks = ax.scatter(np.array(prominences)[scatt_idx], np.array(elev_2ndpeaks)[scatt_idx], 
                            s=30*np.array(prominences)[scatt_idx], edgecolors='k', facecolors=np.array(edgecols)[scatt_idx], 
                            linewidth=0.5, zorder=1000, label='bathymetric peaks')
                
        ax.set_xlim(xl)
        ax.set_ylim(yl)
        ax.set_xlabel('signal confidence')
        set_axis_color(ax, ax_color)

        ##############################################################################################
        ax = axs[2]
            
        qsign = '>' if is_lake else '<'
        qpass = 'pass' if is_lake else 'fail'
        
        q1 = Patch(visible=False, label=r'$q_1 = %.2f$ (\# of peaks)' % length_penalty)
        q2 = Patch(visible=False, label=r'$q_2 = %.2f$ (prominence)' % quality_secondreturns)
        q3 = Patch(visible=False, label=r'$q_3 = %.2f$ (elev. spread)' % (range_penalty))
        q4 = Patch(visible=False, label=r'$q_4 = %.2f$ (alignment)'% alignment_penalty)
        qs = Patch(visible=False, label=r'$q_s = %.2f %s %.1f \Rightarrow$ %s' % (quality_summary,qsign,minqual,qpass))
    
        hdls = [psurf, ppeaks, q1, q2, q3, q4, qs]
        
        is_pass = 'b passes' if is_lake else 'c fails'
        
        tit = 'major frame %s\nbathymetric return check' % is_pass
        leg = ax.legend(handles=hdls, fontsize=9, loc='center', framealpha=0.3, scatterpoints=n_scatter_pts)
        frame = leg.get_frame()
        frame.set_edgecolor('none')
        style_legend_titles_by_removing_handles(leg)

        ax.set_xlim(xl)
        ax.set_ylim(yl)

    qualities = [length_penalty, quality_secondreturns, range_penalty, alignment_penalty, quality_summary]
        
    return is_lake, elev_2ndpeaks, prominences, subpeaks_xatc, yl, qualities

def set_axis_color(ax, axcolor):
    ax.spines['bottom'].set_color(axcolor)
    ax.spines['top'].set_color(axcolor) 
    ax.spines['right'].set_color(axcolor)
    ax.spines['left'].set_color(axcolor)
    ax.tick_params(axis='x', colors=axcolor)
    ax.tick_params(axis='y', colors=axcolor)
    ax.yaxis.label.set_color(axcolor)
    ax.xaxis.label.set_color(axcolor)
    ax.title.set_color(axcolor)

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
        print('    mean cloud probability:         %.1f %%' % lk.imagery_info['mean_cloud_probability'])
    print('')

def get_major_frame_level_info(dfm):
    stats = ['idx', 'passes_bathymetry_check', 'peak', 'xatc', 'q_1', 'q_2', 'q_3', 'q_4', 'q_s']
    dfm['idx'] = np.arange(len(dfm))
    dfm_print = dfm[stats].copy()
    dfm_print[stats[4:]] = np.array([[np.round(x,3) for x in dfm_print[s]] for s in stats[4:]]).transpose()
    dfm_print[stats[2:4]] = np.array([[np.round(x,1) for x in dfm_print[s]] for s in stats[2:4]]).transpose()
    return dfm_print

#####################################################################
def add_gt_to_imagery(fn, img, ax, xlm=[None, None], arrow_width=20, arrow_col='k', arrow_ls='-', line_col='r', line_width=1, 
                      arrow_label=None, line_label=None, bounds=None, passing_list=None, arrow_extend=0):
    
    lk = dictobj(read_melt_lake_h5(fn))
    df = lk.photon_data.copy()
    dfd = lk.depth_data.copy()

    if not xlm[0]:
        xlm[0] = df.xatc.min()
    if not xlm[1]:
        xlm[1] = df.xatc.max()
    if xlm[1] < 0:
        xlm[1] = df.xatc.max() + xlm[1]
    df = df[(df.xatc >= xlm[0]) & (df.xatc <= xlm[1])].reset_index(drop=True).copy()
    x_off = np.min(df.xatc)
    df.xatc -= x_off
    dfd.xatc -= x_off
                          
    # df['x10'] = np.round(df.xatc, -1)
    df['x10'] = np.round(df.xatc)
    gt = df.groupby(by='x10')[['lat', 'lon']].median().reset_index()
    ximg, yimg = warp.transform(src_crs='epsg:4326', dst_crs=img.crs, xs=np.array(gt.lon), ys=np.array(gt.lat))
    gt['ximg'] = ximg
    gt['yimg'] = yimg
    if not arrow_label:
        arrow_label = '%s (%s beam)' % (lk.gtx.upper(), lk.beam_strength)
    arr_xlen = ximg[-1]-ximg[0]
    arr_ylen = yimg[-1]-yimg[0]
    arr_xstart = ximg[0]
    arr_ystart = yimg[0]
    arr_xstart -= arr_xlen*arrow_extend/2
    arr_ystart -= arr_ylen*arrow_extend/2
    arr_xlen *= (1+arrow_extend)
    arr_ylen *= (1+arrow_extend)

    arrow_gt = ax.arrow(arr_xstart, arr_ystart, arr_xlen, arr_ylen, label=arrow_label, length_includes_head=True,
                        width=arrow_width, head_width=3*arrow_width, head_length=5*arrow_width, color=arrow_col, ls=arrow_ls, lw=1)
    
    if not passing_list:
        isdepth = dfd.depth>0
        bed = dfd.h_fit_bed
        bed[~isdepth] = np.nan
        bed[(dfd.depth>2) & (dfd.conf < 0.3)] = np.nan
        surf = np.ones_like(dfd.xatc) * lk.surface_elevation
        surf[~isdepth] = np.nan
        xatc_surf = np.array(dfd.xatc)[~np.isnan(surf)]
        lon_bed = np.array(dfd.lon)
        lat_bed = np.array(dfd.lat)
        lon_bed[(np.isnan(surf)) & (np.isnan(bed))] = np.nan
        lat_bed[(np.isnan(surf)) & (np.isnan(bed))] = np.nan
        xb, yb = warp.transform(src_crs='epsg:4326', dst_crs=img.crs, xs=lon_bed, ys=lat_bed)
        if not line_label:
            line_label = 'along-track lake extent'
        line_extent, = ax.plot(xb, yb, color=line_col, lw=line_width, zorder=5000, solid_capstyle='butt', label=line_label)
        hdls = [line_extent]
    elif bounds:
        hdls = [0, 0]
        is_flat_list = []
        for i, passes in enumerate(passing_list):
            minx = bounds[i]
            maxx = bounds[i+1]
            thiscol = 'g' if passes else 'r'
            label = 'has bathymetry' if passes else 'does not have\nbathymetry'
            ls = '-' if passes else ':'
            j = 0 if passes else 1
            gtmf = gt[(gt.x10 >= minx) & (gt.x10 <= maxx)]
            ax.plot(gtmf.ximg, gtmf.yimg, color=thiscol, lw=line_width, zorder=5000, solid_capstyle='butt', label=label, ls=ls)
            hdl, = ax.plot([0,1], [0,1], color=thiscol, lw=1, zorder=5000, solid_capstyle='butt', label=label, ls=ls)
            if (i < (len(passing_list)-1)) & (len(gtmf)>0):
                ax.scatter(gtmf.ximg.iloc[-1], gtmf.yimg.iloc[-1], s=1, color='k', zorder=6000)
            hdls[j] = hdl

    return arrow_gt, hdls

def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    return mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.5*height)

#####################################################################
def get_sentinel2_cloud_collection(area_of_interest, date_time, days_buffer):

    datetime_requested = datetime.strptime(date_time, '%Y-%m-%dT%H:%M:%SZ')
    start_date = (datetime_requested - timedelta(days=days_buffer)).strftime('%Y-%m-%dT%H:%M:%SZ')
    end_date = (datetime_requested + timedelta(days=days_buffer)).strftime('%Y-%m-%dT%H:%M:%SZ')
    print('Looking for Sentinel-2 images from %s to %s' % (start_date, end_date), end=' ')

    # Import and filter S2 SR HARMONIZED
    s2_sr_collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
        .filterBounds(area_of_interest)
        .filterDate(start_date, end_date))

    # Import and filter s2cloudless.
    s2_cloudless_collection = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY')
        .filterBounds(area_of_interest)
        .filterDate(start_date, end_date))

    # Join the filtered s2cloudless collection to the SR collection by the 'system:index' property.
    cloud_collection = ee.ImageCollection(ee.Join.saveFirst('s2cloudless').apply(**{
        'primary': s2_sr_collection,
        'secondary': s2_cloudless_collection,
        'condition': ee.Filter.equals(**{
            'leftField': 'system:index',
            'rightField': 'system:index'
        })
    }))

    cloud_collection = cloud_collection.map(lambda img: img.addBands(ee.Image(img.get('s2cloudless')).select('probability')))

    def set_is2_cloudiness(img, aoi=area_of_interest):
        cloudprob = img.select(['probability']).reduceRegion(reducer=ee.Reducer.mean(), 
                                                             geometry=aoi, 
                                                             bestEffort=True, 
                                                             maxPixels=1e6)
        return img.set('ground_track_cloud_prob', cloudprob.get('probability'))

    cloud_collection = cloud_collection.map(set_is2_cloudiness)

    return cloud_collection

    
#####################################################################
def download_imagery(fn, lk, gt, imagery_filename, days_buffer=5, max_cloud_prob=15, gamma_value=1.8, buffer_factor=1.2):

    lake_mean_delta_time = lk.mframe_data.dt.mean()
    ATLAS_SDP_epoch_datetime = datetime(2018, 1, 1, tzinfo=timezone.utc) # 2018-01-01:T00.00.00.000000 UTC, from ATL03 data dictionary 
    ATLAS_SDP_epoch_timestamp = datetime.timestamp(ATLAS_SDP_epoch_datetime)
    lake_mean_timestamp = ATLAS_SDP_epoch_timestamp + lake_mean_delta_time
    lake_mean_datetime = datetime.fromtimestamp(lake_mean_timestamp, tz=timezone.utc)
    time_format_out = '%Y-%m-%dT%H:%M:%SZ'
    is2time = datetime.strftime(lake_mean_datetime, time_format_out)

    # get the bounding box
    lon_rng = gt.lon.max() - gt.lon.min()
    lat_rng = gt.lat.max() - gt.lat.min()
    fac = 0.25
    bbox = [gt.lon.min()-fac*lon_rng, gt.lat.min()-fac*lat_rng, gt.lon.max()+fac*lon_rng, gt.lat.max()+fac*lat_rng]
    poly = [(bbox[x[0]], bbox[x[1]]) for x in [(0,1), (2,1), (2,3), (0,3), (0,1)]]
    roi = ee.Geometry.Polygon(poly)

    # get the earth engine collection
    collection_size = 0
    if days_buffer > 200:
        days_buffer = 200
    increment_days = days_buffer
    while (collection_size<5) & (days_buffer <= 200):
    
        collection = get_sentinel2_cloud_collection(area_of_interest=roi, date_time=is2time, days_buffer=days_buffer)
    
        # filter collection to only images that are (mostly) cloud-free along the ICESat-2 ground track
        cloudfree_collection = collection.filter(ee.Filter.lt('ground_track_cloud_prob', max_cloud_prob))
        
        collection_size = cloudfree_collection.size().getInfo()
        if collection_size == 1: 
            print('--> there is %i cloud-free image.' % collection_size)
        elif collection_size > 1: 
            print('--> there are %i cloud-free images.' % collection_size)
        else:
            print('--> there are not enough cloud-free images: widening date range...')
        days_buffer += increment_days
    
        # get the time difference between ICESat-2 and Sentinel-2 and sort by it 
        is2time = lk.date_time
        def set_time_difference(img, is2time=is2time):
            timediff = ee.Date(is2time).difference(img.get('system:time_start'), 'second').abs()
            return img.set('timediff', timediff)
        cloudfree_collection = cloudfree_collection.map(set_time_difference).sort('timediff')

    # create a region around the ground track over which to download data
    lon_center = gt.lon.mean()
    lat_center = gt.lat.mean()
    gt_length = gt.x10.max() - gt.x10.min()
    point_of_interest = ee.Geometry.Point(lon_center, lat_center)
    region_of_interest = point_of_interest.buffer(gt_length*0.5*buffer_factor)

    if collection_size > 0:
        # select the first image, and turn the colleciton into an 8-bit RGB for download
        selectedImage = cloudfree_collection.first()
        mosaic = cloudfree_collection.sort('timediff', False).mosaic()
        rgb = mosaic.select('B4', 'B3', 'B2')
        rgb = rgb.unitScale(0, 10000).clamp(0.0, 1.0)
        rgb_gamma = rgb.pow(1/gamma_value)
        rgb8bit= rgb_gamma.multiply(255).uint8()
        
        # from the selected image get some stats: product id, cloud probability and time difference from icesat-2
        prod_id = selectedImage.get('PRODUCT_ID').getInfo()
        cld_prb = selectedImage.get('ground_track_cloud_prob').getInfo()
        s2datetime = datetime.fromtimestamp(selectedImage.get('system:time_start').getInfo()/1e3)
        s2datestr = datetime.strftime(s2datetime, '%Y-%b-%d')
        s2time = datetime.strftime(s2datetime, time_format_out)
        is2datetime = datetime.strptime(is2time, '%Y-%m-%dT%H:%M:%SZ')
        timediff = s2datetime - is2datetime
        days_diff = timediff.days
        if days_diff == 0: diff_str = 'Same day as'
        if days_diff == 1: diff_str = '1 day after'
        if days_diff == -1: diff_str = '1 day before'
        if days_diff > 1: diff_str = '%i days after' % np.abs(days_diff)
        if days_diff < -1: diff_str = '%i days before' % np.abs(days_diff)
        
        print('--> Closest cloud-free Sentinel-2 image:')
        print('    - product_id: %s' % prod_id)
        print('    - time difference: %s' % timediff)
        print('    - mean cloud probability: %.1f' % cld_prb)
        if not imagery_filename:
            imagery_filename = 'data/' + prod_id + '.tif'

        try:
            with h5py.File(fn, 'r+') as f:
                if 'time_utc' in f['properties'].keys():
                    del f['properties/time_utc']
                dset = f.create_dataset('properties/time_utc', data=is2time)
                if 'imagery_info' in f.keys():
                    del f['imagery_info']
                props = f.create_group('imagery_info')
                props.create_dataset('product_id', data=prod_id)
                props.create_dataset('mean_cloud_probability', data=cld_prb)
                props.create_dataset('time_imagery', data=s2time)
                props.create_dataset('time_icesat2', data=is2time)
                props.create_dataset('time_diff_from_icesat2', data='%s' % timediff)
                props.create_dataset('time_diff_string', data='%s ICESat-2' % diff_str)
        except:
            print('WARNING: Imagery attributes could not be written to the associated lake file!')
            traceback.print_exc()
        
        # get the download URL and download the selected image
        success = False
        scale = 10
        tries = 0
        while (success == False) & (tries <= 7):
            try:
                downloadURL = rgb8bit.getDownloadUrl({'name': 'mySatelliteImage',
                                                          'crs': selectedImage.select('B3').projection().crs(),
                                                          'scale': scale,
                                                          'region': region_of_interest,
                                                          'filePerBand': False,
                                                          'format': 'GEO_TIFF'})
        
                response = requests.get(downloadURL)
                with open(imagery_filename, 'wb') as f:
                    f.write(response.content)
        
                print('--> Downloaded the 8-bit RGB image as %s.' % imagery_filename)
                success = True
                tries += 1
                return imagery_filename
            except:
                traceback.print_exc()
                scale *= 2
                print('-> download unsuccessful, increasing scale to %.1f...' % scale)
                success = False
                tries += 1


#####################################################################
def add_graticule(img, ax_img):
    from lakeanalysis.curve_intersect import intersection
    latlon_bbox = warp.transform(img.crs, {'init': 'epsg:4326'}, 
                                 [img.bounds[i] for i in [0,2,2,0,0]], 
                                 [img.bounds[i] for i in [1,1,3,3,1]])
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
    # xl = (img.bounds.left, img.bounds.right)
    # yl = (img.bounds.bottom, img.bounds.top)
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
    for me in meridians:
        gr_trans = warp.transform({'init': 'epsg:4326'},img.crs,me*np.ones_like(latseq),latseq)
        deglab = ' %.10g°E' % me if me >= 0 else ' %.10g°W' % -me
        intx,inty = intersection(bottomline[0], bottomline[1], gr_trans[0], gr_trans[1])
        rot = np.arctan2(gr_trans[1][-1] - gr_trans[1][0], gr_trans[0][-1] - gr_trans[0][0]) * 180 / np.pi
        if len(intx) > 0:
            intx = intx[0]
            inty = inty[0]
            ax_img.text(intx, inty, deglab, fontsize=8, color='gray',verticalalignment='top',horizontalalignment='center',
                    rotation=rot)
        thislw = gridlw
        ax_img.plot(gr_trans[0],gr_trans[1],c=gridcol,ls=gridls,lw=thislw,alpha=0.5)
    for pa in parallels:
        gr_trans = warp.transform({'init': 'epsg:4326'},img.crs,lonseq,pa*np.ones_like(lonseq))
        thislw = gridlw
        deglab = ' %.10g°N' % pa if pa >= 0 else ' %.10g°S' % -pa
        intx,inty = intersection(leftline[0], leftline[1], gr_trans[0], gr_trans[1])
        rot = np.arctan2(gr_trans[1][-1] - gr_trans[1][0], gr_trans[0][-1] - gr_trans[0][0]) * 180 / np.pi
        if len(intx) > 0:
            intx = intx[0]
            inty = inty[0]
            ax_img.text(intx, inty, deglab, fontsize=8, color='gray',verticalalignment='center',horizontalalignment='right',
                       rotation=rot)
        ax_img.plot(gr_trans[0],gr_trans[1],c=gridcol,ls=gridls,lw=thislw,alpha=0.5)
        ax_img.set_xlim(xl)
        ax_img.set_ylim(yl)


#####################################################################
def plot_imagery(fn, days_buffer=5, max_cloud_prob=15, xlm=[None, None], ylm=[None, None], gamma_value=1.8, imagery_filename=None,
                 re_download=True, ax=None, buffer_factor=1.2):
                     
    lk = dictobj(read_melt_lake_h5(fn))
    df = lk.photon_data.copy()
    if not xlm[0]:
        xlm[0] = df.xatc.min()
    if not xlm[1]:
        xlm[1] = df.xatc.max()
    if xlm[1] < 0:
        xlm[1] = df.xatc.max() + xlm[1]
    if not ylm[0]:
        ylm[0] = lk.surface_elevation-2*lk.max_depth
    if not ylm[1]:
        ylm[1] = lk.surface_elevation+lk.max_depth

    df = df[(df.xatc >= xlm[0]) & (df.xatc <= xlm[1]) & (df.h >= ylm[0]) & (df.h <= ylm[1])].reset_index(drop=True).copy()
    x_off = np.min(df.xatc)
    df.xatc -= x_off
    dfd = lk.depth_data.copy()
    dfd.xatc -= x_off

    # get the ground track
    df['x10'] = np.round(df.xatc, -1)
    gt = df.groupby(by='x10')[['lat', 'lon']].median().reset_index()
    lon_center = gt.lon.mean()
    lat_center = gt.lat.mean()

    thefile = 'none' if not imagery_filename else imagery_filename
    if ((not os.path.isfile(thefile)) or re_download) and ('modis' not in thefile):
        imagery_filename = download_imagery(fn=fn, lk=lk, gt=gt, imagery_filename=imagery_filename, days_buffer=days_buffer, 
                         max_cloud_prob=max_cloud_prob, gamma_value=gamma_value, buffer_factor=buffer_factor)
    
    try:
        myImage = rio.open(imagery_filename)
        
        # make the figure
        if not ax:
            fig, ax = plt.subplots(figsize=[6,6])
        
        rioplot.show(myImage, ax=ax)
        ax.axis('off')
        
        if not ax:
            fig.tight_layout(pad=0)
    
        return myImage, lon_center, lat_center
    except: 
        return None, lon_center, lat_center
        traceback.print_exc()

def make_plot(file, major_frames, yl1, yl2, yl3, plot_fn='plots/bathymetry_check.jpg', dot_size=1, 
              imagery_filename=None, re_download_imagery=False):

    # read in the lake data
    lk = dictobj(read_melt_lake_h5(file))
    df = lk.photon_data
    dfm = lk.mframe_data
    n_subsegs = lk.n_subsegs_per_mframe

    # figure settings
    plt.rcParams.update({
        'font.size': 10,
        'text.usetex': True,
        'font.family': 'Optima',
        'text.latex.preamble': r"\usepackage{amsmath}"
    })

    # create figure and subfigures
    fig = plt.figure(figsize=[10, 6], dpi=100)
    gs = fig.add_gridspec(ncols=10, nrows=5)
    ax0 = fig.add_subplot(gs[:3, :])
    ax0.axis('off')
    
    ax1 = ax0.inset_axes([-0.05, 0, 0.37, 1])
    ax1.axis('off')
    ax2 = ax0.inset_axes([0.35, 0, 0.63, 1])
    cax = ax0.inset_axes([0.985, 0, 0.015, 1])
    
    axs = [ax2]
    nrows = 2
    for i in np.arange(nrows):
        axs.append(fig.add_subplot(gs[3:, 5*i:5*i+2]))
        if i!=0:
            plt.setp(axs[-1].get_yticklabels(), visible=False)
        axs.append(fig.add_subplot(gs[3:, 5*i+2], sharey=axs[-1]))
        plt.setp(axs[-1].get_yticklabels(), visible=False)
        axs.append(fig.add_subplot(gs[3:, 5*i+3:5*i+5]))
        axs[-1].axis('off')
    axs.append(ax1)
    
    ###############################################################################################
    # plot the photon data for the whole lake (upper right)
    ax = axs[0]
    ylm = yl1
    dfsnr = df.sort_values(by='snr')
    scatt = ax.scatter(dfsnr.xatc, dfsnr.h, s=dot_size, alpha=1, c=dfsnr.snr, edgecolors='none', cmap=cmc.lajolla, vmin=0, vmax=1)
    cbar = plt.colorbar(scatt, cax=cax, orientation='vertical')
    cbar.set_label('photon signal confidence')

    # plot major frame boundaries as black vertical lines
    lns = []
    for i in range(len(dfm)-1):
        ln = (dfm.iloc[i].xatc_max + dfm.iloc[i+1].xatc_min) / 2
        ax.plot([ln]*2, ylm, 'k-', lw=0.7)
        lns.append(ln)
    
    bounds = [df.xatc.min()] + lns + [df.xatc.max()]
    passlist, bathypeaks = list(dfm.passes_bathymetry_check), lk.detection_2nd_returns
    # passlist, bathypeaks = run_bathymetry_checks(file) # to re-run the bathymetry checks
    
    # indicate whether major frames passed by hatching
    for i, is_lake in enumerate(passlist):
        thiscol = 'g' if is_lake else 'r'
        thishatch = '///' if is_lake else 'XXX'
        label = 'passes batymetric return check' if is_lake else 'does not pass bathymetric return check'
        ax.fill_between([bounds[i], bounds[i+1]], ylm[0], ylm[1], facecolor="none", hatch=thishatch, edgecolor=thiscol, 
                                    linewidth=1, alpha=0.3, label=label, zorder=-100)
    
    # plot the bathymetry peaks
    ax.scatter(bathypeaks['xatc'], bathypeaks['h'], s=np.array(bathypeaks['prom'])*180, edgecolors='b', facecolors='none', linewidth=0.3)

    # lazy way to create legend artists
    hdl1 = ax.fill_between([-9999, -9998], -9999, -9998, facecolor="none", hatch='///', edgecolor='g', 
                                    linewidth=1, alpha=0.3, label='major frame passes bathymetric return check', zorder=-100)
    hdl2 = ax.fill_between([-9999, -9998], -9999, -9998, facecolor="none", hatch='XXX', edgecolor='r', 
                                    linewidth=1, alpha=0.3, label='major frame does not pass bathymetric return check', zorder=-100)
    hdls = [hdl1, hdl2]
    prom100 = ax.scatter(-9999, -9999, s=1*50, edgecolors='b', facecolors='none', linewidth=0.3, label='prominence = 1.0')
    prom50 = ax.scatter(-9999, -9999, s=0.5*50, edgecolors='b', facecolors='none', linewidth=0.3, label='prominence = 0.5')
    prom20 = ax.scatter(-9999, -9999, s=0.2*50, edgecolors='b', facecolors='none', linewidth=0.3, label='prominence = 0.2')
    leg1 = ax.legend(handles=[prom100, prom50, prom20], loc='lower left',fontsize=10, title=r'\textbf{bathymetric peaks}')

    # add two legends here
    plt.setp(leg1.get_title(),fontsize=10)
    ax.add_artist(leg1)
    leg2 = ax.legend(handles=hdls, loc='upper center',fontsize=12)

    # set up axes
    ax.set_xlim((df.xatc.min(), df.xatc.max()))
    ax.set_ylim(ylm)
    ax.set_xlabel('along-track distance (m)')
    ax.set_ylabel('elevation (m)')
    
    ###############################################################################################
    # add the example of a passing major frame
    boxprops = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none', pad=0.1)
    arr_len = 0.08
    ax_color = 'g'
    imf=major_frames[0]
    # imf=5
    yl = yl2
    is_lake, h2, prom2, xatc2, yl, quals = lake_check(imf=imf, xmin=bounds[imf], xmax=bounds[imf+1], n_subsegs=n_subsegs, 
                                               df=df, dfm=dfm, axs=axs[1:4], ax_color=ax_color, plot=True, yl=yl)
    axs[0].plot([lns[i] for i in [imf-1,imf, imf, imf-1, imf-1]], [yl[i] for i in [0,0,1,1,0]], c=ax_color)
    xarr = (lns[imf]+lns[imf-1])/2
    ax.annotate(r'\textbf{c}', xy=(xarr, yl[1]), xytext=(xarr, yl[1]+(yl1[1]-yl1[0])*arr_len), ha='center', va='bottom', color='g', bbox=boxprops,
                arrowprops=dict(facecolor='g', shrink=0.01, width=1.5, headwidth=5, headlength=5, edgecolor='none'), fontsize=14)
    
    ###############################################################################################
    # add the example of a failing major frame
    ax_color = 'r'
    imf = major_frames[1]
    yl = yl3
    is_lake, h2, prom2, xatc2, yl, quals = lake_check(imf=imf, xmin=bounds[imf], xmax=bounds[imf+1], n_subsegs=n_subsegs, 
                                               df=df, dfm=dfm, axs=axs[4:7], ax_color=ax_color, plot=True, yl=yl)
    axs[0].plot([lns[i] for i in [imf-1,imf, imf, imf-1, imf-1]], [yl[i] for i in [0,0,1,1,0]], c=ax_color)
    xarr = (lns[imf]+lns[imf-1])/2
    ax.annotate(r'\textbf{d}', xy=(xarr, yl[1]), xytext=(xarr, yl[1]+(yl1[1]-yl1[0])*arr_len), ha='center', va='bottom', color='r', bbox=boxprops,
                arrowprops=dict(facecolor='r', shrink=0.01, width=1.5, headwidth=5, headlength=5, edgecolor='none'), fontsize=14)

    axs[1].set_ylabel(r'elevation (m)',color='k')

    # draw boundaries around the lower panels (each include 3 subplots)
    axpass = fig.add_subplot(gs[3:, :5])
    axpass.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)
    axpass.patch.set_facecolor('none')
    set_axis_color(axpass, 'green')
    axpass.text(0.5, 1.02, r'\textbf{major frame passes bathymetric return check}', ha='center', transform=axpass.transAxes, color='green')
    axfail = fig.add_subplot(gs[3:, 5:])
    axfail.tick_params(bottom=False,left=False,labelbottom=False,labelleft=False)
    axfail.patch.set_facecolor('none')
    set_axis_color(axfail, 'red')
    axfail.text(0.5, 1.02, r'\textbf{major frame fails bathymetric return check}', ha='center', transform=axfail.transAxes, color='red')

    #### add imagery
    img_aspect=1.2
    ax = axs[-1]
    ax.axis('off')
    img, center_lon, center_lat = plot_imagery(fn=file, days_buffer=5, max_cloud_prob=15, xlm=[None, None], ylm=[None, None], 
        gamma_value=1.0, imagery_filename=imagery_filename, re_download=re_download_imagery, ax=ax)
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    hdl_arr, hdls_ext = add_gt_to_imagery(file, img, ax, arrow_width=40, arrow_col=(0,0,0,0.5), arrow_ls='-', line_col='r', 
        arrow_label='ICESat-2\nground track', line_width=0.75, bounds=bounds, passing_list=passlist, arrow_extend=0.15)
    leg = ax.legend(handles=[hdl_arr]+hdls_ext, loc='upper right', fontsize=7, framealpha=0.7,
              handler_map={mpatches.FancyArrow : HandlerPatch(patch_func=make_legend_arrow)})
    ax.set_xlim(xl)
    ax.set_ylim(yl)

    # set proper aspect ratio for imagery
    if (img_aspect > 1): 
        h_rng = img.bounds.top - img.bounds.bottom
        cntr = (img.bounds.right + img.bounds.left) / 2
        ax.set_xlim(cntr-0.5*h_rng/img_aspect, cntr+0.5*h_rng/img_aspect)
    elif img_aspect < 1: 
        w_rng = img.bounds.right - img.bounds.left
        cntr = (img.bounds.top + img.bounds.bottom) / 2
        ax.set_ylim(cntr-0.5*w_rng*img_aspect, cntr+0.5*w_rng/img_aspect)
            
    add_graticule(img, ax)

    # add panel labels
    axs[-1].text(0.01, 0.98, r'\textbf{a)}', transform=axs[-1].transAxes, color='k', ha='left', va='top', fontsize=14)
    axs[0].text(0.01, 0.98, r'\textbf{b)}', transform=axs[0].transAxes, color='k', ha='left', va='top', fontsize=14, bbox=boxprops)
    axs[1].text(0.03, 0.98, r'\textbf{c)}', transform=axs[1].transAxes, color='g', ha='left', va='top', fontsize=14, bbox=boxprops)
    axs[4].text(0.03, 0.98, r'\textbf{d)}', transform=axs[4].transAxes, color='r', ha='left', va='top', fontsize=14, bbox=boxprops)
    
    gs.tight_layout(fig, pad=0.3, w_pad=0.2)

    return fig