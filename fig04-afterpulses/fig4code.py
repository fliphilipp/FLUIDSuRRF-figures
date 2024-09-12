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
from IPython.display import Image, display
from cmcrameri import cm as cmc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Rectangle
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from matplotlib.patches import Polygon
from matplotlib.ticker import AutoMinorLocator

import sys
sys.path.append('../utils/')
from lakeanalysis.utils import dictobj, convert_time_to_string, read_melt_lake_h5
from lakeanalysis.nsidc import download_is2, read_atl03

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

def plot_saturation_counts(file_name_saturation_counts, axes=None, beam_select='all', xlim_ax=(1e-5,20), set_title=False):

    if not axes:
        fig = plt.figure(figsize=[16, 6], dpi=50)
        gs = fig.add_gridspec(30, 1)
        ax = fig.add_subplot(gs[:15, 0])
        ax1 = fig.add_subplot(gs[15:25, 0], sharex=ax)
        ax2 = fig.add_subplot(gs[25:, 0], sharex=ax)
    else:
        ax = axes[0]
        ax1 = axes[1]
        ax2 = axes[2]
        
    thresh_upper = 0.35
    thresh_mid = -1.97
    thresh_lower = -7
    thresh_tail = -60.0
    
    peak_target_elevs = [0.0, -0.55, -0.92, -1.50, -1.85, -2.47, -4.26, -6.52]
    peak_elevs_dict = {
        'all':    [0.0,  -0.55,  -0.92, np.nan,  -1.50,  -1.85,  -2.47,  -4.26,  -6.52, -29.31],
        'strong': [0.0,  -0.55,  -0.92, np.nan,  -1.48, np.nan,  -2.44,  -4.23,  -6.46, -29.05],
        'weak':   [0.0,  -0.53, -0.946, np.nan,  -1.50,  -1.85,  -2.49,  -4.42, -6.605, -29.34],
        '1':      [0.0,  -0.54,  -0.88, np.nan,  -1.40, np.nan,  -2.37,  -4.19,  -6.44, -28.68],
        '2':      [0.0,  -0.51,  -0.88, np.nan,  -1.43, np.nan,  -2.31,  -4.24,  -6.52, -30.54],
        '3':      [0.0,  -0.56,  -0.92,  -1.22,  -1.50, -1.795,  -2.47,  -4.24,  -6.46, -29.27],
        '4':      [0.0, -0.575, -0.946, np.nan,  -1.50,  -1.85,  -2.50,  -4.45,  -6.65, -29.33],
        '5':      [0.0,  -0.54,  -0.89, np.nan, np.nan, np.nan,  -2.35,  -4.21,  -6.47, -28.99],
        '6':      [0.0,  -0.55,  -0.95, np.nan, -1.484,  -1.82,  -2.38,  -4.25,  -6.53, -30.61],
    }
    peak_starts = [0.0, -0.31, -0.79, np.nan, -1.30, -1.77, -2.09, -3.95, -6.3]
    peak_ends = [0.0, -0.67, -1.06, np.nan, -1.62, -1.92, -2.87, -4.75, -6.8]
    # lower internal reflection and PMT ionization: -6.52, -29.31
    peak_labels = ['surface'] + [r'$AP_%i^{(dead)}=' % (i+1) for i in range(5)] + [r'$AP_%i^{(ir)}=' % (i+1) for i in range(3)]
    widths_pk = [0.0, 0.18, 0.15, 0.1, 0.11, 0.1, 0.35, 0.35, 0.2, 17.0]
    cols_pk = ['black','#67004F', '#87363B', '#A76B28', '#C6A114', '#E6D600', '#08007F', '#0480AA', '#00FFD4', 'green']
    lsty_pk = ['-', '-', '-', '-', '-', '-', '--', '--', '--', ':']

    df = pd.read_csv(file_name_saturation_counts)
    vals = df['smooth_%s'%beam_select]
    peak_target_elevs = peak_elevs_dict[str(beam_select)]
    
    ##############################################################    
    ax.plot(vals, df.elev_bins, 'k-', lw=1, zorder=1000)
    for i in np.arange(0,6):
        thish = peak_target_elevs[i]
        if not np.isnan(thish):
            thispeak_height = vals.iloc[np.argmin(np.abs(df.elev_bins-thish))]
            ax.plot([xlim_ax[0], thispeak_height], [thish]*2, color=cols_pk[i], ls=lsty_pk[i], zorder=100,solid_capstyle='butt')
            if i == 0:
                ax.text(1.2*xlim_ax[0], thish, 'saturated surface return', color=cols_pk[i], ha='left', va='bottom')
            else:
                ax.text(thispeak_height*1.2, thish, r'%s%.2f$ m' % (peak_labels[i],thish), color=cols_pk[i], weight='bold', va='center')
                if beam_select == 'all':
                    u = peak_starts[i]
                    l = peak_ends[i]
                else:
                    u = thish + widths_pk[i]
                    l = thish - widths_pk[i]
                ys = [u, l] + list(df.elev_bins[(df.elev_bins <= u) & (df.elev_bins >= l)]) + [u]
                xs = [1e-12, 1e-12] +  list(vals[(df.elev_bins <= u) & (df.elev_bins >= l)]) + [1e-12]
                ax.add_patch(Polygon(np.transpose(np.vstack((xs,ys))), color=cols_pk[i], alpha=0.15,zorder=50))
    
    ax.set_xlim(xlim_ax)
    ylms = (thresh_mid, thresh_upper)
    ax.set_ylim(ylms)
    ax.set_xscale('log')
    ax.axes.yaxis.grid(which='major', color='#EEEEEE', linestyle=':', linewidth=0.8, zorder=-1000)
    ax.axes.xaxis.grid(which='major', color='#EEEEEE', linewidth=0.8, zorder=-1000)
    ax.axes.xaxis.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5, zorder=-1000)

    if type(beam_select) == str:
        tit = r'\textbf{%s beams}' % beam_select
    else:
        tit = r'\textbf{beam %i}' % beam_select

    if set_title:
        ax.set_title(tit)

    ##############################################################
    ax1.plot(vals, df.elev_bins, 'k-', lw=1, zorder=1000)
    for i in np.arange(6,9):
        thish = peak_target_elevs[i]
        if not np.isnan(thish):
            thispeak_height = vals.iloc[np.argmin(np.abs(df.elev_bins-thish))]
            ax1.plot([xlim_ax[0], thispeak_height], [thish]*2, color=cols_pk[i], ls=lsty_pk[i], zorder=100,solid_capstyle='butt')
            ax1.text(thispeak_height*1.2, thish, r'%s%.2f$ m' % (peak_labels[i],thish), color=cols_pk[i], weight='bold', va='center')
            if beam_select == 'all':
                u = peak_starts[i]
                l = peak_ends[i]
            else:
                u = thish + widths_pk[i]
                l = thish - widths_pk[i]
            ys = [u, l] + list(df.elev_bins[(df.elev_bins <= u) & (df.elev_bins >= l)]) + [u]
            xs = [1e-12, 1e-12] +  list(vals[(df.elev_bins <= u) & (df.elev_bins >= l)]) + [1e-12]
            ax1.add_patch(Polygon(np.transpose(np.vstack((xs,ys))), color=cols_pk[i], alpha=0.15,zorder=50))

    ax1.set_ylabel('elevation relative to surface (m)')
    ax1.set_xlim(xlim_ax)
    ylms = (thresh_lower, thresh_mid)
    ax1.set_ylim(ylms)
    ax1.set_xscale('log')
    ax1.axes.yaxis.grid(which='major', color='#EEEEEE', linestyle=':', linewidth=0.8, zorder=-1000)
    ax1.axes.xaxis.grid(which='major', color='#EEEEEE', linewidth=0.8, zorder=-1000)
    ax1.axes.xaxis.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5, zorder=-1000)
            
    ##############################################################
    ax2.plot(vals, df.elev_bins, 'k-', lw=1, zorder=1000)
    u = -12
    l = -45
    ys = [u, l] + list(df.elev_bins[(df.elev_bins <= u) & (df.elev_bins >= l)]) + [u]
    xs = [1e-12, 1e-12] +  list(vals[(df.elev_bins <= u) & (df.elev_bins >= l)]) + [1e-12]
    ax2.add_patch(Polygon(np.transpose(np.vstack((xs,ys))), color='g', alpha=0.15,zorder=50))
    thish = peak_target_elevs[9]
    thispeak_height = vals.iloc[np.argmin(np.abs(df.elev_bins-thish))]
    ax2.plot([xlim_ax[0], thispeak_height], [thish]*2, color='g', ls=':', zorder=100,solid_capstyle='butt')
    ax2.text(thispeak_height*1.2, thish, r'$AP^{(ion)}=%.2f$ m' % thish, color='g', weight='bold', va='center')
    
    ax2.set_xlabel('relative photon counts')
    ylms = (thresh_tail, thresh_lower)
    ax2.set_ylim(ylms)
    ax2.yaxis.set_minor_locator(AutoMinorLocator(20))
    ax2.axes.yaxis.grid(which='both', color='#EEEEEE', linestyle=':', linewidth=0.8, zorder=-1000)
    ax2.axes.xaxis.grid(which='major', color='#EEEEEE', linewidth=0.8, zorder=-1000)
    ax2.axes.xaxis.grid(which='minor', color='#EEEEEE', linestyle=':', linewidth=0.5, zorder=-1000)

def plot_afterpulse_removal(fn, fig=None, axes=None):
    if not axes:
        fig = plt.figure(figsize=[16, 6], dpi=50)
        gs = fig.add_gridspec(30, 1)
        axs = []
        for i in range(5):
            if i == 0:
                axs.append(fig.add_subplot(gs[i*6:(i+1)*6, 0]))
            else:
                axs.append(fig.add_subplot(gs[i*6:(i+1)*6, 0],sharex=axs[-1]))
            if i < 4:
                plt.setp(axs[-1].get_xticklabels(), visible=False)
    else:
        axs = axes
        
    lk = dictobj(read_melt_lake_h5(fn))
    df = lk.photon_data
    df['pulseid'] = df.apply(lambda row: 1000*row.mframe+row.ph_id_pulse, axis=1)
    df_mframe = lk.mframe_data
    xatcmin = df.xatc.min()
    df.xatc -= xatcmin
    surf_elev = lk.surface_elevation
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
    
    beams_available, ancillary, dfs = read_atl03(fn_atl03, geoid_h=True, gtxs_to_read=gtx)
    df03 = dfs[gtx]
    
    ylms = np.array((-1.7*lk.max_depth, 0.4*lk.max_depth)) + surf_elev
    xlms = (df.lat.min(), df.lat.max())
    
    txt_y = 0.89
    boxprops = dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='none', pad=0.1)
    sz = 0.5
    
    ax = axs[-5]
    ax.scatter(df.lat, df.h, s=sz, c='k', alpha=1)
    ax.text(0.5, txt_y, 'all ATL03 photons', transform=ax.transAxes, bbox=boxprops, ha='center')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.05)
    cax.axis('off')
    
    ax = axs[-4]
    dfs3 = df03[df03.qual == 0]
    ax.scatter(dfs3.lat, dfs3.h, s=sz, c='k')
    ax.text(0.5, txt_y, r'ATL03 \texttt{quality_ph} filter', transform=ax.transAxes, bbox=boxprops, ha='center')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.05)
    cax.axis('off')
    
    ax = axs[-3]
    df['remove_afterpulse'] = df.prob_afterpulse > np.random.uniform(0,1,len(df))
    dfp = df[~df.remove_afterpulse]
    scatt = ax.scatter(dfp.lat, dfp.h, s=sz, c='k', alpha=1)
    ax.text(0.5, txt_y, r'FLUID afterpulse removal', transform=ax.transAxes, bbox=boxprops, ha='center')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.05)
    cax.axis('off')
    
    ax = axs[-2]
    df_sat = df.sort_values(by='sat_ratio')
    scatt = ax.scatter(df_sat.lat, df_sat.h, s=sz, alpha=1, c=df_sat.sat_ratio, cmap=cmc.batlow_r, vmin=0, vmax=5)
    dfpulse = df.groupby('pulseid').median()
    dfpulse = dfpulse[dfpulse.sat_ratio > 1.0]
    satelevs = ax.scatter(dfpulse.lat, dfpulse.sat_elev, s=0.5, c='r', alpha=0.3, label='elevations of saturated returns')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.05)
    cbar = fig.colorbar(scatt, cax=cax, orientation='vertical')
    cbar.set_label('saturation ratio')
    ax.text(0.5, txt_y, r'FLUID pulse saturation estimate', transform=ax.transAxes, bbox=boxprops, ha='center')
    satelevs = ax.scatter(-9999, -9999, s=1, c='r', alpha=1, label='elevations of saturated returns')
    ax.legend(handles=[satelevs], loc='lower left', fontsize=13, scatterpoints=4)
    cax.set_yticks([1,2,3,4])
    
    ax = axs[-1]
    df_prob = df.sort_values(by='prob_afterpulse')
    scatt = ax.scatter(df_prob.lat, df_prob.h, s=sz, alpha=1, c=df_prob.prob_afterpulse, cmap=cmc.batlow_r, vmin=0, vmax=1)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.05)
    cbar = fig.colorbar(scatt, cax=cax, orientation='vertical')
    cbar.set_label('afterpulse probability')
    cax.set_yticks([0.25, 0.5, 0.75])
    ax.text(0.5, txt_y, r'FLUID afterpulse likelihood estimate', transform=ax.transAxes, bbox=boxprops, ha='center')
    ymin, ymax = ax.get_ylim()
    mframe_bounds_xatc = np.array(list(df_mframe['xatc_min']) + [df_mframe['xatc_max'].iloc[-1]]) - xatcmin
    for xmframe in mframe_bounds_xatc:
        ax.plot([xmframe, xmframe], [ymin, ymax], 'k-', lw=0.5)
    
    for ax in axs:
        ax.set_xlim(xlms)
        ax.set_ylim(ylms)
        ax.set_ylabel('elevation (m a.s.l.)')
        
    axs[-1].ticklabel_format(useOffset=False, style='plain')
    axs[-1].tick_params(axis='x', labelsize=16)
    
    df['x10'] = np.round(df.xatc, -1)
    gt = df.groupby(by='x10')[['lat', 'lon']].median().sort_values(by='x10').reset_index()
    
    # flip x-axis if track is descending, to make along-track distance go from left to right
    if gt.lat.iloc[0] > gt.lat.iloc[-1]:
        for axx in axs[-5:]:
            axx.set_xlim(np.flip(np.array(xlms)))

    # add along-track distance
    lx = gt.sort_values(by='x10').iloc[[0,-1]][['x10','lat']].reset_index(drop=True)
    _lat = np.array(lx.lat)
    _xatc = np.array(lx.x10) / 1e3
    def lat2xatc(l):
        return _xatc[0] + (l - _lat[0]) * (_xatc[1] - _xatc[0]) /(_lat[1] - _lat[0])
    def xatc2lat(x):
        return _lat[0] + (x - _xatc[0]) * (_lat[1] - _lat[0]) / (_xatc[1] - _xatc[0])
    secax = ax.secondary_xaxis(-0.17, functions=(lat2xatc, xatc2lat))
    secax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    secax.set_xlabel('latitude / along-track distance (km)',labelpad=0)
    secax.tick_params(axis='both', which='major')
    secax.ticklabel_format(useOffset=False, style='plain')
    