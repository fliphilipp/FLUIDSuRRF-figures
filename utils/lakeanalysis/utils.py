import os
os.environ["GDAL_DATA"] = "/home/parndt/anaconda3/envs/geo_py37/share/gdal"
os.environ["PROJ_LIB"] = "/home/parndt/anaconda3/envs/geo_py37/share/proj"
import h5py
import math
from datetime import datetime
from datetime import timedelta
from datetime import timezone
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


###########################################################################
def get_quality_summary(detection_quality, lake_quality):
    detection_quality = np.clip(detection_quality, 0, 1)
    lake_quality = np.clip(lake_quality, 0, 200)
    if (lake_quality + detection_quality) > 0:
        return np.clip((detection_quality+0.00001) * (lake_quality+0.1), 0, 100)
    else:
        return 0.0

###########################################################################
class dictobj:
    
    def __init__(self, in_dict:dict):
        assert isinstance(in_dict, dict)
        for key, val in in_dict.items():
            setattr(self, key, val)
            
    ###########################################################################        
    def plot_lake(self, verbose=False, print_mframe_info=True, closefig=False, set_yl='auto'):

        import matplotlib
        from matplotlib.patches import Rectangle
        fig, ax = plt.subplots(figsize=[9, 5], dpi=100)

        if not hasattr(self, 'depth_data'):
            print('Lake has no depth data. Skipping...')
            return
            
        dfd = self.depth_data
        surf_elev = self.surface_elevation
        below_surf = dfd.depth > 0.0
        plot_surf = np.ones_like(dfd.h_fit_surf)*surf_elev
        plot_surf[~below_surf] = np.nan
        plot_bed = np.array(dfd.h_fit_bed)
        plot_bed[~below_surf] = np.nan

        # plot the ATL03 photon data
        scatt = ax.scatter(self.photon_data.xatc, self.photon_data.h,s=5, c=self.photon_data.snr, alpha=1, 
                           edgecolors='none', cmap=cmc.lajolla, vmin=0, vmax=1)
        p_scatt = ax.scatter([-9999]*4, [-9999]*4, s=15, c=[0.0,0.25,0.75,1.0], alpha=1, edgecolors='none', cmap=cmc.lajolla, 
                             vmin=0, vmax=1, label='ATL03 photons')

        # plot the second returns from detection
        for j, prom in enumerate(self.detection_2nd_returns['prom']):
            ax.plot(self.detection_2nd_returns['xatc'][j], self.detection_2nd_returns['h'][j], 
                                    marker='o', mfc='none', mec='b', linestyle = 'None', ms=prom*5, alpha=0.5)
        p_2nd_return, = ax.plot(-9999, -9999, marker='o', mfc='none', mec='b', ls='None', ms=3, label='second return peaks (detection)')

        # plot surface elevation and the fit to the lake bed
        p_surf_elev, = ax.plot(dfd.xatc, plot_surf, 'g-', label='lake surface')
        p_bed_fit, = ax.plot(dfd.xatc, plot_bed, 'b-', label='lake bed fit')

        # plot the water depth on second axis (but zero aligned with the lake surface elevation 
        ax2 = ax.twinx()
        p_water_depth = ax2.scatter(dfd.xatc, dfd.depth, s=3, c=[(1, 1-x, 1-x) for x in dfd.conf], label='water depth')
        if set_yl == 'auto':
            yl1 = np.array([surf_elev - 1.5*1.333*dfd.depth.max(), surf_elev + 1.333*dfd.depth.max()])
        elif ((len(set_yl) == 2) and (isinstance(set_yl[0], (int, float)) and not isinstance(set_yl[0], bool)) and 
             (isinstance(ar[1], (int, float)) and not isinstance(ar[1], bool))):
            yl1 = np.array(set_yl)
        elif set_yl == 'none':
            yl1 = np.array([self.photon_data.h.min(), self.photon_data.h.max()])
            
        ylms = yl1
        yl2 = yl1 - surf_elev

        # plot mframe bounds
        ymin, ymax = ax.get_ylim()
        mframe_bounds_xatc = list(self.mframe_data['xatc_min']) + [self.mframe_data['xatc_max'].iloc[-1]]
        for xmframe in mframe_bounds_xatc:
            ax.plot([xmframe, xmframe], [ymin, ymax], 'k-', lw=0.5)

        # visualize which segments initially passed
        for i, passing in enumerate(self.mframe_data['lake_qual_pass']):
            mf = self.mframe_data.iloc[i]
            if passing:
                xy = (mf.xatc_min, ylms[0])
                width = mf.xatc_max - mf.xatc_min
                height = ylms[1] - ylms[0]
                rct = Rectangle(xy, width, height, ec=(1,1,1,0), fc=(0,0,1,0.1), zorder=-1000, label='major frame passed lake check')
                p_passed = ax.add_patch(rct)
            p_mfpeak, = ax.plot((mf.xatc_min,mf.xatc_max), (mf.peak,mf.peak),'k-',lw=0.5, label='major frame peak')

        # add a legend
        hdls = [p_scatt, p_surf_elev, p_bed_fit, p_water_depth, p_2nd_return, p_mfpeak, p_passed]
        ax.legend(handles=hdls, loc='lower left', fontsize=7, scatterpoints=4)

        # add the colorbar 
        divider = make_axes_locatable(ax2)
        cax = divider.append_axes('right', size='4%', pad=0.5)
        cax.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.5)
        cbar = fig.colorbar(scatt, cax=cax, orientation='vertical')
        cbar.ax.get_yaxis().set_ticks([])
        for j, lab in enumerate([0.2, 0.4, 0.6, 0.8]):
            cbar.ax.text(.5, lab, '%.1f'%lab, ha='center', va='center', fontweight='black')
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('photon density', rotation=270, fontsize=8)

        # add labels and description in title
        txt  = 'ICESat-2 Melt Lake: %s, ' % ('Greenland Ice Sheet' if self.lat>=0 else 'Antarctic Ice Sheet')
        txt += '%s Melt Season' % self.melt_season
        fig.suptitle(txt, y=0.95, fontsize=14)

        txt  = 'location: %s, %s (area: %s) | ' % (self.lat_str, self.lon_str, self.polygon_name)
        txt += 'time: %s UTC | surface elevation: %.2f m\n' % (self.date_time, self.surface_elevation)
        txt += 'RGT %s %s cycle %i | ' % (self.rgt, self.gtx.upper(), self.cycle_number)
        txt += 'beam %i (%s, %s spacecraft orientation) | ' % (self.beam_number, self.beam_strength, self.sc_orient)
        txt += 'granule ID: %s' % self.granule_id
        ax.set_title(txt, fontsize=8)

        ax.set_ylabel('elevation above geoid [m]',fontsize=8,labelpad=0)
        ax2.set_ylabel('water depth [m]',fontsize=8,labelpad=0)
        ax.tick_params(axis='x', which='major', labelsize=7)
        ax.tick_params(axis='y', which='major', labelsize=6)
        ax2.tick_params(axis='y', which='major', labelsize=7)
        
        # set limits
        ax.set_xlim((dfd.xatc.min(), dfd.xatc.max()))
        ax.set_ylim(yl1)
        ax2.set_ylim(-yl2)

        # add latitude
        #_________________________________________________________
        lx = self.photon_data.sort_values(by='xatc').iloc[[0,-1]][['xatc','lat']].reset_index(drop=True)
        _lat = np.array(lx.lat)
        _xatc = np.array(lx.xatc)
        def lat2xatc(l):
            return _xatc[0] + (l - _lat[0]) * (_xatc[1] - _xatc[0]) /(_lat[1] - _lat[0])
        def xatc2lat(x):
            return _lat[0] + (x - _xatc[0]) * (_lat[1] - _lat[0]) / (_xatc[1] - _xatc[0])
        secax = ax.secondary_xaxis(-0.065, functions=(xatc2lat, lat2xatc))
        secax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        secax.set_xlabel('latitude / along-track distance (km)',fontsize=8,labelpad=0)
        secax.tick_params(axis='both', which='major', labelsize=7)
        # secax.ticklabel_format(useOffset=False) # show actual readable latitude values
        secax.ticklabel_format(useOffset=False, style='plain')
        ax.ticklabel_format(useOffset=False, style='plain')
        
        # lx = self.photon_data.sort_values(by='lat').iloc[[0,-1]][['lat','xatc']].reset_index(drop=True)
        # lat = np.array(lx.lat)
        # xatc = np.array(lx.xatc)
        # def forward(x):
        #     return lat[0] + x * (lat[1] - lat[0]) / (xatc[1] - xatc[0])
        # def inverse(l):
        #     return xatc[0] + l * (xatc[1] - xatc[0]) / (lat[1] - lat[0])
        # secax = ax.secondary_xaxis(-0.065, functions=(forward, inverse))
        # secax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        # secax.set_xlabel('latitude / along-track distance',fontsize=8,labelpad=0)
        # secax.tick_params(axis='both', which='major', labelsize=7)
        # secax.ticklabel_format(useOffset=False) # show actual readable latitude values

        # rename x ticks
        xticklabs = ['%g km' % (xt/1000) for xt in list(ax.get_xticks())]
        ticks = ax.get_xticks()
        ax.set_xticks(ticks)
        ax.set_xticklabels(xticklabs)

        # add mframe info text
        if print_mframe_info:
            txt  = 'mframe:\n' % (mf.name%1000)
            txt += 'photons:\n' % mf.n_phot
            txt += 'peak:\n'
            txt += 'flat:\n'
            txt += 'SNR surf all:\n'
            txt += 'SNR surf above:\n'
            txt += 'SNR up:\n'
            txt += 'SNR low:\n'
            txt += '2nds:\n'
            txt += '2nds strength:\n'
            txt += '2nds number:\n'
            txt += '2nds spread:\n'
            txt += '2nds align:\n'
            txt += '2nds quality:\n'
            txt += 'pass:'
            # trans = ax.get_xaxis_transform()
            bbox = {'fc':(1,1,1,0.75), 'ec':(1,1,1,0), 'pad':1}
            ax.text(-0.005, 0.98, txt, transform=ax.transAxes, fontsize=4, ha='right', va='top', bbox=bbox)
            for i,loc in enumerate(self.mframe_data['xatc']):
                mf = self.mframe_data.iloc[i]
                txt  = '%i\n' % (mf.name%1000)
                txt += '%i\n' % mf.n_phot
                txt += '%.2f\n' % mf.peak
                txt += '%s\n' % ('Yes.' if mf.is_flat else 'No.')
                txt += '%i\n' % np.round(mf.snr_surf)
                txt += '%i\n' % np.round(mf.snr_allabove)
                txt += '%i\n' % np.round(mf.snr_upper)
                txt += '%i\n' % np.round(mf.snr_lower)
                txt += '%i%%\n' % np.round(mf.ratio_2nd_returns*100)
                txt += '%.2f\n' % mf.quality_secondreturns
                txt += '%.2f\n' % mf.length_penalty
                txt += '%.2f\n' % mf.range_penalty
                txt += '%.2f\n' % mf.alignment_penalty
                txt += '%.2f\n' % mf.quality_summary
                txt += '%s' % ('Yes.' if mf.lake_qual_pass else 'No.')
                trans = ax.get_xaxis_transform()
                
                bbox = {'fc':(1,1,1,0.75), 'ec':(1,1,1,0), 'pad':1}
                if (loc>dfd.xatc.min()) & (loc<dfd.xatc.max()):
                    ax.text(loc, 0.98, txt, transform=trans, fontsize=4,ha='center', va='top', bbox=bbox)

        # add detection quality description
        txt  = 'LAKE QUALITY: %6.2f'%self.lake_quality
        bbox = {'fc':(1,1,1,0.75), 'ec':(1,1,1,0), 'pad':1}
        ax.text(0.99, 0.02, txt, transform=ax.transAxes, ha='right', va='bottom',fontsize=10, weight='bold', bbox=bbox)

        fig.patch.set_facecolor('white')
        fig.tight_layout()
        ax.set_ylim(yl1)
        ax2.set_ylim(-yl2)
        ax.set_xlim((dfd.xatc.min(), dfd.xatc.max()))
        ax2.set_xlim((dfd.xatc.min(), dfd.xatc.max()))
        
        self.figure = fig
        if closefig: 
                plt.close(fig)
        return fig
        
###########################################################################
# def convert_time_to_string(dt):
#     epoch = dt + datetime.datetime.timestamp(datetime.datetime(2018,1,1))
#     return datetime.datetime.fromtimestamp(epoch).strftime("%Y-%m-%dT%H:%M:%SZ")
# 
def convert_time_to_string(lake_mean_delta_time): # fixed to match UTC timezone
    # ATLAS SDP epoch is 2018-01-01:T00.00.00.000000 UTC, from ATL03 data dictionary 
    ATLAS_SDP_epoch_datetime = datetime(2018, 1, 1, tzinfo=timezone.utc)
    ATLAS_SDP_epoch_timestamp = datetime.timestamp(ATLAS_SDP_epoch_datetime)
    lake_mean_timestamp = ATLAS_SDP_epoch_timestamp + lake_mean_delta_time
    lake_mean_datetime = datetime.fromtimestamp(lake_mean_timestamp, tz=timezone.utc)
    time_format_out = '%Y-%m-%dT%H:%M:%SZ'
    is2time = datetime.strftime(lake_mean_datetime, time_format_out)
    return is2time

###########################################################################
def read_melt_lake_h5(fn):
    
    lakedict = {}
    with h5py.File(fn, 'r') as f:

        # metadata
        for key in f['properties'].keys(): 
            lakedict[key] = f['properties'][key][()]
            if f['properties'][key].dtype == object:
                lakedict[key] = lakedict[key].decode('utf-8')

        # imagery info, if available
        if 'imagery_info' in f.keys():
            lakedict['imagery_info'] = {}
            for key in f['imagery_info'].keys(): 
                lakedict['imagery_info'][key] = f['imagery_info'][key][()]
                if f['imagery_info'][key].dtype == object:
                    lakedict['imagery_info'][key] = lakedict['imagery_info'][key].decode('utf-8')

        # depth data
        depth_data_dict = {}
        for key in f['depth_data'].keys():
            depth_data_dict[key] = f['depth_data'][key][()]
        lakedict['depth_data'] = pd.DataFrame(depth_data_dict)

        # photon data
        photon_data_dict = {}
        for key in f['photon_data'].keys():
            photon_data_dict[key] = f['photon_data'][key][()]
        lakedict['photon_data'] = pd.DataFrame(photon_data_dict)

        # mframe data
        mframe_data_dict = {}
        for key in f['mframe_data'].keys():
            mframe_data_dict[key] = f['mframe_data'][key][()]
        lakedict['mframe_data'] = pd.DataFrame(mframe_data_dict).set_index('mframe')    

        # second returns data
        det_2nds_dict = {}
        for key in f['detection_2nd_returns'].keys():
            det_2nds_dict[key] = f['detection_2nd_returns'][key][()]
        lakedict['detection_2nd_returns'] = det_2nds_dict

        # quality assessment data
        qual_dict = {}
        for key in f['detection_quality_info'].keys():
            qual_dict[key] = f['detection_quality_info'][key][()]
        lakedict['detection_quality_info'] = qual_dict
        
        # re-nest the lake extent segments
        def re_nest_extent(x): return [[x[i], x[i+1]] for i in np.arange(0,len(x),2)]
        lakedict['surface_extent_detection'] = re_nest_extent(lakedict['surface_extent_detection'])
        lakedict['lat_surface_extent_detection'] = re_nest_extent(lakedict['lat_surface_extent_detection'])
        
        lakedict['date_time'] = convert_time_to_string(lakedict['mframe_data']['dt'].mean())
        
        return lakedict