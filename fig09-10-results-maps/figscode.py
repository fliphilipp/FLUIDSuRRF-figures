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

import sys
sys.path.append('../utils/')
from lakeanalysis.utils import dictobj, convert_time_to_string, read_melt_lake_h5

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
def download_imagery(fn, lk, gt, imagery_filename, days_buffer=5, max_cloud_prob=15, gamma_value=1.8, buffer_factor=1.2, imagery_shift_days=0):

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
    
        collection = get_sentinel2_cloud_collection(area_of_interest=roi, date_time=lk.date_time, days_buffer=days_buffer)
    
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
        # is2time = lk.date_time
        def set_time_difference(img, is2time=is2time, imagery_shift_days=imagery_shift_days):
            ref_time = ee.Date(is2time).advance(imagery_shift_days, 'day')
            timediff = ref_time.difference(img.get('system:time_start'), 'second').abs()
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
            imagery_filename = 'data/' + prod_id + '_' + fn.split('/')[-1].replace('.h5','') + '.tif'
            print(imagery_filename)

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
def plot_imagery(fn, days_buffer=5, max_cloud_prob=15, xlm=[None, None], ylm=[None, None], gamma_value=1.8, imagery_filename=None,
                 re_download=True, ax=None, buffer_factor=1.2, imagery_shift_days=0, increase_gtwidth=1):
                     
    lk = dictobj(read_melt_lake_h5(fn))
    df = lk.photon_data.copy()
    if not xlm[0]:
        xlm[0] = df.xatc.min()
    if not xlm[1]:
        xlm[1] = df.xatc.max()
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
                         max_cloud_prob=max_cloud_prob, gamma_value=gamma_value, buffer_factor=buffer_factor, 
                         imagery_shift_days=imagery_shift_days)
    
    try:
        myImage = rio.open(imagery_filename)
        
        # make the figure
        if not ax:
            fig, ax = plt.subplots(figsize=[6,6])
        
        rioplot.show(myImage, ax=ax)
        ax.axis('off')
    
        ximg, yimg = warp.transform(src_crs='epsg:4326', dst_crs=myImage.crs, xs=np.array(gt.lon), ys=np.array(gt.lat))
        if 'modis' in imagery_filename:
            xrng = ximg[-1] - ximg[0]
            yrng = yimg[-1] - yimg[0]
            fac = 5
            print('using saved modis image')
            ax.plot([ximg[-1]+fac*xrng,ximg[0]-fac*xrng], [yimg[-1]+fac*yrng, yimg[0]-fac*yrng], 'k:', lw=1)
            ax.annotate('', xy=(ximg[-1]+fac*xrng, yimg[-1]+fac*yrng), xytext=(ximg[0]-fac*xrng, yimg[0]-fac*yrng),
                             arrowprops=dict(width=0, lw=0, headwidth=5, headlength=5, color='k'),zorder=1000)
            ax.plot(ximg, yimg, 'r-', lw=1, zorder=5000)
        else:
            ax.annotate('', xy=(ximg[-1], yimg[-1]), xytext=(ximg[0], yimg[0]),
                             arrowprops=dict(width=0.7*increase_gtwidth, headwidth=5*increase_gtwidth, 
                                             headlength=5*increase_gtwidth, color='k'),zorder=1000)

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
            xb, yb = warp.transform(src_crs='epsg:4326', dst_crs=myImage.crs, xs=lon_bed, ys=lat_bed)
            ax.plot(xb, yb, 'r-', lw=increase_gtwidth, zorder=5000)
        
        if not ax:
            fig.tight_layout(pad=0)
    
        return myImage, lon_center, lat_center
    except: 
        return None, lon_center, lat_center
        traceback.print_exc()

                     
#####################################################################
def plotIS2(fn, ax=None, xlm=[None, None], ylm=[None,None], cmap=cmc.lapaz_r, name='ICESat-2 data',increase_linewidth=1):
    lk = dictobj(read_melt_lake_h5(fn))
    df = lk.photon_data.copy()
    dfd = lk.depth_data.copy()
    if not xlm[0]:
        xlm[0] = df.xatc.min()
    if not xlm[1]:
        xlm[1] = df.xatc.max()
    if not ylm[0]:
        ylm[0] = lk.surface_elevation-2*lk.max_depth
    if not ylm[1]:
        ylm[1] = lk.surface_elevation+1.5*lk.max_depth
    df = df[(df.xatc >= xlm[0]) & (df.xatc <= xlm[1]) & (df.h >= ylm[0]) & (df.h <= ylm[1])].reset_index(drop=True).copy()
    dfd = dfd[(dfd.xatc >= xlm[0]) & (dfd.xatc <= xlm[1]) & (dfd.h_fit_bed >= ylm[0])].reset_index(drop=True).copy()
    x_off = np.min(df.xatc)
    df.xatc -= x_off
    dfd.xatc -= x_off
    
    isdepth = dfd.depth>0
    bed = dfd.h_fit_bed
    bed[~isdepth] = np.nan
    bed[(dfd.depth>2) & (dfd.conf < 0.3)] = np.nan
    surf = np.ones_like(dfd.xatc) * lk.surface_elevation
    surf[~isdepth] = np.nan
    surf_only = surf[~np.isnan(surf)]
    bed_only = bed[(~np.isnan(surf)) & (~np.isnan(bed))]
    xatc_surf = np.array(dfd.xatc)[~np.isnan(surf)]
    xatc_bed = np.array(dfd.xatc)[(~np.isnan(surf)) & (~np.isnan(bed))]
    
    # make the figure
    if not ax:
        fig, ax = plt.subplots(figsize=[8,5])

    df['is_afterpulse']= df.prob_afterpulse > np.random.uniform(0,1,len(df))
    if not cmap:
        ax.scatter(df.xatc[~df.is_afterpulse], df.h[~df.is_afterpulse], s=1, c='k')
    else:
        ax.scatter(df.xatc, df.h, s=1, c=df.snr, cmap=cmap)
        
    # ax.scatter(dfd.xatc[isdepth], dfd.h_fit_bed[isdepth], s=4, color='r', alpha=dfd.conf[isdepth])
    # ax.plot(dfd.xatc, dfd.h_fit_bed, color='gray', lw=0.5)
    ax.plot(dfd.xatc, bed, color='r', lw=increase_linewidth)
    ax.plot(dfd.xatc, surf, color='C0', lw=increase_linewidth)

    # add the length of surface
    arr_y = lk.surface_elevation+lk.max_depth*0.25
    x_start = np.min(xatc_surf)
    x_end = np.max(xatc_surf)
    x_mid = (x_end + x_start) / 2
    len_surf_m = np.floor((x_end-x_start)/100)*100
    len_surf_km = len_surf_m/1000
    arr_x1 = x_mid - len_surf_m / 2
    arr_x2 = x_mid + len_surf_m / 2
    ax.annotate('', xy=(arr_x1, arr_y), xytext=(arr_x2, arr_y),
                         arrowprops=dict(width=0.7, headwidth=5, headlength=5, color='C0'),zorder=1000)
    ax.annotate('', xy=(arr_x2, arr_y), xytext=(arr_x1, arr_y),
                         arrowprops=dict(width=0.7, headwidth=5, headlength=5, color='C0'),zorder=1000)
    ax.text(x_mid, arr_y, r'\textbf{%.1f km}' % len_surf_km, fontsize=plt.rcParams['font.size']+2, ha='center', va='bottom', color='C0', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.2,rounding_size=0.5', lw=0))

    # add surface length based on what was determined by the confidence threshold here
    try:
        with h5py.File(fn, 'r+') as f:
            if 'len_surf_km' in f['properties'].keys():
                del f['properties/len_surf_km']
            dset = f.create_dataset('properties/len_surf_km', data=len_surf_km)
    except:
        print('WARNING: Surface length could not be written to the associated lake file!')
        traceback.print_exc()

    # add the max depth
    y_low = np.min(bed_only)
    y_up = lk.surface_elevation
    arr_x = xatc_bed[np.argmin(bed_only)]
    arr_x = xlm[0] - 0.0* (xlm[1] - xlm[0])
    y_len = y_up - y_low
    y_mid = (y_up + y_low) / 2
    arr_len = y_len
    arr_y1 = y_mid + arr_len / 2
    arr_y2 = y_mid - arr_len / 2
    ref_index = 1.336
    dep_round = np.round(y_len / ref_index, 1)
    ax.annotate('', xy=(arr_x, arr_y2), xytext=(arr_x, arr_y1),
                         arrowprops=dict(width=0.7, headwidth=5, headlength=5, color='r'),zorder=1000)
    ax.annotate('', xy=(arr_x, arr_y1), xytext=(arr_x, arr_y2),
                         arrowprops=dict(width=0.7, headwidth=5, headlength=5, color='r'),zorder=1000)
    ax.text(arr_x, y_mid, r'\textbf{%.1f m}' % dep_round, fontsize=plt.rcParams['font.size']+2, ha='right', va='center', color='r', fontweight='bold',
            bbox=dict(facecolor='white', alpha=0.8, lw=0, boxstyle='round,pad=0.2,rounding_size=0.5'), rotation=90)

    # change the maximum depth to what was determined by the confidence threshold here
    try:
        with h5py.File(fn, 'r+') as f:
            if 'max_depth' in f['properties'].keys():
                del f['properties/max_depth']
            dset = f.create_dataset('properties/max_depth', data= y_len/ref_index)
    except:
        print('WARNING: Maximum depth could not be written to the associated lake file!')
        traceback.print_exc()

    # add the title
    datestr = datetime.strftime(datetime.strptime(lk.date_time[:10],'%Y-%m-%d'), '%d %B %Y')
    if True:
        sheet = lk.ice_sheet
        region = lk.polygon_filename.split('_')[-1].replace('.geojson', '')
        if sheet == 'AIS':
            region = region + ' (%s)' % lk.polygon_filename.split('_')[-2]
        latstr = lk.lat_str[:-2] + '°' + lk.lat_str[-1]
        lonstr = lk.lon_str[:-2] + '°' + lk.lon_str[-1]
        name = '(%s, %s), %d m.a.s.l.' % (latstr, lonstr, np.round(lk.surface_elevation))
    
    ax.text(0.5, 0.87, '%s' % name, fontsize=plt.rcParams['font.size'], ha='center', va='top', transform=ax.transAxes,
           bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.2,rounding_size=0.5', lw=0), fontweight='bold')
    ax.text(0.5, 0.89, r'\textbf{%s}' % datestr, fontsize=plt.rcParams['font.size']+2, ha='center', va='bottom', transform=ax.transAxes,
           bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.2,rounding_size=0.5', lw=0))
    # print(name, datestr, 'quality:', lk.lake_quality)
    ax.text(0.988, 0.015, r'quality: \textbf{%.1f}' % lk.lake_quality, fontsize=plt.rcParams['font.size'], ha='right', va='bottom', transform=ax.transAxes,
           bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.2,rounding_size=0.5', lw=0))
    ax.set_xlim(xlm)
    ax.set_ylim(ylm)
    ax.axis('off')

    
#####################################################################
def plot_IS2_imagery(fn, axes=None, xlm=[None,None], ylm=[None,None], cmap=None, days_buffer=5, max_cloud_prob=15, 
                     gamma_value=1.3, imagery_filename=None, re_download=True, img_aspect=3/2, name='ICESat-2 data',
                     return_fig=False, imagery_shift_days=0.0, increase_linewidth=1, increase_gtwidth=1):

    if not axes:
        fig = plt.figure(figsize=[12,6], dpi=80)
        gs = fig.add_gridspec(1,3)
        axp = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1:])]
    else:
        axp = axes
        
    ax = axp[1]
    plotIS2(fn=fn, ax=ax, xlm=xlm, ylm=ylm, cmap=cmap, name=name, increase_linewidth=increase_linewidth)
    
    ax = axp[0]
    img, center_lon, center_lat = plot_imagery(fn=fn, days_buffer=days_buffer, max_cloud_prob=max_cloud_prob, xlm=xlm, ylm=ylm, 
        gamma_value=gamma_value, imagery_filename=imagery_filename, re_download=re_download, ax=ax, imagery_shift_days=imagery_shift_days,
        increase_gtwidth=increase_gtwidth)
        
    if img:        
        if imagery_filename:
            if 'modis' in imagery_filename:
                center_x, center_y = warp.transform(src_crs='epsg:4326', dst_crs=img.crs, xs=[center_lon], ys=[center_lat])
                center_x = center_x[0]
                center_y = center_y[0]
                rng = 75000
                if img_aspect > 1:
                    ax.set_xlim(center_x - 0.5*rng/img_aspect, center_x + 0.5*rng/img_aspect)
                    ax.set_ylim(center_y - 0.5*rng, center_y + 0.5*rng)
                if img_aspect < 1:
                    ax.set_xlim(center_x - 0.5*rng, center_x + 0.5*rng)
                    ax.set_ylim(center_y - 0.5*rng*img_aspect, center_y + 0.5*rng*img_aspect)
                
        if (img_aspect > 1): 
            h_rng = img.bounds.top - img.bounds.bottom
            cntr = (img.bounds.right + img.bounds.left) / 2
            ax.set_xlim(cntr-0.5*h_rng/img_aspect, cntr+0.5*h_rng/img_aspect)
        elif img_aspect < 1: 
            w_rng = img.bounds.right - img.bounds.left
            cntr = (img.bounds.top + img.bounds.bottom) / 2
            ax.set_ylim(cntr-0.5*w_rng*img_aspect, cntr+0.5*w_rng/img_aspect)
            
    
    if not axes:
        fig.tight_layout(pad=1, h_pad=0, w_pad=0)
        if not name:
            name = 'zzz' + lk.polygon_filename.split('_')[-1].replace('.geojson', '')
        outname = 'figplots/' + name.replace(' ', '') + fn[fn.rfind('/')+1:].replace('.h5','.jpg')
        fig.savefig(outname, dpi=300)

    if return_fig:
        plt.close(fig)
        return center_lon, center_lat, fig
    else:
        return center_lon, center_lat

#####################################################################
def plot_coords(coords, ax, crs_dst, crs_src='EPSG:4326', text=None, color='b', ms=10, fs=18, annot_loc={}):
    coords_trans = warp.transform(src_crs=crs_src, dst_crs=crs_dst, xs=[coords[0]], ys=[coords[1]])
    x = coords_trans[0][0]
    y = coords_trans[1][0]
    if text:
        text = r'\textbf{%s}' % text
    if not text:
        ax.scatter(x, y, coords_trans[1][0], s=ms, color=color)
    elif ('x' not in annot_loc.keys()) and ('y' not in annot_loc.keys()):
        ax.text(x, y, text, fontsize=fs, color='white', ha='center', va='center', fontweight='bold',
                bbox=dict(facecolor=color, alpha=1, boxstyle='round,pad=0.3,rounding_size=0.5', lw=0))
    else:
        ax.annotate(' ', xy=(x,y), xytext=(annot_loc['x'], annot_loc['y']),
                    ha='center',va='center', arrowprops=dict(width=1, headwidth=5, headlength=5, color=color),zorder=1000)
        ax.text(annot_loc['x'], annot_loc['y'], text, fontsize=fs, color='white', ha='center', va='center',
                bbox=dict(facecolor=color, alpha=1, boxstyle='round,pad=0.3,rounding_size=0.5', lw=0), zorder=2000, fontweight='bold')
        
def add_letter(ax, text, fs=16, col='b'):
    ax.text(0.09,0.95,r'\textbf{%s}'%text,color='w',fontsize=fs,ha='left', va='top', fontweight='bold',
              bbox=dict(fc=col, boxstyle='round,pad=0.3,rounding_size=0.5', lw=0), transform=ax.transAxes)

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