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
from datetime import timezone
import rasterio as rio
from rasterio import plot as rioplot
from rasterio import warp
import matplotlib
import matplotlib.pylab as plt
from matplotlib.legend_handler import HandlerPatch
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from cmcrameri import cm as cmc
from mpl_toolkits.axes_grid1 import make_axes_locatable
from IPython.display import Image, display

import sys
sys.path.append('../utils/')
from lakeanalysis.utils import dictobj, convert_time_to_string, read_melt_lake_h5

import warnings
warnings.filterwarnings("ignore")

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
        # is2time = lk.date_time
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
        is2datetime = datetime.strptime(is2time, time_format_out)
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
def add_gt_to_imagery(fn, img, ax, xlm=[None, None], arrow_width=20, arrow_col='k', arrow_ls='-', line_col='r', line_width=1, 
                      arrow_label=None, line_label=None):
    
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
                          
    df['x10'] = np.round(df.xatc, -1)
    gt = df.groupby(by='x10')[['lat', 'lon']].median().reset_index()
    ximg, yimg = warp.transform(src_crs='epsg:4326', dst_crs=img.crs, xs=np.array(gt.lon), ys=np.array(gt.lat))
    if not arrow_label:
        arrow_label = '%s (%s beam)' % (lk.gtx.upper(), lk.beam_strength)
    arrow_gt = ax.arrow(ximg[0], yimg[0], ximg[-1]-ximg[0], yimg[-1]-yimg[0], label=arrow_label, length_includes_head=True,
                        width=arrow_width, head_width=3*arrow_width, head_length=5*arrow_width, color=arrow_col, ls=arrow_ls)
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

    return arrow_gt, line_extent


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
    # if not imagery_filename:
    #     imagery_filename = 'imagery' + fn[fn.rfind('/'):].replace('.h5','.tif')
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

                     
#####################################################################
def plotIS2(fn, ax=None, xlm=[None, None], ylm=[None,None], cmap=cmc.lapaz_r, name='ICESat-2 data',
            add_legend=True, add_lat_ax=False, phot_color='k'):
    lk = dictobj(read_melt_lake_h5(fn))
    df = lk.photon_data.copy()
    dfd = lk.depth_data.copy()
    if not xlm[0]:
        xlm[0] = df.xatc.min()
    if not xlm[1]:
        xlm[1] = df.xatc.max()
    if xlm[1] < 0:
        xlm[1] = df.xatc.max() + xlm[1]
    if not ylm[0]:
        ylm[0] = lk.surface_elevation-2*lk.max_depth
    if not ylm[1]:
        ylm[1] = lk.surface_elevation+1.0*lk.max_depth
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
        # ax.scatter(df.xatc, df.h, s=1, c='k')
        hdl_phot = ax.scatter(df.xatc[~df.is_afterpulse], df.h[~df.is_afterpulse], s=1, color=phot_color, label='ATL03 photons')
    else:
        ax.scatter(df.xatc[~df.is_afterpulse], df.h[~df.is_afterpulse], s=1, c=df.snr, cmap=cmap)
        
    # ax.scatter(dfd.xatc[isdepth], dfd.h_fit_bed[isdepth], s=4, color='r', alpha=dfd.conf[isdepth])
    # ax.plot(dfd.xatc, dfd.h_fit_bed, color='gray', lw=0.5)
    
    hdl_bed, = ax.plot(dfd.xatc, bed, color='r', lw=1, label='lakebed')
    hdl_surf, = ax.plot(dfd.xatc, surf, color='C0', lw=1, label='open water surface')

    df['x10'] = np.round(df.xatc, -1)
    gt = df.groupby(by='x10')[['lat', 'lon']].median().sort_values(by='x10').reset_index()
    
    # add latitude
    lx = gt.sort_values(by='x10').iloc[[0,-1]][['x10','lat']].reset_index(drop=True)
    _lat = np.array(lx.lat)
    _xatc = np.array(lx.x10)
    def lat2xatc(l):
        return _xatc[0] + (l - _lat[0]) * (_xatc[1] - _xatc[0]) /(_lat[1] - _lat[0])
    def xatc2lat(x):
        return _lat[0] + (x - _xatc[0]) * (_lat[1] - _lat[0]) / (_xatc[1] - _xatc[0])
    
    ax.tick_params(axis='both', which='major', pad=0.5)
    if add_lat_ax:
        secax = ax.secondary_xaxis(-0.1, functions=(xatc2lat, lat2xatc))
        secax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
        secax.set_xlabel('along-track distance / latitude',labelpad=0)
        secax.tick_params(axis='both', which='major', pad=0.5)
    # secax.tick_params(axis='both', which='major')
    # secax.ticklabel_format(useOffset=False, style='plain')

    # add the length of surface
    arr_y = lk.surface_elevation+lk.max_depth*0.1
    x_start = np.min(xatc_surf)
    x_end = np.max(xatc_surf)
    x_mid = (x_end + x_start) / 2
    len_surf_m = np.floor((x_end-x_start)/10)*10
    len_surf_km = len_surf_m/1000
    arr_x1 = x_mid - len_surf_m / 2
    arr_x2 = x_mid + len_surf_m / 2
    ax.annotate('', xy=(arr_x1, arr_y), xytext=(arr_x2, arr_y),
                         arrowprops=dict(width=0.7, headwidth=5, headlength=5, color='C0'),zorder=1000)
    ax.annotate('', xy=(arr_x2, arr_y), xytext=(arr_x1, arr_y),
                         arrowprops=dict(width=0.7, headwidth=5, headlength=5, color='C0'),zorder=1000)
    ax.text(x_mid, arr_y, r'\textbf{%.2f km}' % len_surf_km, fontsize=10, ha='center', va='bottom', color='C0', fontweight='bold',
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
    # arr_x = xlm[0] - 0.0* (xlm[1] - xlm[0])
    y_len = y_up - y_low
    y_mid = (y_up + y_low) / 2
    arr_len = y_len
    arr_y1 = y_mid + arr_len / 2.1
    arr_y2 = y_mid - arr_len / 2.1
    ref_index = 1.336
    dep_round = np.round(y_len / ref_index, 1)
    ax.annotate('', xy=(arr_x, arr_y2), xytext=(arr_x, arr_y1),
                         arrowprops=dict(width=0.7, headwidth=5, headlength=5, color='r'),zorder=1000)
    ax.annotate('', xy=(arr_x, arr_y1), xytext=(arr_x, arr_y2),
                         arrowprops=dict(width=0.7, headwidth=5, headlength=5, color='r'),zorder=1000)
    ax.text(arr_x, y_mid, r'\textbf{%.1f m}' % dep_round, fontsize=10, ha='right', va='center', color='r', fontweight='bold',
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
    sheet = lk.ice_sheet
    region = lk.polygon_filename.split('_')[-1].replace('.geojson', '')
    if sheet == 'AIS':
        region = region + ' (%s)' % lk.polygon_filename.split('_')[-2]
    latstr = lk.lat_str[:-1] + '°' + lk.lat_str[-1]
    lonstr = lk.lon_str[:-1] + '°' + lk.lon_str[-1]
    description = '%s\n%s - %s\n(%s, %s)' % (datestr, sheet, region, latstr, lonstr)

    # ax.text(0.5, 0.87, description, fontsize=12, ha='center', va='top', transform=ax.transAxes,
    #        bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.2,rounding_size=0.5', lw=0), zorder=3000)
    # ax.text(0.5, 0.9, '%s' % name, fontsize=18, ha='center', va='bottom', transform=ax.transAxes,
    #        bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.2,rounding_size=0.5', lw=0), fontweight='bold')

    if add_legend:
        # ax.legend(handles=[hdl_phot, hdl_surf, hdl_bed], fontsize=9, scatterpoints=4, framealpha=0.95,
        #           bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=3)
        ax.legend(handles=[hdl_phot, hdl_surf, hdl_bed], fontsize=9, scatterpoints=4, framealpha=0.95, loc='lower right', ncol=3)

    ax.set_xlim(xlm)
    ax.set_ylim(ylm)
    # ax.axis('off')

    # set axis labels for this particular lake
    # xticks = secax.get_xticks()
    if add_lat_ax:
        xticks = [77.525, 77.52, 77.515, 77.51, 77.505]
        xticklabs = [r'$%g$\textdegree N' % xt if xt>=0 else r'$%g$\textdegree S' % xt for xt in xticks]
        secax.set_xticks(xticks)
        secax.set_xticklabels(xticklabs, fontsize=10)

    # xticks = ax.get_xticks()
    xticks = [500., 1000., 1500., 2000., 2500.]
    xticklabs = [r'$%g$ km' % (xt/1000) for xt in xticks]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabs)

    h0 = lk.surface_elevation
    refract_idx = 1.336
    def h2d(h):
        return (h0 - h) / refract_idx
    def d2h(d):
        return h0 - d * refract_idx 
    dax = ax.secondary_yaxis(location='right', functions=(h2d, d2h))
    dax.xaxis.set_minor_locator(matplotlib.ticker.AutoMinorLocator())
    dax.set_ylabel(r'water depth (m)', rotation=-90, color='r', labelpad=10)
    
    max_depth = 12
    ylm = ax.get_ylim()
    xlm = ax.get_xlim()
    ax.plot([xlm[1]]*2, [h0, d2h(max_depth)], 'r-', zorder=1000)
    
    yticks = np.arange(0, max_depth+0.01, 2)
    dax.set_yticks(yticks)
    dax.tick_params(axis='y', colors='red')
    ax.set_ylabel('elevation above geoid (m)')

def make_legend_arrow(legend, orig_handle, xdescent, ydescent, width, height, fontsize):
    return mpatches.FancyArrow(0, 0.5*height, width, 0, length_includes_head=True, head_width=0.5*height)

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