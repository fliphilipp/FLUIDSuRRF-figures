import os
import re
import datetime
import requests
import zipfile
import shutil
from tqdm import tqdm
import scipy
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import rasterio as rio
from rasterio import plot as rioplot
from rasterio import warp
import rioxarray as rxr
import geopandas as gpd
import matplotlib.pyplot as plt
import cmcrameri.cm as cmc
import shapely
from IPython.display import Image, display

import sys
sys.path.append('../utils/')

region_locs = {
    'NO': {'x': -120431.650, 'y': -1168898.581},
    'NW': {'x': -78158.149, 'y': -1645985.187},
    'NE': {'x': 228324.732, 'y': -1440656.812},
    'CW': {'x': 145762.098, 'y': -2133990.738},
    'CE': {'x': 421873.448, 'y': -2131883.238},
    'SW': {'x': -66365.812, 'y': -2568102.241},
    'SE': {'x': 211221.203, 'y': -2542867.058},
    'B-C': {'x': 1390088.023, 'y': 390187.645},
    'E-Ep': {'x': 650235.078, 'y': -508470.614},
    'C-Cp': {'x': 2181162.763, 'y': -80855.393},
    'I-Ipp': {'x': -2000326.289, 'y': 1683057.240, 'x_arr': -2297371.959, 'y_arr': 1206478.372},
    'Dp-E': {'x': 13508.011, 'y': -2149665.988, 'x_arr': 435363.176, 'y_arr': -1797591.449},
    'Ep-F': {'x': -652195.939, 'y': -714117.616},
    'F-G': {'x': -1543332.950, 'y': -1431025.917, 'x_arr': -1216313.545, 'y_arr': -983539.323},
    'G-H': {'x': -1344923.206, 'y': -253876.324},
    'H-Hp': {'x': -2343071.293, 'y': -419112.120, 'x_arr': -1851555.623, 'y_arr': -126547.188},
    'Hp-I': {'x': -2251672.625, 'y': 289227.556, 'x_arr': -1908927.621, 'y_arr': 716842.777},
    'Ipp-J': {'x': -1565396.912, 'y': 1271066.807, 'x_arr': -1783881.485, 'y_arr': 923787.180},
    'J-Jpp': {'x': -1154888.612, 'y': 330438.612},
    'K-A': {'x': -434982.839, 'y': 1643070.016},
}

#####################################################################
def add_graticule(gdf, ax_img, meridians_locs=['bottom','right'], parallels_locs=['top','left']):
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
                ax_img.text(intx, inty, deglab_, fontsize=8, color='gray',verticalalignment='center',horizontalalignment=ha,
                        rotation=rot, rotation_mode='anchor')
        if 'top' in meridians_locs:
            ha = 'left' if rot>0 else 'right'
            deglab_ = addright+deglab if rot>0 else deglab+addleft
            intx,inty = intersection(topline[0], topline[1], gr_trans[0], gr_trans[1])
            if len(intx) > 0:
                intx, inty = intx[0], inty[0]
                ax_img.text(intx, inty, deglab_, fontsize=8, color='gray',verticalalignment='center',horizontalalignment=ha,
                        rotation=rot, rotation_mode='anchor')
        if 'right' in meridians_locs:
            intx,inty = intersection(rightline[0], rightline[1], gr_trans[0], gr_trans[1])
            if len(intx) > 0:
                intx, inty = intx[0], inty[0]
                ax_img.text(intx, inty, addright+deglab, fontsize=8, color='gray',verticalalignment='center',horizontalalignment='left',
                        rotation=rot, rotation_mode='anchor')
        if 'left' in meridians_locs:
            intx,inty = intersection(leftline[0], leftline[1], gr_trans[0], gr_trans[1])
            if len(intx) > 0:
                intx, inty = intx[0], inty[0]
                ax_img.text(intx, inty, deglab+addleft, fontsize=8, color='gray',verticalalignment='center',horizontalalignment='right',
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
                ax_img.text(intx, inty, deglab+addleft, fontsize=8, color='gray',verticalalignment='center',horizontalalignment='right',
                           rotation=rot, rotation_mode='anchor')
        if 'right' in parallels_locs:
            intx,inty = intersection(rightline[0], rightline[1], gr_trans[0], gr_trans[1])
            if len(intx) > 0:
                intx, inty = intx[0], inty[0]
                ax_img.text(intx, inty, addright+deglab, fontsize=8, color='gray',verticalalignment='center',horizontalalignment='left',
                           rotation=rot, rotation_mode='anchor')
        if 'top' in parallels_locs:
            ha = 'left' if rot>0 else 'right'
            deglab_ = addright+deglab if rot>0 else deglab+addleft
            intx,inty = intersection(topline[0], topline[1], gr_trans[0], gr_trans[1])
            if len(intx) > 0:
                intx, inty = intx[0], inty[0]
                ax_img.text(intx, inty, deglab_, fontsize=8, color='gray',verticalalignment='center',horizontalalignment=ha,
                           rotation=rot, rotation_mode='anchor')
        if 'bottom' in parallels_locs:
            ha = 'right' if rot>0 else 'left'
            deglab_ = deglab+addleft if rot>0 else addright+deglab
            intx,inty = intersection(bottomline[0], bottomline[1], gr_trans[0], gr_trans[1])
            if len(intx) > 0:
                intx, inty = intx[0], inty[0]
                ax_img.text(intx, inty, deglab_, fontsize=8, color='gray',verticalalignment='center',horizontalalignment=ha,
                           rotation=rot, rotation_mode='anchor')
        
        ax_img.plot(gr_trans[0],gr_trans[1],c=gridcol,ls=gridls,lw=thislw,alpha=0.5)
        ax_img.set_xlim(xl)
        ax_img.set_ylim(yl)

# for downloading all KML files (need to update url list manually)
def download_kmls(download_dir_kmls_zip, kml_url_list=None, download_zip=True):

    # Download the Zip files
    if not os.path.exists(download_dir_kmls_zip):
        os.makedirs(download_dir_kmls_zip)
        
    if download_zip: 
        for url in kml_url_list:
            kml_zip_fn = url[url.rfind('/')+1:]
            kml_zip_path = download_dir_kmls_zip + '/' + kml_zip_fn
            print('downloading', kml_zip_path)
            response = requests.get(url, stream=True, allow_redirects=True)
            total_size_in_bytes= int(response.headers.get('content-length', 0))
            block_size = 1024 #1 Kibibyte
            progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
            with open(kml_zip_path, 'wb') as file:
                for data in response.iter_content(block_size):
                    progress_bar.update(len(data))
                    file.write(data)
            progress_bar.close()
            if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
                print("ERROR, something went wrong")

    # Unzip outputs
    for z in os.listdir(download_dir_kmls_zip): 
        if z.endswith('.zip'): 
            zip_name = download_dir_kmls_zip + "/" + z 
            print('--> extracting', zip_name)
            zip_ref = zipfile.ZipFile(zip_name) 
            zip_ref.extractall(download_dir_kmls_zip) 
            zip_ref.close() 
            os.remove(zip_name)

    print('Cleaning up outputs folder...', end=' ')
    for root, dirs, files in os.walk(download_dir_kmls_zip, topdown=False):
        for file in files:
            try:
                shutil.move(os.path.join(root, file), download_dir_kmls_zip)
            except OSError:
                pass
        for name in dirs:
            shutil.rmtree(root+'/'+name, ignore_errors=True)
    print(' --> DONE!')

    kml_filelist = [download_dir_kmls_zip+'/'+f for f in os.listdir(download_dir_kmls_zip) \
                    if os.path.isfile(os.path.join(download_dir_kmls_zip, f))]
    
# for downloading Antarctic and Greenland Orbits
def download_ground_tracks_antarctica_greenland(download_dir_kmls_zip, kml_url_list=None, download_zip=True):
    download_path = 'data/kmls/raw'
    
    url_list = ['https://icesat-2.gsfc.nasa.gov/sites/default/files/page_files/antarcticaallorbits.zip',
                'https://icesat-2.gsfc.nasa.gov/sites/default/files/page_files/arcticallorbits.zip'
                ]
    
    download_kmls(download_path, kml_url_list=url_list, download_zip=True)

# for reading in KMLs and turning them into a dataframe
def kml2df(kml_filename):

    with open(kml_filename, 'r') as file:
        kml_string = file.read()
    
    rgt = int(kml_filename[kml_filename.find('RGT_')+4:kml_filename.find('RGT_')+8])
        
    kml_string_line = kml_string[kml_string.find('LineString_kml'):kml_string.find('</LineString>')]
    kml_coords_str = '[[' + kml_string_line[kml_string_line.find('<coordinates>')+len('<coordinates>'):kml_string_line.find('</coordinates>')-1] + ']]'
    kml_coords_str = kml_coords_str.replace(' ', '],[')
    kml_coords_array = np.array(eval(kml_coords_str))
    kml_lat = kml_coords_array[:,1]
    kml_lon = kml_coords_array[:,0] 
    kml_df = pd.DataFrame({'lat': kml_lat, 'lon': kml_lon, 'rgt': rgt})
    
    # find the first timestamp
    substr = kml_string[kml_string.find('<Point id='):kml_string.find('</Point>')+100]
    descr = substr[substr.find('<name>')+len('<name>'):substr.find('</name>')]
    dt_str = descr[descr.find('DOY-'):]
    dt_str = dt_str[dt_str.find(' ')+1:]
    day = int(dt_str[:dt_str.find('-')])
    month_abbr = dt_str[dt_str.find('-')+1:dt_str.rfind('-')]
    year = int(dt_str[dt_str.rfind('-')+1:dt_str.find(' ')])
    hrs = int(dt_str[dt_str.find(' ')+1:dt_str.find(':')])
    mins = int(dt_str[dt_str.find(':')+1:dt_str.rfind(':')])
    secs = int(dt_str[dt_str.rfind(':')+1:])
    datetime_str = '%4i-%3s-%02iT%02i:%02i:%02iZ' % (year, month_abbr, day, hrs, mins, secs)
    dt = datetime.datetime.strptime(datetime_str,'%Y-%b-%dT%H:%M:%SZ')
    timestamp_utc = datetime.datetime.timestamp(dt)
    kml_df['timestamp'] = timestamp_utc + np.arange(len(kml_df))
    kml_df['time_str'] = [datetime.datetime.strftime(datetime.datetime.fromtimestamp(t), '%a %Y-%b-%d %H:%M:%S') for t in kml_df.timestamp]
    
    return kml_df