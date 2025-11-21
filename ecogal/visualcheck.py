
'''
Getting the summary (if exists) file for a given ra and dec provided
'''

import numpy as np
from astropy.table import Table
from astropy.utils.data import download_file
from astropy.coordinates import SkyCoord
from astropy import units as u
import shapely
from shapely import Point, Polygon
from grizli import utils

slits_url = "https://grizli-cutout.herokuapp.com/nirspec_slits?coord={0:s},{1:s}"
base_url = 'https://s3.amazonaws.com/alma-ecogal/dr1'
CACHE_DOWNLOADS=True

##################################################
#---- define plotting function
##################################################


def fpstr_to_region(fp_str):
    fp_str = fp_str.replace('(','').replace(')','')
    if "," in fp_str:
        cords = fp_str.strip().split(",")

    poly_array = np.asarray(cords[:], dtype=float).reshape((-1, 2))
    poly_region = Polygon(poly_array)
    return poly_region


def get_footprint(ra,dec,metadata= None, version='v1'):

    if metadata==None:
        table_url = f"{base_url}/ancillary/ecogal_{version}_metadata.fits"
        meta = utils.read_catalog(download_file(table_url, cache=CACHE_DOWNLOADS), format='fits')
    else:
        meta = metadata
    
    gal_pos = Point(ra,dec)
    bool_region = np.zeros(len(meta),dtype=bool)
    
    for i in range(len(meta)):
        poly_region = fpstr_to_region(meta['footprint'][i])
        if shapely.within(gal_pos, poly_region):
            bool_region[i] = True
    
    fp = meta[bool_region]
    
    if np.sum(bool_region)>0:
        print(f'There are #{np.sum(bool_region)} ALMA projects overlapping')
    else:
        print('No overlap found within ALMA/ECOGAL')

    return fp, bool_region

def get_summary(ra, dec, r_search = 0.4, catname = 'ecogal_all_priors_v1.csv'):
    #######################################################
    ### getting the closest galaxies from the prior catalogue
    #getting the source within the footprint
    #cross-match with the meta data
    #rfile = Table.read('ecogal_v1_metadata.fits')

    rfile = Table.read(
        download_file(
            base_url+'/ancillary/ecogal_v1_metadata.fits',
            cache=True,
        ),
        format="fits"
    )

    cord_target = SkyCoord(ra,dec,unit=(u.degree, u.degree))

    gal_pos = Point(ra,dec)

	#getting the footprint
    fp, bool_region = get_footprint(ra,dec,metadata=rfile)

    if 1:
        atb = Table.read(
            download_file(
            base_url+f'/catalogue/{catname}',
                cache=True,
            ),
            format="csv",
        )
    con_file = np.zeros(len(atb), dtype=bool)

    for ii in range(np.sum(bool_region)):
        if np.sum(con_file) == 0:
            con_file = atb['file_alma'] == rfile[bool_region]['file_alma'][ii]
        else:
            con_file |= atb['file_alma'] == rfile[bool_region]['file_alma'][ii]

    ra_opt = atb[con_file]['RA_parent']
    dec_opt = atb[con_file]['Dec_parent']
    cord_opt = SkyCoord(ra_opt, dec_opt, unit=(u.degree, u.degree))
    sep = cord_opt.separation(cord_target).arcsec
    con_sep = sep < r_search 

    if np.sum(con_sep)>0 and (True):
        print(f'There are {np.sum(con_sep)} ECOGAL+DJA cross-match!')
        table_info = atb[con_file][con_sep]
        gal_uniq = np.unique(table_info['id_new'])
        
        if len(gal_uniq)==1:
            tab = table_info
            gal_uniq = gal_uniq[0]
            sep = sep[con_sep][0]
        else:
            print(f'There are multiple sources ({len(gal_uniq)}) within the searching area; choosing the closest available')
            idx_close = np.argmin(sep[con_sep])
            print(f'closest separation = {sep[con_sep][idx_close]:.2f}')
            # data retreival
            gal_uniq = table_info['id_new'][idx_close]
            tab = table_info[idx_close:idx_close+1]
            sep = sep[con_sep][idx_close]

        Summary_URL = f'{base_url}/pngs/ecogal__0_all_filters_{gal_uniq}.png'
        print(Summary_URL)
        print(f'A source found at a distance of = {sep:.2f} arcsec')

        return Summary_URL
    
    else:
        print('It is either not detected, no coverage')
        return None

