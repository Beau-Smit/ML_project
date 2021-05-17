import pandas as pd
from shapely import wkt
from shapely.ops import unary_union
import geopandas as gpd
from pathlib import Path
import numpy as np
import config
import requests
import netCDF4
import h5py
from multiprocessing import Pool
import subprocess
import random
from datetime import date

RANDOM_SEED = 12
DATA_PATH = Path.cwd() / '..' / 'data'


def main():
	main_df_path = DATA_PATH / 'nasa_global_landslide_catalog_point.csv'
	df = pd.read_csv(main_df_path)
	df = df[(df['country_name'] is not np.nan) and (df['country_name'] == 'United States')]
	null_categories = df[df['landslide_category'].isnull()]
	df = df[~df['landslide_category'].isnull()]
	df = df[df['landslide_category'] != 'snow_avalanche']
	df = df[~df['landslide_trigger'].isin(['construction', 'earthquake', 'vibration'])]
	df = pd.concat([df, null_categories])
	pull_elevations(df, pull_trues=False, pull_falses=True)
	# pull_precipitation()
	# organize_precipitation()

def pull_elevations(df, pull_trues, pull_falses):
	if pull_trues:
		elev_files = []
		for index_val in df.index.tolist():
			obs = df.loc[index_val]
			# pull_elevations_helper(obs, index_val)
			elev_files.append(str(index_val)+'.tiff')
		df['elev_files'] = elev_files
		df.to_csv('nasa_global_landslide_catalog_point.csv')
	if pull_falses:
		# The falses were generated with QGIS3 by loading the USA shapefile found in data/usa_shp
		# then going to vector -> research tools -> random points inside polygons
		# and selecting a number that's 4x the length of the dataframe and adding the stipulation
		# that the points must be minimum .001 degree away from each other
		# We then assign them dates randomly here
		false_slide_pts = gpd.read_file('../data/false_slide_points/false_slide_points.shp')
		false_slide_pts['longitude'] = false_slide_pts['geometry'].x
		false_slide_pts['latitude'] = false_slide_pts['geometry'].y
		start_date = date.today().replace(day=1, month=1, year=2000).toordinal()
		end_date = date.today().replace(day=31, month=12, year=2020).toordinal()
		dates = []
		elev_files = []
		for index_val in false_slide_pts.index.tolist():
			obs = false_slide_pts.loc[index_val]
			pull_elevations_helper(obs, index_val)
			random_date = date.fromordinal(random.randint(start_date, end_date))
			print(type(random_date))
			elev_file = 'false_'+str(index_val)+'.tiff'
			elev_files.append(elev_file)
			dates.append(random_date)
		false_slide_pts['event_date'] = dates
		# false_slide_pts['elev_files'] = elev_files
		false_slide_pts['Landslide Occurred'] = 0
		falses = falses.to_csv('../data/false_landslide_pts.csv')


def pull_elevations_helper(obs, index_val):
	args = {'west': obs['longitude']-.1, 'east': obs['longitude']+.1, 'south': obs['latitude']-.1, 'north': obs['latitude']+.1,
			'demtype': 'SRTMGL3', 'API_key': config.OPEN_ELEV_KEY}
	tiff_path = DATA_PATH / 'elevations' / ('false_'+str(index_val)+'.tiff')
	if tiff_path.exists():
		print(f'{str(tiff_path)} exists, continuing')
		return
	response = requests.get('https://portal.opentopography.org/API/globaldem', params=args)
	with open(tiff_path, 'wb+') as wb_file:
		wb_file.write(response.content)
	print(f'wrote {str(tiff_path)}')


def pull_precipitation():
	link_path = DATA_PATH / 'subset_GPM_3IMERGHH_06_20210422_204824.txt'
	links = link_path.read_text()
	links = links.split('\n')

	with Pool(8) as p:
		p.map(pull_precip_helper, links)


def pull_precip_helper(link):
	if link.endswith('.HDF5'):	
		link_file = link.strip('https://gpm1.gesdisc.eosdis.nasa.gov/data/GPM_L3/')
		link_file = link_file.replace('/', '_')
		link_path = DATA_PATH / 'precip' / link_file
		if link_path.exists():
			print(f'{link_path} exists, skipping')
			return
		proc = subprocess.run(['curl', '-n', '-c', '~/.urs_cookies', '-b', '~/.urs_cookies', '-LJO', '--url', link], 
			                  check=True,
			                  capture_output=True,
			                  cwd='../data/precip')
		print(f'Downloaded {link}')


def organize_precipitation():
	precip_path = DATA_PATH / 'precip'
	for path in precip_path.iterdir():
		with h5py.File(path.as_posix(), 'r') as f:
			a_group_key = list(f.keys())
			print(a_group_key)
			data = list(f[a_group_key])
			print(data)
			sys.exit()

if __name__ == '__main__':
	main()