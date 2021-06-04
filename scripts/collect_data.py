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
	df = pd.concat([df, null_categories])
	df = df[~df['landslide_trigger'].isin(['construction', 'earthquake', 'vibration'])]
	df = df[~df['event_date'].isnull()]
	df['event_date'] = pd.to_datetime(df['event_date'])
	df['event_date'] = df['event_date'].dt.round('h')
	pull_elevations(df, pull_trues=True, pull_falses=True)
	# pull_precipitation()
	# organize_precipitation()


def pull_elevations(df, pull_trues, pull_falses):
	gen_prev_year_falses = True
	if pull_trues:
		elev_files = []
		tups = [(df.loc[index_val], index_val, False) for index_val in df.index.tolist()]
		with Pool(8) as p:
			p.map(pull_elevations_helper, tups)
		elev_files = [str(index_val)+'.tiff' for index_val in df.index.tolist()]
		df['elev_files'] = elev_files
		df.to_csv('nasa_global_landslide_catalog_point.csv')
	if pull_falses:
		if gen_prev_year_falses:
			prev_year_false_dates = create_prev_year_falses(df)
		# The falses were generated with QGIS3 by loading the USA shapefile found in data/continuous_us
		# then going to vector -> research tools -> random points inside polygons
		# and selecting 15000 points with the additional stipulation
		# that the points must be minimum .001 degree away from each other
		# We then assign them dates randomly here
		false_slide_pts = gpd.read_file('../data/fake_slides/fake_slides.shp')
		false_slide_pts['longitude'] = false_slide_pts['geometry'].x
		false_slide_pts['latitude'] = false_slide_pts['geometry'].y
		false_slide_pts = create_false_dates(false_slide_pts)
		start_date = date.today().replace(day=1, month=1, year=2000).toordinal()
		end_date = date.today().replace(day=31, month=12, year=2020).toordinal()
		elev_files = []
		tups = [(false_slide_pts.loc[index_val], index_val, True) for index_val in false_slide_pts.index.tolist()]
		with Pool(8) as p:
			p.map(pull_elevations_helper, tups)
		elev_files = ['false_'+str(index_val)+'.tiff' for index_val in df.index.tolist()]
		false_slide_pts['event_date'] = dates
		false_slide_pts['elev_files'] = elev_files
		false_slide_pts['Landslide Occurred'] = 0
		falses = false_slide_pts.to_csv('../data/false_landslide_pts.csv')


def create_prev_year_falses(df):
	# pseudocode: for every slide, look a year back
	# and calculate the date for that
	df['event_date'] = [pd.Timestamp(str(event_date)) for event_date in df['event_date']]
	df['true_slide'] = 1
	df['elevation_file'] = [str(index_val) + '.tiff' for index_val in df.index.tolist()]
	false_df = df.copy()
	false_df['true_slide'] = 0
	false_df['event_date'] = false_df['event_date'].apply(lambda date: date - pd.DateOffset(years=1))
	false_df['elevation_file'] = ['year_later_'+true_slide_tiff for true_slide_tiff in df['elevation_file']]
	for year_later_tiff, true_slide_tiff in zip(false_df['elevation_file'], df['elevation_file']):
		with open('../data/elevations/'+true_slide_tiff, 'rb') as f:
			tiff_file = f.read()
		with open('../data/elevations/'+year_later_tiff, 'wb') as f:
			f.write(tiff_file)
	full_df = pd.concat([df,false_df])
	return full_df


def create_false_dates(false_slide_pts):
	# for every false slide point in the shapefile, generate a random date in the range
	# June 2000 to end of 2020
	start_date = date.today().replace(day=1, month=1, year=2000).toordinal()
	end_date = date.today().replace(day=31, month=12, year=2020).toordinal()
	false_slide_pts['event_date'] = [date.fromordinal(random.randint(start_date, end_date)) for x in range(len(false_slide_pts))]
	false_slide_pts['true_slide'] = 0
	return false_slide_pts


def pull_elevations_helper(tup):
	obs, index_val, false = tup[0], tup[1], tup[2]
	args = {'west': obs['longitude']-.1, 'east': obs['longitude']+.1, 'south': obs['latitude']-.1, 'north': obs['latitude']+.1,
			'demtype': 'SRTMGL3', 'API_key': config.OPEN_ELEV_KEY}
	if false:
		tiff_path = DATA_PATH / 'elevations' / ('false_'+str(index_val)+'.tiff')
	else:
		tiff_path = DATA_PATH / 'elevations' / (str(index_val)+'.tiff')
	if tiff_path.exists():
		print(f'{str(tiff_path)} exists, continuing')
		return
	response = requests.get('https://portal.opentopography.org/API/globaldem', params=args)
	with open(tiff_path, 'wb+') as wb_file:
		wb_file.write(response.content)
	print(f'wrote {str(tiff_path)}')


def pull_precipitation():
	link_path = DATA_PATH / 'subset_GPM_3IMERGHHE_06_20210511_165905.txt'
	links = link_path.read_text()
	links = links.split('\n')

	with Pool(8) as p:
		p.map(pull_precip_helper, links)


def pull_precip_helper(link):
	if 'pdf' not in link:
		link_file = link.split('3IMERGHHE.')[-1]
		link_file = link_file.split('?')[0]
		link_file = link_file.replace('/', '_')
		link_path = DATA_PATH / 'precip' / link_file
		if link_path.exists():
			print(f'{link_path} exists, skipping')
			return
		response = requests.get(link)
			# result.raise_for_status()
		with open('../data/precip/'+link_file, 'wb') as f:
			f.write(response.content)
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