import requests
from multiprocessing import Pool
import random
from datetime import date
from pathlib import Path
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from netCDF4 import Dataset
import geopandas as gpd
from shapely.geometry import Point
import rasterio
import statistics

RANDOM_SEED = 12
DATA_PATH = Path.cwd() / '..' / 'data'
PULL_PRECIPITATION = False

def main():
	df = process_landslides()
	df = add_elevations(df)
	all_dates, file_format_dates = process_dates(df)
	if PULL_PRECIPITATION:
		pull_precip(df, file_format_dates)
	df = add_precip(df, file_format_dates)
	df = add_fires(df)
	df = clean(df)
	df.to_csv('../data/final_landslides.csv')


def clean(df):
	df = df.reset_index()
	df = df[['true_slide', 'latitude', 'longitude', 'max_elev_change', 'median_elev', 
	         'mean_elev', 'mean_median_diff', 'total_elev', 'median_elev_change', 
	         'mean_elev_change', 'total_elev_change', 'Precip_30_min_before', 
	         'Precip_60_min_before', 'Precip_90_min_before', 'Precip_120_min_before',
	         'fire_<=1_year_before', 'fire_<=3_year_before', 'fire_<=5_year_before']]
	return df


def add_elevations(df):
	elevation_helper_tuples = [(index_val, elev_filename) for index_val, elev_filename in zip(df.index.tolist(), df['elevation_file'])]
	with Pool(8) as p:
		elev_var_dict_list = p.map(elevation_helper, elevation_helper_tuples)
	elev_vars = pd.DataFrame(data=elev_var_dict_list)
	elev_vars.index = elev_vars['index']
	for col in elev_vars.columns:
		df[col] = elev_vars[col].tolist()
	return df


def elevation_helper(elevation_helper_tuple):
	index_val, elev_filename = elevation_helper_tuple[0], elevation_helper_tuple[1]
	elev_path = DATA_PATH / 'elevations' / elev_filename
	elev_var_dict = {}
	with rasterio.open(elev_path) as elev_file:
		elev_array = elev_file.read()[0]
		flat_elev_array  = elev_array.flatten()
		flat_elev_array.sort()
		elev_var_dict['max_elev_change']  = abs(max(flat_elev_array) - min(flat_elev_array))
		elev_var_dict['median_elev']      = np.median(flat_elev_array)
		elev_var_dict['mean_elev']	      = (sum(flat_elev_array)/len(flat_elev_array))
		elev_var_dict['mean_median_diff'] = abs(elev_var_dict['mean_elev'] - elev_var_dict['median_elev'])
		elev_var_dict['total_elev']       = np.sum(elev_array)
		
		elev_diff_rows   = np.diff(elev_array)
		elev_diff_cols   = np.diff(np.flip(elev_array, axis=0))
		elev_diff        = np.add(elev_diff_cols, elev_diff_rows)
		flat_elev_diff   = elev_diff.flatten()
		elev_var_dict['median_elev_change']  = np.median(flat_elev_diff)
		elev_var_dict['mean_elev_change']    = np.mean(flat_elev_diff)
		elev_var_dict['total_elev_change']   = np.sum([abs(x) for x in flat_elev_diff])
		elev_var_dict['index'] = index_val
	return elev_var_dict



def add_fires(df):
	df = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
	fires_path = DATA_PATH / 'wildfires' / 'wildfire.geojson'
	wildfire_gdf = gpd.read_file(fires_path)
	wildfire_gdf.crs = 'ESRI:102008'
	wildfire_gdf = wildfire_gdf.to_crs('EPSG:4326')
	df = gpd.sjoin(df, wildfire_gdf, how='left')
	df['year'] = [int(event_date.year) for event_date in df['event_date']]
	df['fire_<=1_year_before'] = [1 if year - fire_year <= 1 else 0 for year, fire_year in zip(df['year'], df['DN'])]
	df['fire_<=3_year_before'] = [1 if year - fire_year <= 3 else 0 for year, fire_year in zip(df['year'], df['DN'])]
	df['fire_<=5_year_before'] = [1 if year - fire_year <= 5 else 0 for year, fire_year in zip(df['year'], df['DN'])]
	return df


def add_precip(df, file_format_dates):
	precip_path = DATA_PATH / 'precip'
	precip_files = [file for file in precip_path.iterdir()]
	precip_data_dict_list = []
	for date_set, df_index_val in zip(file_format_dates, range(len(df.index.tolist()))):
		obs = df.iloc[df_index_val]
		obs_lat = obs['latitude']
		obs_lon = obs['longitude']
		precip_data_dict = {}
		for minute_offset, file_format_date in date_set.items():
			file_found = False
			precip_path_index = 0
			while not file_found:
				if file_format_date in precip_files[precip_path_index].name:
					file_found = True
					found_file = precip_files[precip_path_index]
					break
				precip_path_index += 1
			if not file_found:
				raise Exception(f'File matching {file_format_date} not found')
			with Dataset(found_file) as precip_dataset:
				precip_matrix = np.asarray(precip_dataset['precipitationCal'][:])[0]
				lat_matrix = precip_dataset['lat'][:]
				lon_matrix = precip_dataset['lon'][:]
				closest_lat_index, closest_lon_index = closest_val_idx(lat_matrix, obs_lat), closest_val_idx(lon_matrix, obs_lon)
				precip_data_dict[minute_offset] = precip_matrix[closest_lon_index][closest_lat_index]
		precip_data_dict_list.append(precip_data_dict)
		# sys.exit()
	precip_df = pd.DataFrame(data=precip_data_dict_list)
	precip_df = precip_df.rename(columns={30: 'Precip_30_min_before', 60: 'Precip_60_min_before', 
		                                  90: 'Precip_90_min_before', 120: 'Precip_120_min_before'})
	for col in precip_df.columns:
		df[col] = precip_df[col].tolist()
	return df


def closest_val_idx(value_list, main_val):
	value_list = np.asarray(value_list)
	closest_val_index = (np.abs(value_list - main_val)).argmin()
	return closest_val_index

def create_falses(df):
	false_slide_pts = gpd.read_file('../data/fake_slides/fake_slides.shp')
	false_slide_pts['longitude'] = false_slide_pts['geometry'].x
	false_slide_pts['latitude'] = false_slide_pts['geometry'].y
	date_set = list(set(df['event_date']))
	false_slide_pts = create_drops_and_files(false_slide_pts, 'false_')
	dates = [random.sample(date_set, 1)[0] for index_val in false_slide_pts.index.tolist()]
	false_slide_pts['event_date'] = dates
	false_slide_pts['true_slide'] = 0
	df = pd.concat([df, false_slide_pts])
	return df


def pull_precip(df, file_format_dates):
	all_file_format_dates = [file_format_date_set.values() for file_format_date_set in file_format_dates]
	all_file_format_dates = [file_format_date for sub_list in all_file_format_dates for file_format_date in sub_list]
	link_path = DATA_PATH / 'subset_GPM_3IMERGHHE_06_20210511_165905.txt'
	links = link_path.read_text()
	links = links.split('\n')
	useful_links = []
	for link in links:
		if link.endswith('pdf'):
			continue
		link_date_part = link.split('3IMERG.')[1]
		link_date_part = link_date_part.split('.')[0]
		if link_date_part in all_file_format_dates:
			useful_links.append(link)
	with open('useful_precip_links.txt', 'w+') as f:
		f.writelines(useful_links)
	with Pool(8) as p:
		p.map(pull_precip_helper, useful_links)


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


def process_dates(df):
	dates = df['event_date']
	all_dates, file_format_dates = date_helper(dates)
	return all_dates, file_format_dates


def date_helper(dates):
	all_dates = []
	file_format_dates = []
	minute_offsets = [30, 60, 90, 120] 
	for datey in dates:
		date_set = {}
		for offset in minute_offsets:
			offset_date = datey-relativedelta(minutes=offset)
			all_dates.append(offset_date)
			date_set[offset] = format_date(offset_date)
		file_format_dates.append(date_set)
	return all_dates, file_format_dates


def format_date(datey):
	date_part        = str(datey).split(' ')[0].replace('-', '')
	start_time_part  = str(datey).split(' ')[1].replace(':', '')
	end_date_part	 = datey+relativedelta(minutes=30)
	end_date_part	 = str(end_date_part-relativedelta(seconds=1)).split(' ')[1].replace(':', '')
	full_date_string = date_part+'-S'+start_time_part+'-E'+str(end_date_part)
	return full_date_string


def process_landslides():
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
	df = df.loc[df['event_date'].dt.year >= 2001]
	df['event_date'] = df['event_date'].dt.round('h')
	df['event_date'] = [pd.Timestamp(str(event_date)) for event_date in df['event_date']]
	df['true_slide'] = 1
	df['geometry'] = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
	df = create_drops_and_files(df, '')
	df = add_prev_year_falses(df)
	df = create_falses(df)
	df = df.drop(['Unnamed: 0'], 1)
	return df


def create_drops_and_files(df, file_name_start):
	drops = []
	files = []
	for index_val in df.index.tolist():
		slide_file = file_name_start + str(index_val) + '.tiff'
		slide_path = DATA_PATH / 'elevations' / slide_file
		if slide_path.stat().st_size < 10000:
			drops.append(False)
		else:
			drops.append(True)
		files.append(slide_file)
	df['drops'] = drops
	df['elevation_file'] = files
	df = df[df['drops']]
	df = df.drop('drops', 1)
	return df


def add_prev_year_falses(df):
	df['elevation_file'] = [str(index_val) + '.tiff' for index_val in df.index.tolist()]
	false_df = df.copy()
	false_df['true_slide'] = 0
	false_df['event_date'] = false_df['event_date'].apply(lambda date: date - pd.DateOffset(years=1))
	false_df['elevation_file'] = ['year_later_'+true_slide_tiff for true_slide_tiff in df['elevation_file']]
	false_df = false_df.loc[false_df['event_date'].dt.year >= 2001]
	drops = []
	for year_later_tiff, true_slide_tiff in zip(false_df['elevation_file'], df['elevation_file']):		
		year_later_tiff_path = DATA_PATH / 'elevations' / year_later_tiff
		true_slide_tiff_path = DATA_PATH / 'elevations' / true_slide_tiff
		if true_slide_tiff_path.stat().st_size < 10000:
			drops.append(False)
		else:
			drops.append(True)

		if year_later_tiff_path.exists():
			continue
		with open(true_slide_tiff_path, 'rb') as f:
			tiff_file = f.read()
		with open(year_later_tiff_path, 'wb') as f:
			f.write(tiff_file)
	false_df['drop'] = drops
	false_df = false_df[false_df['drop']]
	false_df = false_df.drop('drop', 1)
	full_df = pd.concat([df,false_df])
	return full_df

if __name__ == '__main__':
	main()