import requests
from multiprocessing import Pool
import random
from datetime import date
from pathlib import Path
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta


RANDOM_SEED = 12
DATA_PATH = Path.cwd() / '..' / 'data'

def main():
	df = process_landslides()
	all_dates, file_format_dates = process_dates(df)
	df = pull_precip(df, file_format_dates)


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
	dates = list(set(df['event_date']))
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
	end_date_part    = datey+relativedelta(minutes=30)
	end_date_part    = str(end_date_part-relativedelta(seconds=1)).split(' ')[1].replace(':', '')
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
	df['event_date'] = df['event_date'].dt.round('h')
	df['event_date'] = [pd.Timestamp(str(event_date)) for event_date in df['event_date']]
	df['true_slide'] = 1
	df = add_prev_year_falses(df)
	return df


def add_prev_year_falses(df):
	df['elevation_file'] = [str(index_val) + '.tiff' for index_val in df.index.tolist()]
	false_df = df.copy()
	false_df['true_slide'] = 0
	false_df['event_date'] = false_df['event_date'].apply(lambda date: date - pd.DateOffset(years=1))
	false_df['elevation_file'] = ['year_later_'+true_slide_tiff for true_slide_tiff in df['elevation_file']]
	for year_later_tiff, true_slide_tiff in zip(false_df['elevation_file'], df['elevation_file']):		
		year_later_tiff_path = DATA_PATH / 'elevations' / year_later_tiff
		true_slide_tiff_path = DATA_PATH / 'elevations' / true_slide_tiff
		if year_later_tiff_path.exists():
			continue
		with open(true_slide_tiff_path, 'rb') as f:
			tiff_file = f.read()
		with open(year_later_tiff_path, 'wb') as f:
			f.write(tiff_file)
	full_df = pd.concat([df,false_df])
	return full_df

if __name__ == '__main__':
	main()