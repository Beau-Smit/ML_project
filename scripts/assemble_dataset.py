import pandas as pd
import rasterio
from pathlib import Path
import numpy as np



def main():
	# Workflow: 
	# 1. Grab a point from the landslide dataset
	# 2. Grab that point from the elevations
	# 	2.1 Convert geotiff into geopandas dataframe or something keras can interpret
	# 3. Grab that point & date from the precipitation datasets
	#	3.1 Convert precip data into something keras can interpret
	# 4. Grab that point & date from the landscan data
	# 	4.1 Convert into something keras can interpret
	# 5. Repeat
	df = read_in_landslides()
	for index_val in df.index.tolist():
		row = df.loc[index_val]
		lat, lon = row['latitude'], row['longitude']
		elev = read_elev(index_val)


def read_in_landslides():
	landslide_path = Path.cwd() / '..' / 'data' / 'landslides_to_use.csv'
	if not landslide_path.exists():
		# subset extant dataset
		df = pd.read_csv('../data/nasa_global_landslide_catalog_point.csv')
		df = df[(df['country_name'] is not np.nan) and (df['country_name'] == 'United States')]
		null_categories = df[df['landslide_category'].isnull()]
		df = df[~df['landslide_category'].isnull()]
		df = df[df['landslide_category'] != 'snow_avalanche']
		df = df[~df['landslide_trigger'].isin(['construction', 'earthquake', 'vibration'])]
		df = pd.concat([df, null_categories])
		df = df[~df['event_date'].isnull()]
		df['event_year'] = [int(date.split('-')[0]) for date in df['event_date']]
		df = df[df['event_year'] >= 2000]
		df['Landslide Occurred'] = 1
		print(len(df))
		# sys.exit()

		# Duplicate the landslides for the prior year where year >2000
		later_than_2001 = df[df['event_year'] > 2000]
		later_than_2001['event_year'] = [int(year)-1 for year in later_than_2001['event_year']]
		later_than_2001['event_date'] = [str(year) + date[4:] for date, year in zip(df['event_date'], df['event_year'])]
		later_than_2001['Landslide Occurred'] = 0
		df = pd.concat([df, later_than_2001])

		# Read in set of negative examples
		# falses = 
		df.to_csv('../data/landslides_to_use.csv')
	else:
		df = pd.read_csv(landslide_path, index_col='event_id')
	return df


def read_elev(index_val):
	elev_path = Path.cwd() / '..' / 'data' / 'elevations' / (str(index_val) + '.tiff')
	elev = rasterio.open(elev_path.as_posix())
	print(elev.crs)
	print(elev.height, elev.width)
	print(elev.count)
	sys.exit()


if __name__ == '__main__':
	main()