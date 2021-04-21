import pandas as pd
from shapely import wkt
import geopandas as gpd
from pathlib import Path
import numpy as np
import config
import requests
import netCDF4


DATA_PATH = Path.cwd() / '..' / 'data'

def main():
	main_df_path = DATA_PATH / 'nasa_global_landslide_catalog_point.csv'
	df = pd.read_csv(main_df_path)
	df = df[(df['country_name'] is not np.nan) and (df['country_name'] == 'United States')]
	pull_elevations(df)
	df 

def pull_elevations(df):
	for index_val in df.index.tolist():
		obs = df.loc[index_val]
		args = {'west': obs['longitude']-.1, 'east': obs['longitude']+.1, 'south': obs['latitude'], 'north': obs['latitude'],
		        'demtype': 'SRTMGL3', 'API_key': config.OPEN_ELEV_KEY}
		response = requests.get('https://portal.opentopography.org/API/globaldem', params=args)
		tiff_path = DATA_PATH / 'elevations' / (str(obs['event_id'])+'.tiff')
		with open(tiff_path, 'wb+') as wb_file:
			wb_file.write(response.content)


if __name__ == '__main__':
	main()