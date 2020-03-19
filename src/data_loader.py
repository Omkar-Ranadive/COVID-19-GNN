import pandas as pd
from geopy.geocoders import Nominatim
import pickle
import time
from src.utils import save_pickle, load_pickle


def form_lldict(df, geo_run=False, val_provided=True):
	'''
	This function forms a dictionary with key = (latitude, longitude)
	Each such key denotes a node and has certain node statistic associated with it.

	Args:
		df (pandas dataframe): Pandas dataframe
		val_provided (bool): Is true if the number of cases as provided as an explicit column
	Returns:
		Saves/returns a dictionary
	'''

	columns_of_interest = ['country', 'latitude', 'longitude', 'value']

	if not val_provided:
		# Add a new column to the dataframe if the value is not provided
		df['value'] = 1

	df = df[columns_of_interest]
	save_path = '../dataset/generated/usa/lldict_usa'
	# Initialize geopy
	geolocator = Nominatim(user_agent="COVID-19-Spread", timeout=5)

	# Only consider those where latitude and longitude values are non-zero
	df = df.dropna(subset=['latitude', 'longitude'])

	# Form the lat and long dictionary
	coordinate_dict = dict()
	for values in df.values:
		country, lat, lon, count_infected = values
		# Store the count and country information in the co-ordinate dict. Later, city and more stats can be added.
		coordinate = (lat, lon)
		if coordinate in coordinate_dict:
			coordinate_dict[coordinate][-1] += 1  # Increase count of cases at that lat, lon
		else:
			if geo_run:
				print("Running geo-mapping for : ", coordinate)
				time.sleep(1)  # To comply with 1 req/s limit
				# We use geopy to get accurate node statistcs for that lat, lon
				location = geolocator.reverse("{}, {}".format(str(lat), str(lon)), language='en')
				if 'address' not in location.raw:
					print("Geo-mapping failed for the following: ", coordinate, country)
				else:
					address = location.raw['address']
					# For now, we are enforcing country = 'United States of America'
					if address['country'] == 'United States of America':
						node_statistcs = [address['country']]
						# We are choosing lowest granularity as a city. If not available, choose a bigger one like county.
						granularity = ['city', 'county', 'state_district', 'state']
						for val in granularity:
							if val in address:
								node_statistcs.append(address[val])
								break

						# Always separately append the state to later deal with flight data correctly
						if 'state' in address:
							node_statistcs.append(address['state'])
							# Finally, add the count
							node_statistcs.append(count_infected)
							coordinate_dict[coordinate] = node_statistcs
						else:
							print("State not found for {}. Debug info: {}".format(coordinate, address))
					else:
						print("Country other than US found. Remove the if condition if all countries are desired")
			else:
				# If not geopy, simply add the country
				# A specific fix only for the COVID19_openline_list.csv
				if coordinate[0] == 23.75947 and coordinate[1] == 120.9559:
					country = 'Taiwan'
				coordinate_dict[coordinate] = [country, count_infected]

	# Save the dictionary
	with open(save_path, 'wb') as file:
		pickle.dump(coordinate_dict, file)

	return coordinate_dict


def filter_data(df, filter_dict):
	'''

	Args:
		df (Pandas dataframe): A pandas dataframe
		filter_dict (dict): A dictionary where key = column name and value = the filter value.
	Returns: Filtered Pandas dataset

	'''
	for k, v in filter_dict.items():
		df = df[df[k] == v]

	return df


def create_flight_data(df):
	'''

	Args:
		df (Pandas dataframe): Travel data

	Returns: Saves a dictionary where key = (state1, state2) and value = count of flights
	'''

	columns_of_interest = ['ORIGIN_STATE_NM', 'DEST_STATE_NM']

	# Loop through the columns and increase the count
	flight_dict = dict()
	for s1, s2 in df[columns_of_interest].values:
		# Assuming an undirected graph, hence using a frozen set
		if s1 != s2:
			pair = frozenset([s1, s2])
			if pair in flight_dict:
				flight_dict[pair] += 1
			else:
				flight_dict[pair] = 1

	save_path = '../dataset/generated/flightdict'
	save_pickle(flight_dict, save_path)


def form_population_data():
	'''
	Specialized function for turning state-wise population information into features
	Returns: Saves a dictionary with key = state name and value = (population, population density, population over 65 in %)
	'''

	path1 = '../dataset/population_density_usa.csv'
	path2 = '../dataset/population_old_usa.csv'

	df1 = pd.read_csv(path1)
	df2 = pd.read_csv(path2)
	column1 = ['State', 'Population', 'Density']
	column2 = ['State', 'Population65+%']
	pop_info = dict()

	for values in df1[column1].values:
		state, pop, density = values
		pop_info[state] = [float(pop.replace(',', '')), float(density.replace(',', ''))]

	for values in df2[column2].values:
		state, pop65 = values
		pop_info[state].append(float(pop65))

	save_pickle(pop_info, '../dataset/generated/usa/pop_info')


if __name__ == '__main__':
	# path = '../dataset/COVID19_open_line_list.csv'
	path = '../dataset/data/COVID19.csv'
	filter_dict = {'country': 'US', 'date': '3/13/2020'}
	df = pd.read_csv(path)
	df = filter_data(df, filter_dict)
	# lldict = form_lldict(df, geo_run=True)
	form_population_data()

	# df = pd.read_csv('../dataset/data/travel_data.csv')
	# create_flight_data(df)
	# print(df)


# df = pd.read_csv(path)
	# columns = ['age', 'sex', 'city', 'province', 'country', 'latitude', 'longitude',  'date_onset_symptoms',
	# 		   'date_admission_hospital']
	# df = df[columns]
	# print(len(df))
	# missing_values = df.isnull().sum()
	# non_missing = df.notnull().sum()
	#
	# test = {}
	#
	# for lat, lon in df[['latitude', 'longitude']].values:
	# 	if (lat, lon) in test:
	# 		test[(lat, lon)] += 1
	# 	else:
	# 		test[(lat, lon)] = 1
	#
	# print(len(test))
	# print(missing_values[missing_values > 0])
	#
	# df_fil = df[df.country.eq('Japan')]
	# print(df_fil[['city', 'country', 'province', 'latitude', 'longitude']])
	# form_lldict(path, geo_run=True)
