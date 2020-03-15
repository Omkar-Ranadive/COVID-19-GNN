import pandas as pd
from geopy.geocoders import Nominatim
import pickle
import time


def form_lldict(path, geo_run=False):
	'''
	This function forms a dictionary with key = (latitude, longitude)
	Each such key denotes a node and has certain node statistic associated with it.

	Args:
		path (string): The path to dataset file
	Returns:
		Saves/returns a dictionary
	'''

	df = pd.read_csv(path)
	columns_of_interest = ['city', 'province', 'country', 'latitude', 'longitude']
	df = df[columns_of_interest]
	save_path = '../dataset/generated/lldict'
	# Initialize geopy
	geolocator = Nominatim(user_agent="COVID-19-Spread", timeout=5)

	# Only consider those where latitude and longitude values are non-zero
	df = df.dropna(subset=['latitude', 'longitude'])

	# Form the lat and long dictionary
	coordinate_dict = dict()
	for values in df.values:
		city, province, country, lat, lon = values
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
					print(location.raw)
				else:
					address = location.raw['address']
					node_statistcs = [address['country']]
					# We are choosing lowest granularity as a city. If not available, choose a bigger one like county.
					granularity = ['city', 'county', 'state_district', 'state']

					for val in granularity:
						if val in address:
							node_statistcs.append(address[val])
							break

					# Finally, add the count
					node_statistcs.append(1)
					coordinate_dict[coordinate] = node_statistcs
			else:
				# If not geopy, simply add the country
				# A specific fix only for the COVID19_openline_list.csv
				if coordinate[0] == 23.75947 and coordinate[1] == 120.9559:
					country = 'Taiwan'
				coordinate_dict[coordinate] = [country, 1]

	# Save the dictionary
	with open(save_path, 'wb') as file:
		pickle.dump(coordinate_dict, file)

	return coordinate_dict


if __name__ == '__main__':
	path = '../dataset/COVID19_open_line_list.csv'
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
	form_lldict(path, geo_run=True)
