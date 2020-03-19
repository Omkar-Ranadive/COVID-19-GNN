import pandas as pd
import numpy as np
import geopy.distance
from src.utils import load_pickle, save_pickle
from pathlib import Path
'''
This file is specifically for time-series data 
'''

SOURCE_PATH = Path(__file__).parent / '../dataset/timeseries'
n_rows = 133752


def get_US_states_data(states, dataframe):
    """

    Args: dataframe: coronavirus dataset time_series-ncov-Confirmed.csv for all countries (
    https://data.humdata.org/dataset/novel-coronavirus-2019-ncov-cases)

    Returns:
        features (numpy.ndarray): no. of cases in all countries
        dataframe: coronavirus data for US
    """
    dataframe.Date = pd.to_datetime(df.Date)
    dataframe = dataframe[(dataframe.Province.isin(states)) & (dataframe.Country == 'US') & (dataframe.Date >= '3/1/2020') & (dataframe.Date <= '3/12/2020')]
    # print(dataframe.groupby('Province')['Value'].apply(np.hstack))
    features = np.stack(dataframe.groupby('Province')['Value'].apply(np.hstack))
    # feature = dataframe.Value.values
    # print(feature)

    return features, dataframe


def create_dist_matrix(dataframe):
    """

    Args:
        dataframe: coronavirus data for US

    Returns:
        provinces (numpy.ndarray): nodes in the graph
        dist_matrix (numpy.ndarray): geographical distance between two points (miles)
    """
    provinces = dataframe.Province.unique()
    # print(provinces)
    lat_longs = list(zip(dataframe.Lat.unique(), dataframe.Long.unique()))
    province_coordinate_mappings = dict(zip(provinces, lat_longs))
    # print(province_coordinate_mappings)
    # print(province_coordinate_mappings)
    distance_matrix = np.zeros((len(provinces), len(provinces)))
    for i in range(len(provinces)):
        for j in range(i + 1, len(provinces)):

            source = province_coordinate_mappings[provinces[i]]
            destination = province_coordinate_mappings[provinces[j]]
            distance_matrix[i, j] = geopy.distance.distance(source, destination).miles
            if i == 0 and j == 2:
                print(source, destination, distance_matrix[i,j])

            distance_matrix[j, i] = distance_matrix[i, j]

    # print(distance_matrix)
    return distance_matrix


def create_flight_matrix(provinces, travel_data):
    """

    Args: provinces (numpy.ndarray): nodes in the graph travel_data (numpy.ndarray): flights frequencies data (
    https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=236)

    Returns:
        flight_matrix (numpy.ndarray): flight frequencies between two regions
    """
    # print(provinces)
    # print(travel_data)
    print(provinces)
    flight_matrix = np.zeros((len(provinces), len(provinces)))
    travel_data = travel_data.head(n_rows)
    travel_data = travel_data.groupby(['ORIGIN_STATE_NM', 'DEST_STATE_NM']).size().to_frame('COUNT').reset_index()\
        .rename(columns={'ORIGIN_STATE_NM': 'ORIGIN', 'DEST_STATE_NM': 'DEST'})
    # print(travel_data)
    # city2state = travel_data.groupby(['ORIGIN_CITY_NAME', 'DEST_STATE_NM']).size().to_frame('COUNT').reset_index()
    # .rename(columns = {'ORIGIN_CITY_NAME': 'ORIGIN', 'DEST_STATE_NM': 'DEST'})
    # state2city = travel_data.groupby(['ORIGIN_STATE_NM', 'DEST_CITY_NAME']).size().to_frame('COUNT').reset_index()
    # .rename(columns = {'ORIGIN_STATE_NM': 'ORIGIN', 'DEST_CITY_NAME': 'DEST'})
    # city2city = travel_data.groupby(['ORIGIN_CITY_NAME', 'DEST_CITY_NAME']).size().to_frame('COUNT').reset_index()
    # .rename(columns = {'ORIGIN_CITY_NAME': 'ORIGIN', 'DEST_CITY_NAME': 'DEST'})
    # travel_data = pd.concat([state2state, city2city, state2city, city2state])
    frequency_mappings = dict(zip(list(zip(travel_data.ORIGIN.tolist(), travel_data.DEST.tolist())), travel_data.COUNT
                                  .tolist()))
    # print(frequency_mappings)
    abbr_state_mappings = load_pickle(SOURCE_PATH / 'data/abbrev_state_mappings.pkl')

    for i in range(len(provinces)):
        for j in range(len(provinces)):
            try:
                # if provinces[i].find(', ') != -1 and provinces[j].find(', ') != -1:
                #     flight_matrix[i, j] = frequency_mappings[(abbr_state_mappings[provinces[i][-2:]],
                #                                               abbr_state_mappings[provinces[j][-2:]])]
                # elif provinces[i].find(', ') != -1:
                #     flight_matrix[i, j] = frequency_mappings[(abbr_state_mappings[provinces[i][-2:]], provinces[j])]
                # elif provinces[j].find(', ') != -1:
                #     flight_matrix[i, j] = frequency_mappings[(provinces[i], abbr_state_mappings[provinces[j][-2:]])]
                # else:
                flight_matrix[i, j] = frequency_mappings[(provinces[i], provinces[j])]
            except KeyError:
                continue

    return flight_matrix


def get_us_pop_data(df1, df2):
    """

    Args:
        dataframe:

    Returns:

    """
    df1 = df1.set_index('State')
    df2 = df2.set_index('State')
    df1.sort_index(inplace=True)
    df2.sort_index(inplace=True)

    df1['Population65+%'] = df2['Population65+%']

    print(df1[['Population', 'Density', 'Population65+%']])
    return df1[['Population', 'Density', 'Population65+%']].to_numpy()


if __name__ == "__main__":
    df = pd.read_csv(SOURCE_PATH / 'data/COVID19.csv')
    travel_df = pd.read_csv(SOURCE_PATH / 'data/travel_data.csv')
    selected_cols = ['Province', 'Country', 'Lat', 'Long', 'Date', 'Value']
    list_of_states = load_pickle(SOURCE_PATH / 'data/us_states_list.pkl')
    df = df[selected_cols]
    features, df = get_US_states_data(list_of_states, df)
    dist_matrix = create_dist_matrix(df)
    travel_matrix = create_flight_matrix(list_of_states, travel_df)
    age_df = pd.read_csv(SOURCE_PATH / 'data/population_old_usa.csv')
    pop_df = pd.read_csv(SOURCE_PATH / 'data/population_density_usa.csv')
    pop_age_df = get_us_pop_data(pop_df, age_df)
    features = np.append(features, pop_age_df, axis=1)
    print(travel_matrix)
    save_pickle(features, SOURCE_PATH / 'data/features.pkl')
    save_pickle(dist_matrix, SOURCE_PATH / 'data/dist_matrix.pkl')
    save_pickle(travel_matrix, SOURCE_PATH / 'data/travel_matrix.pkl')
    a = load_pickle('data/pop_info')
    print(a)

