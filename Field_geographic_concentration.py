import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from Utilities import generate_olympic_data, dendrogram_plot_labels, haversine
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import re

path = '/Users/tassjames/Desktop/Olympic_data/olympic_data/field' # use your path
all_files = glob.glob(path + "/*.csv")

li = []
li_specialised = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    df_slice = df[["Rank", "Mark", "Nat", "gender", "event", "Date"]]
    li.append(df)
    li_specialised.append(df_slice)

# Generate dataframe with attributes of interest
frame = pd.concat(li, axis=0, ignore_index=True)
frame_sp = pd.concat(li_specialised, axis=0, ignore_index=True)

# Change date format to just year %YYYY
frame_sp['Date'] = frame_sp['Date'].astype(str).str.extract('(\d{4})').astype(int)

# Slice male and female performance
men_best_performance = frame_sp[(frame_sp['gender'] == "men")]
women_best_performance = frame_sp[(frame_sp['gender'] == "women")]
genders = [men_best_performance, women_best_performance]
gender_labels = ["men", "women"]

# Get event lists
events_list_m = men_best_performance["event"].unique()
events_list_w = women_best_performance["event"].unique()
# events_list = ['discus', 'high jump', 'shot put', 'triple jump', 'pole vault', 'long jump', 'javelin', 'hammer throw']
events_list_m.sort()
events_list_w.sort()

# Loop over years of analysis
years = np.linspace(2001,2021,21)
years = years.astype("int")
coordinates = pd.read_csv("/Users/tassjames/Desktop/Olympic_data/olympic_data/country_coordinates_olympic.csv")
coordinates["Code"] = coordinates["Code"].str.replace('"', '')
# coordinates["Latitude"] = coordinates["Latitude"].str.replace('"', '')
# coordinates["Longitude"] = coordinates["Longitude"].str.replace('"', '')

# def haversine(lon1, lat1, lon2, lat2):
for g in range(len(genders)):
    for i in range(len(events_list_m)):
        # Slice a particular event
        event_m = genders[0][(genders[0]["event"] == events_list_m[i])]
        event_f = genders[1][(genders[1]["event"] == events_list_w[i])]

        # Mean/event/year - men
        geo_list_year = [] # Average Male distance
        for j in range(len(years)):
            geo_country_event_m = event_m.loc[(event_m['Date'] == years[j]), 'Nat']
            # geo_country_event_m_s = geo_country_event_m.str.replace('"', '')
            geo_list_year.append(np.array(geo_country_event_m))

        # Loop over each year and get lat and long from each country
        lats_longs = []
        for r in range(len(geo_list_year)):
            for c in range(len(geo_list_year[0])):
                country_code = geo_list_year[r][c]
                country_code_strip = country_code.replace('"', '')
                slice = coordinates.loc[coordinates['Athlete_Code'] == country_code]
                lat = slice["Latitude"].to_numeric()
                long = slice["Longitude"].to_numeric()
                lats_longs.append([lat, long])
                print(country_code)

        counter = 0
        while counter < 100 * len(years):
            location_slice = lats_longs[counter:counter+100]
            distance_matrix = np.zeros((len(location_slice), len(location_slice)))
            for a in range(len(location_slice)):
                for b in range(len(location_slice)):
                    lat1 = location_slice[a][0]
                    long1 = np.array(location_slice[a][1]).flatten()
                    lat2 = np.array(location_slice[b][0]).flatten()
                    long2 = np.array(location_slice[b][1]).flatten()
                    h_distance = haversine(long1, lat1, long2, lat2)
                    block = 1
            counter += 100