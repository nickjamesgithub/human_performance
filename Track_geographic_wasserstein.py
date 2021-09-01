import scipy.stats as sp
from pyemd import emd, emd_with_flow, emd_samples
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mutual_info_score
import math
from Utilities import generate_olympic_data, dendrogram_plot_labels, haversine
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import re
from scipy.stats import wasserstein_distance


# We better push our code!
make_plots = False
path = '/Users/tassjames/Desktop/Olympic_data/olympic_data/track' # use your path
all_files = glob.glob(path + "/*.csv")

li = []
li_specialised = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    df_slice = df[["Rank", "Mark_seconds", "Nat", "gender", "event", "Date"]]
    li.append(df)
    li_specialised.append(df_slice)

# Generate dataframe with attributes of interest
frame = pd.concat(li, axis=0, ignore_index=True)
frame_sp = pd.concat(li_specialised, axis=0, ignore_index=True)

# Change date format to just year %YYYY
frame_sp['Date_Y'] = frame_sp['Date'].str[-2:]
frame_sp['Date_Y'] = str("20") + frame_sp['Date'].str[-2:]
frame_sp['Date_Y'] = pd.to_numeric(frame_sp['Date_Y'])
frame_sp['Mark_seconds'] = pd.to_numeric(frame_sp['Mark_seconds'], errors='coerce')

# Slice male and female performance
men_best_performance = frame_sp[(frame_sp['gender'] == "men")]
women_best_performance = frame_sp[(frame_sp['gender'] == "women")]
genders = [men_best_performance, women_best_performance]
gender_labels = ["men", "women"]

# Get event lists
events_list_m = men_best_performance["event"].unique()
events_list_w = women_best_performance["event"].unique()
events_list_m = np.sort(events_list_m)
events_list_w = np.sort(events_list_w)

# Loop over years of analysis
years = np.linspace(2001,2019,19)
years = years.astype("int")
coordinates = pd.read_csv("/Users/tassjames/Desktop/Olympic_data/olympic_data/country_coordinates_olympic.csv")
country_labels = coordinates["Athlete_Code"]

geographic_concentration_norms = []
event_labels = []

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    """
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])

    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 6378  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

# Global geographic distance
global_geographic_distance = np.zeros((len(coordinates), len(coordinates)))
for i in range(len(coordinates)):
    for j in range(len(coordinates)):
        lat_i = coordinates['Latitude_n'][i]
        long_i = coordinates['Longitude_n'][i]
        lat_j = coordinates['Latitude_n'][j]
        long_j = coordinates['Longitude_n'][j]
        distance = haversine(long_i, lat_i, long_j, lat_j)
        global_geographic_distance[i, j] = distance
global_geographic_distance = np.nan_to_num(global_geographic_distance)

# Loop over years of analysis and get counts of each country
years = np.linspace(2001,2019,19)
years = years.astype("int")

# Label_list
label_list = []
# def haversine(lon1, lat1, lon2, lat2):
for g in range(len(genders)):
    for i in range(len(events_list_m)):

        # Slice a particular event
        event = genders[g][(genders[g]["event"] == events_list_m[i])]

        # Generate distribution counts
        country_names_list = []
        country_counts_list = []
        for j in range(len(years)):
            geo_country_event = event.loc[(event['Date_Y'] == years[j])]

            country_names = []
            country_counts = []
            for c in range(len(country_labels)):
                ctry_label = country_labels[c]
                counts = len(geo_country_event[geo_country_event["Nat"] == ctry_label])
                country_counts.append(counts)
                country_names.append(ctry_label)
            # Append country names and country counts to major list
            country_names_list.append(country_names)
            country_counts_list.append(country_counts)

            # Test counts
            print("Test total counts", np.sum(country_counts))

        # relabel
        label = re.sub('[!@#$\/]', '', events_list_m[i])

        # Generate Density function
        mutual_information = []
        geodesic_wasserstein = []
        for t in range(len(country_counts_list)):
            # Generate distribution of nationalities at each year
            counts_time_1 = np.float64(np.nan_to_num(country_counts_list[t]))
            counts_time_array = np.nan_to_num(np.float64(counts_time_1).flatten())
            counts_time_pdf = np.float64(country_counts_list[t]/np.sum(country_counts_list[t]))

            # Uniform samples
            unif_ones = np.array([np.random.normal(1,.001,238)]).flatten().astype('float64')
            unif_pdf = np.float64(unif_ones/np.sum(unif_ones))
            gw_distance_t = np.nan_to_num(emd(counts_time_pdf, unif_pdf, global_geographic_distance))
            geodesic_wasserstein.append(gw_distance_t)

            print("Geodesic Wasserstein is "+gender_labels[g]+label, gw_distance_t)
            print(t)

        # # Plot mutual information between uniform and Olympic distribution
        # plt.plot(mutual_information)
        # plt.xlabel("Date")
        # plt.ylabel("Mutual Information")
        # plt.savefig("Mutual_information_"+gender_labels[g]+"_"+label)
        # plt.show()

        # Plot Geodesic Wasserstein between uniform and Olympic distribution
        plt.plot(geodesic_wasserstein)
        plt.xlabel("Date")
        plt.ylabel("Geodesic Wasserstein")
        plt.savefig("Geodesic_wasserstein_"+gender_labels[g]+"_"+label)
        plt.show()
