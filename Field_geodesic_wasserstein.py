import scipy.stats as sp
from pyemd import emd, emd_with_flow
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
events_list_m = np.sort(events_list_m)
events_list_w = np.sort(events_list_w)

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
    r = 6378000  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

# Read in geographic data
coordinates = pd.read_csv("/Users/tassjames/Desktop/Olympic_data/olympic_data/country_coordinates_olympic_GW.csv")
country_labels = coordinates["Athlete_Code"]

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
            geo_country_event = event.loc[(event['Date'] == years[j])]

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

            # Test
            test = np.sum(country_counts)
            print("test counts", test)

        # Generate Density function
        mutual_information = []
        for t1 in range(len(country_counts_list)):
            # Generate distribution of nationalities at each year
            counts_time_1 = country_counts_list[t1]
            counts_time_pdf_1 = country_counts_list[t1]/np.sum(country_counts_list[t1])
            a= 0.001
            b = 1.0
            unif_dist = np.linspace(sp.uniform.ppf(0.01, a, b), sp.uniform.ppf(0.99, a, b), len(counts_time_pdf_1))

            # Compute Mutual Information
            mi = mutual_info_score(counts_time_pdf_1, unif_dist)
            mutual_information.append(mi)

        plt.plot(mutual_information)
        plt.show()

        block = 1



#
# # Loop over time series
# counter = 0
# geodesic_variance = []
# for t in range(len(new_cases_burn[0])):  # Looping over time
#     cases_slice = new_cases_burn[:, t]  # Slice of cases in time
#     cases_slice_pdf = np.nan_to_num(cases_slice / np.sum(cases_slice))
#
#     # Country variance matrix
#     country_variance = np.zeros((len(cases_slice_pdf), len(cases_slice_pdf)))
#
#     # Loop over all the countries in the pdf
#     for x in range(len(cases_slice_pdf)):
#         for y in range(len(cases_slice_pdf)):
#             # # Print state names
#             # print(names_dc[x])
#             # print(names_dc[y])
#             country_x_density = cases_slice_pdf[x]
#             country_y_density = cases_slice_pdf[y]
#             lats_x = usa_location_data["Latitude"][x]
#             lats_y = usa_location_data["Latitude"][y]
#             longs_x = usa_location_data["Longitude"][x]
#             longs_y = usa_location_data["Longitude"][y]
#             geodesic_distance = haversine(longs_x, lats_x, longs_y, lats_y)
#
#             # Compute country variance
#             country_variance[x, y] = geodesic_distance ** 2 * country_x_density * country_y_density
#
#     # Sum above the diagonal
#     upper_distance_sum = np.triu(country_variance).sum() - np.trace(country_variance)
#     geodesic_variance.append(upper_distance_sum)
#     print("Iteration " + str(t) + " / " + str(len(new_cases_burn[0])))
#
# # Time-varying geodesic variance
# plt.plot(date_index_array_burn, geodesic_variance)
# plt.xlabel("Time")
# plt.ylabel("Geodesic Wasserstein variance")
# plt.title("Spatial variance")
# plt.savefig("Geodesic_variance_individual_PDF")
# plt.show()




