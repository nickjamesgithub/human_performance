from pyemd import emd, emd_with_flow, emd_samples
from math import radians, cos, sin, asin, sqrt
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
import re
from Utilities import dendrogram_plot_labels

make_plots = False

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
    r = 6378  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

# Read in geographic data
coordinates = pd.read_csv("/Users/tassjames/Desktop/Olympic_data/olympic_data/country_coordinates_olympic_GW.csv")
country_labels = coordinates["Athlete_Code"]

# Read in Uniform distribution values
uniform_values = pd.read_excel("/Users/tassjames/Desktop/Olympic_data/Uniform_distribution.xlsx")
unif_vals = uniform_values.iloc[:,0]
unif_pdf = np.float64(unif_vals/np.sum(unif_vals))

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
gw_trajectories = []
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

        # Append GW trajectories to major list
        gw_trajectories.append(geodesic_wasserstein)

        if make_plots:
            # Plot Geodesic Wasserstein between uniform and Olympic distribution
            plt.plot(geodesic_wasserstein)
            plt.xlabel("Date")
            plt.ylabel("Geodesic Wasserstein")
            plt.title(gender_labels[g] + "_" + label)
            plt.savefig("Geodesic_wasserstein_"+gender_labels[g]+"_"+label)
            plt.show()

# Geodesic Wasserstein trajectories
gw_trajectory_matrix = np.zeros((len(gw_trajectories), len(gw_trajectories)))
for i in range(len(gw_trajectories)):
    for j in range(len(gw_trajectories)):
        gw_tr_i = np.array(gw_trajectories[i]/np.sum(gw_trajectories[i]))
        gw_tr_j = np.array(gw_trajectories[j]/np.sum(gw_trajectories[j]))
        gw_trajectory_matrix[i,j] = np.sum(np.abs(gw_tr_i - gw_tr_j))

cluster_labels = ["M high jump", "M long jump", "M pole vault", "M triple jump", "M discus", "M hammer throw", "M javelin",  "M shot put",
                  "W high jump", "W long jump", "W pole vault", "W triple jump", "W discus", "W hammer throw", "W javelin",  "W shot put"]

# Create an array for geodesic matrix trajectories list of lists
gw_trajectory_matrix_array = np.array(gw_trajectory_matrix)

# Form distance matrix between geodesic wasserstein trajectories
dendrogram_plot_labels(gw_trajectory_matrix_array, "Geodesic_variance_trajectory_", "Field_", cluster_labels)