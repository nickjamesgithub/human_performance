import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
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
events_list_w =np.sort(events_list_w)

# Loop over years of analysis
years = np.linspace(2001,2021,21)
years = years.astype("int")
coordinates = pd.read_csv("/Users/tassjames/Desktop/Olympic_data/olympic_data/country_coordinates_olympic.csv")
geographic_concentration_norms = []
event_labels = []

# def haversine(lon1, lat1, lon2, lat2):
for g in range(len(genders)):
    for i in range(len(events_list_m)):
        # Slice a particular event
        event = genders[g][(genders[g]["event"] == events_list_m[i])]

        # Mean/event/year - men
        geo_list_year = [] # Average Male distance
        for j in range(len(years)):
            geo_country_event = event.loc[(event['Date'] == years[j]), 'Nat']
            # geo_country_event_m_s = geo_country_event_m.str.replace('"', '')
            geo_list_year.append(np.array(geo_country_event))

        # Loop over each year and get lat and long from each country
        nationalities = []
        lats_longs = []
        country_codes_list = []
        for r in range(len(geo_list_year)):
            for c in range(len(geo_list_year[0])):
                country_code = geo_list_year[r][c]
                country_code_strip = country_code.replace('"', '')
                nationalities.append(country_code)
                slice = coordinates.loc[coordinates['Athlete_Code'] == country_code]
                lat = np.array(slice["Latitude_n"])
                long = np.array(slice["Longitude_n"])
                lats_longs.append([lat, long])
                country_codes_list.append(country_code)
                print(country_code)

        # Counter, limit of 100
        counter = 0
        norms = []
        while counter < 100 * len(years):
            location_slice = lats_longs[counter:counter+100]
            distance_matrix = np.zeros((len(location_slice), len(location_slice)))
            missingness = 0
            for a in range(len(location_slice)):
                for b in range(len(location_slice)):
                    try:
                        lat1 = location_slice[a][0][0]
                        long1 = location_slice[a][1][0]
                    except: # If distance is not found use central latitude/longitude
                        lat1 = 40.52
                        long1 = 34.34
                        missingness += 1
                        print("We had a missing lat/long", missingness)
                    try:
                        lat2 = location_slice[b][0][0]
                        long2 = location_slice[b][1][0]
                    except: # If distance is not found use central latitude/longitude
                        lat2 = 40.52
                        long2 = 34.34
                    h_distance = haversine(long1, lat1, long2, lat2)
                    distance_matrix[a,b] = h_distance
                counter += 1

            # relabel
            label = re.sub('[!@#$\/]', '', events_list_m[i])

            # Compute norm of distance matrix
            l2_norm = np.linalg.norm(distance_matrix)
            norms.append(l2_norm) # Append L2 Norm
            # counter += 100
            print(counter)

        # Append Geographic concentration norms
        geographic_concentration_norms.append(norms)
        event_labels.append([gender_labels[g], events_list_m[i]])

        # Print out event ordering
        print(events_list_m[i], gender_labels[g])

# Print all event labels
print(event_labels)

event_names = ["High jump M", "Long jump M", "Pole vault M", "Triple jump M", "Discus M", "Hammer throw M", "Javelin M", "Shot put M",
               "High jump W", "Long jump W", "Pole vault W", "Triple jump W", "Discus W", "Hammer throw W", "Javelin W", "Shot put W"]

# Loop over sequential norms
total_conc_vector = []
for i in range(len(geographic_concentration_norms)):
    title = event_names[i]
    total_concentration = np.sum(geographic_concentration_norms[i]) * 10**-7
    total_conc_vector.append([total_concentration, title])
    plt.plot(geographic_concentration_norms[i], label=title)
plt.title("Field geographic concentration")
plt.legend()
plt.savefig("Field_geographic_concentration")
plt.show()

# Make it an array Concentration
concentration_scores = np.array(total_conc_vector)
concentation_ordered = concentration_scores[concentration_scores[:, 0].argsort()]
print(concentation_ordered)


