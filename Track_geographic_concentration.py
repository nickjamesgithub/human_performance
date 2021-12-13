import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from Utilities import generate_olympic_data, dendrogram_plot_labels, haversine
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import re

gini_sample = 10 # 10/100
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
geographic_concentration_norms = []
geographic_concentration_gini = [] # geographic concentration gini
event_labels = []

# def haversine(lon1, lat1, lon2, lat2):
for g in range(len(genders)):
    for i in range(len(events_list_m)):
        # Slice a particular event
        event = genders[g][(genders[g]["event"] == events_list_m[i])]

        # Mean/event/year - men
        geo_list_year = [] # Average Male distance
        for j in range(len(years)):
            geo_country_event = event.loc[(event['Date_Y'] == years[j]), 'Nat']
            # geo_country_event_m_s = geo_country_event_m.str.replace('"', '')
            geo_list_year.append(np.array(geo_country_event))

        def gini(list_of_values):
            sorted_list = sorted(list_of_values)
            height, area = 0, 0
            for value in sorted_list:
                height += value
                area += height - value / 2.
            fair_area = height * len(list_of_values) / 2.
            return (fair_area - area) / fair_area

        # Gini index
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        gini_coeff_list = []
        for gc in range(len(geo_list_year)):
            if gini_sample == 100:
                values = label_encoder.fit_transform(geo_list_year[gc]) # Transform categorical data into integers
                gini_coeffs = gini(values)
                gini_coeff_list.append(gini_coeffs) # append gini coefficients
            if gini_sample == 10:
                slice = geo_list_year[gc][:10]
                values = label_encoder.fit_transform(slice) # Transform categorical data into integers
                gini_coeffs = gini(values)
                gini_coeff_list.append(gini_coeffs) # append gini coefficients

        # Loop over each year and get lat and long from each country
        lats_longs = []
        country_codes_list = []
        for r in range(len(geo_list_year)):
            for c in range(len(geo_list_year[0])):
                country_code = geo_list_year[r][c]
                country_code_strip = country_code.replace('"', '')
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
                    except:
                        lat1 = 40.52
                        long1 = 34.34
                        missingness += 1
                        print("We had a missing lat/long", missingness)
                    try:
                        lat2 = location_slice[b][0][0]
                        long2 = location_slice[b][1][0]
                    except:
                        lat2 = 40.52
                        long2 = 34.34
                    h_distance = haversine(long1, lat1, long2, lat2)
                    distance_matrix[a,b] = h_distance
                counter += 1

            if make_plots:
                # Plot distance matrix
                plt.matshow(distance_matrix)
                plt.show()

            # relabel
            label = re.sub('[!@#$\/]', '', events_list_m[i])

            # Compute norm of distance matrix
            l2_norm = np.linalg.norm(distance_matrix)
            norms.append(l2_norm) # Append L2 Norm
            # counter += 100
            print(counter)

        # Append Geographic concentration norms
        geographic_concentration_norms.append(norms)
        geographic_concentration_gini.append(gini_coeff_list)
        event_labels.append([gender_labels[g], events_list_m[i]])

        # Print out event ordering
        print(events_list_m[i], gender_labels[g])

# Print all event labels
print(event_labels)

event_names = ["M 10k", "M 1500m", "M 3k", "M 5k", "M 800m", "M 100m", "M 200m", "M 400m",
               "W 10k", "W 1500m", "W 3k", "W 5k", "W 800m", "W 100m", "W 200m", "W 400m"]

# Loop over sequential norms
total_gini_conc_vector = []
for i in range(len(geographic_concentration_gini)):
    # title = event_names[i]
    total_concentration = np.sum(geographic_concentration_gini[i])
    total_gini_conc_vector.append([total_concentration, event_names[i]])
    plt.plot(geographic_concentration_gini[i], label=event_names[i])
plt.title("Track geographic concentration (Gini)")
plt.legend()
plt.savefig("Track_geographic_concentration_gini"+str(gini_sample))
plt.show()

# Loop over sequential norms
total_conc_vector = []
for i in range(len(geographic_concentration_norms)):
    title = event_names[i]
    total_concentration = np.sum(geographic_concentration_norms[i]) * 10**-7
    total_conc_vector.append([total_concentration, title])
    plt.plot(geographic_concentration_norms[i], label=title)
plt.title("Track geographic concentration")
plt.legend()
plt.savefig("Track_geographic_concentration")
plt.show()

# Make it an array Concentration (Gini)
gini_concentration_scores = np.array(total_gini_conc_vector)
gini_concentation_ordered = gini_concentration_scores[gini_concentration_scores[:, 0].argsort()]
print(gini_concentation_ordered)

# Make it an array Concentration
concentration_scores = np.array(total_conc_vector)
concentation_ordered = concentration_scores[concentration_scores[:, 0].argsort()]
print(concentation_ordered)

# Compute linear model components for long jump and hammer throw
date_grid = np.linspace(2001,2019,19)
five_k_m, five_k_b = np.polyfit(date_grid, geographic_concentration_norms[11], 1)
two_hundred_m, two_hundred_b = np.polyfit(date_grid, geographic_concentration_norms[14], 1)

# Check largest and smallest geographic dispersion and get indices: Index 1 and 5
date_grid = np.linspace(2001,2019,19)
fig,ax = plt.subplots()
plt.scatter(date_grid, geographic_concentration_norms[11], label="Women's 5k", color='blue', alpha=0.7)
plt.plot(date_grid, five_k_m * date_grid + five_k_b, color='blue', alpha=0.7)
plt.scatter(date_grid, geographic_concentration_norms[14], label="Women's 200m", color='red', alpha=0.7)
plt.plot(date_grid, two_hundred_m * date_grid + two_hundred_b, color='red', alpha=0.7)
plt.xlabel("Date")
plt.ylabel("Distance matrix norm")
plt.locator_params(axis='x', nbins=4)
plt.tight_layout()
plt.legend()
plt.savefig("Geographic_two_plot_track")
plt.show()




