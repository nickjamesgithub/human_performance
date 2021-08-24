import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from Utilities import generate_olympic_data, dendrogram_plot_labels
# from sklearn.linear_model import LinearRegression
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

events_list_m = np.sort(events_list_m)
events_list_w = np.sort(events_list_w)

# Loop over years of analysis
years = np.linspace(2001,2019,19)
direction_vector_list = []

for i in range(len(events_list_m)):
    # Slice a particular event
    event_m = genders[0][(genders[0]["event"] == events_list_m[i])]
    event_f = genders[1][(genders[1]["event"] == events_list_w[i])]

    # Mean/event/year - men
    means_m = [] # Average Male distance
    for j in range(len(years)):
        mean_year_event = event_m.loc[(event_m['Date'] == years[j]), 'Mark'].mean()
        means_m.append(mean_year_event)

    # Compute male returns
    returns_m = np.diff(means_m)

    # Mean/event/year - women
    means_f = [] # Average female distance
    for j in range(len(years)):
        mean_year_event = event_f.loc[(event_f['Date'] == years[j]), 'Mark'].mean()
        means_f.append(mean_year_event)

    # Compute male returns
    returns_f = np.diff(means_f)

    # Compute direction vector
    direction_vector = np.sum(returns_m*returns_f)/(np.linalg.norm(returns_m)*np.linalg.norm(returns_f))

    # Append to list
    direction_vector_list.append([events_list_m[i], direction_vector])

# Print out direction vector list
print(direction_vector_list)

# Initialize distance matrix
distance_matrix = np.zeros((len(events_list_m),len(events_list_m)))
distance_list = []
for i in range(len(events_list_m)):
    for g1 in range(len(genders)):
        for j in range(len(events_list_m)):
            for g2 in range(len(genders)):
                # Slice a particular event
                event_i = genders[g1][(genders[g1]["event"] == events_list_m[i])]
                event_j = genders[g2][(genders[g2]["event"] == events_list_m[j])]

                # Mean/event/year - i
                means_i = [] # Average i distance
                for k in range(len(years)):
                    mean_year_event_i = event_i.loc[(event_i['Date'] == years[k]), 'Mark'].mean()
                    means_i.append(mean_year_event_i)

                # Compute male returns
                returns_i = np.diff(means_i)

                # Mean/event/year - j
                means_j = [] # Average j distance
                for k in range(len(years)):
                    mean_year_event_j = event_j.loc[(event_j['Date'] == years[k]), 'Mark'].mean()
                    means_j.append(mean_year_event_j)

                # Compute male returns
                returns_j = np.diff(means_j)
                # Compute direction vector
                direction_vector_ij = np.sum(returns_i*returns_j)/(np.linalg.norm(returns_i)*np.linalg.norm(returns_j))
                # Append to list
                distance_list.append(direction_vector_ij)

# Plot distance matrix with inner product between all first differences
distance_array = np.array(distance_list)
distance_array = np.reshape(distance_array, (2*len(events_list_m), 2*len(events_list_m)))
plt.matshow(distance_array)
plt.show()

cluster_labels = ["M high jump", "W high jump", "M long jump", "W long jump", "M pole vault", "W pole vault", "M triple jump", "W triple jump",
                  "M discus", "W discus", "M hammer throw", "W hammer throw", "M javelin", "W javelin", "M shot put", "W shot put"]

dendrogram_plot_labels(distance_array, "_inner_product_", "field", cluster_labels)