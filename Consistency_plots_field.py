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

    # Mean/event/year - women
    means_f = [] # Average female distance
    for j in range(len(years)):
        mean_year_event = event_f.loc[(event_f['Date'] == years[j]), 'Mark'].mean()
        means_f.append(mean_year_event)

    # Compute trajectories
    trajectory_m = means_m/np.sum(means_m)
    trajectory_f = means_f / np.sum(means_f)

    # relabel
    label = re.sub('[!@#$\/]', '', events_list_w[i])

    # Compute linear model components for men and women
    men_m, men_b = np.polyfit(years, trajectory_m, 1)
    women_m, women_b = np.polyfit(years, trajectory_f, 1)

    # Plot of men and women
    fig, ax = plt.subplots()
    plt.scatter(years, trajectory_m, color='blue', alpha=0.7, label="Men")
    plt.plot(years, men_m * years + men_b, color='blue', alpha=0.7)
    plt.scatter(years, trajectory_f, color='red', alpha=0.7, label="Women")
    plt.plot(years, women_m * years + women_b, color='red', alpha=0.7)
    plt.locator_params(axis='x', nbins=4)
    plt.ylabel("Normalized trajectory")
    plt.xlabel("Date")
    plt.legend()
    plt.savefig("Consistency_"+label)
    plt.show()

# Print out direction vector list
print(direction_vector_list)