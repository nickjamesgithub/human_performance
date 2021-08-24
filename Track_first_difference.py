import pandas as pd
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
from Utilities import dendrogram_plot_labels
import statsmodels.api as sm
import re

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
frame_sp['Mark_seconds'] = pd.to_numeric(frame_sp['Mark_seconds'], errors='coerce')
frame_sp['Date_Y'] = pd.to_numeric(frame_sp['Date_Y'])


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
direction_vector_list = []

for i in range(len(events_list_m)):
    # Slice a particular event
    event_m = genders[0][(genders[0]["event"] == events_list_m[i])]
    event_f = genders[1][(genders[1]["event"] == events_list_w[i])]

    # Mean/event/year - men
    means_m = []  # Average Male distance
    for j in range(len(years)):
        mean_year_event = event_m.loc[(event_m['Date_Y'] == years[j]), 'Mark_seconds'].mean()
        means_m.append(mean_year_event)

    # Compute male returns
    returns_m = np.diff(means_m)

    # Mean/event/year - men
    means_f = []  # Average female distance
    for j in range(len(years)):
        mean_year_event = event_f.loc[(event_f['Date_Y'] == years[j]), 'Mark_seconds'].mean()
        means_f.append(mean_year_event)

    # Compute male returns
    returns_f = np.diff(means_f)

    # Slice response variable and two predictors
    y_m = np.array(returns_m).reshape(-1, 1)
    x1_diff = np.reshape(np.linspace(2002, 2019, 18), (len(years)-1, 1))
    x1_ones = sm.tools.tools.add_constant(x1_diff)

    # Model 1 statsmodels
    model_m = sm.OLS(y_m, x1_ones)
    results_m = model_m.fit()

    # relabel
    label = re.sub('[!@#$\/]', '', events_list_m[i])

    # Men's first difference
    plt.plot(x1_diff, results_m.fittedvalues, label="model 1")
    plt.scatter(x1_diff, y_m, label="data")
    plt.title("First_difference_"+label+"_men")
    plt.legend()
    plt.savefig("First_difference_"+label+"_men")
    plt.show()

    # Slice response variable and two predictors
    y_f = np.array(returns_f).reshape(-1, 1)
    x1_diff = np.reshape(np.linspace(2002, 2019, 18), (len(years)-1, 1))
    x1_ones = sm.tools.tools.add_constant(x1_diff)

    # Model 1 statsmodels
    model_f = sm.OLS(y_f, x1_ones)
    results_f = model_f.fit()

    # relabel
    label = re.sub('[!@#$\/]', '', events_list_w[i])

    # Women's first difference
    plt.plot(x1_diff, results_f.fittedvalues, label="model 1")
    plt.scatter(x1_diff, y_m, label="data")
    plt.title("First_difference_"+label+"_women")
    plt.legend()
    plt.savefig("First_difference_"+label+"_women")
    plt.show()


