import pandas as pd
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt

make_trajectories_run = False

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
years = np.linspace(2001,2019,19)
years = years.astype("int")

model = "l1_best" # l1_best, mean_variance

if model == "l1_best":
    # Grab best performance of each year
    # selecting best male performances
    best_performance = frame_sp[(frame_sp['Rank'] == 1)]
    men_best_performance = frame_sp[(frame_sp['gender'] == "men")]
    women_best_performance = frame_sp[(frame_sp['gender'] == "women")]

    # Get event lists
    events_list_m = men_best_performance["event"].unique()
    events_list_w = women_best_performance["event"].unique()
    events_list_m = np.sort(events_list_m)
    events_list_w = np.sort(events_list_w)

    if make_trajectories_run:
        # Drop duplicates of best performance
        normalized_trajectories = []
        for i in range(len(events_list_m)):
            # Compute L^1 norm of event i
            event_i = men_best_performance[(men_best_performance["event"] == events_list_m[i])]
            # Mean/event/year - men
            means_i = []  # Average Male distance
            for k in range(len(years)):
                mean_year_event = event_i.loc[(event_i['Date_Y'] == years[k]), 'Mark_seconds'].mean()
                means_i.append(mean_year_event)
            norm_mark_i = means_i / np.sum(np.abs(means_i))
            normalized_trajectories.append(norm_mark_i)

    # Loop over all events, get time series L^1 normalize and determine distance trajectory
    male_distance_matrix = np.zeros((len(events_list_m),len(events_list_m)))
    normalized_trajectories = []
    for i in range(len(events_list_m)):
        for j in range(len(events_list_m)):
            # Compute L^1 norm of event i
            event_i = men_best_performance[(men_best_performance["event"] == events_list_m[i])]
            means_i = []  # Average Male distance
            for k in range(len(years)):
                mean_year_event = event_i.loc[(event_i['Date_Y'] == years[k]), 'Mark_seconds'].mean()
                means_i.append(mean_year_event)
            norm_mark_i = means_i / np.sum(np.abs(means_i))
            normalized_trajectories.append(norm_mark_i)

            # Compute L^1 norm of event j
            event_j = men_best_performance[(men_best_performance["event"] == events_list_m[j])]
            means_j = []  # Average i distance
            for k in range(len(years)):
                mean_year_event_j = event_j.loc[(event_j['Date_Y'] == years[k]), 'Mark_seconds'].mean()
                means_j.append(mean_year_event_j)
            norm_mark_j = means_j / np.sum(np.abs(means_j))

            # Compute distance between vectors
            difference = np.sum(np.abs(norm_mark_i - norm_mark_j))
            male_distance_matrix[i,j] = np.sum(np.abs(norm_mark_i - norm_mark_j))

    # Loop over all events, get time series L^1 normalize and determine distance trajectory
    female_distance_matrix = np.zeros((len(events_list_w),len(events_list_w)))
    normalized_trajectories = []
    for i in range(len(events_list_w)):
        for j in range(len(events_list_w)):
            # Compute L^1 norm of event i
            event_i = women_best_performance[(women_best_performance["event"] == events_list_w[i])]
            means_i = []  # Average i distance
            for k in range(len(years)):
                mean_year_event_i = event_i.loc[(event_i['Date_Y'] == years[k]), 'Mark_seconds'].mean()
                means_i.append(mean_year_event_i)
            norm_mark_i = means_i / np.sum(np.abs(means_i))
            normalized_trajectories.append(norm_mark_i)

            # Compute L^1 norm of event j
            event_j = women_best_performance[(women_best_performance["event"] == events_list_w[j])]
            means_j = []  # Average i distance
            for k in range(len(years)):
                mean_year_event_j = event_j.loc[(event_j['Date_Y'] == years[k]), 'Mark_seconds'].mean()
                means_j.append(mean_year_event_j)
            norm_mark_j = means_j / np.sum(np.abs(means_j))

            # Compute distance between vectors
            difference = np.sum(np.abs(norm_mark_i - norm_mark_j))
            female_distance_matrix[i,j] = np.sum(np.abs(norm_mark_i - norm_mark_j))

    # Total (normalized) norm for men/women
    female_norm = 1/len(female_distance_matrix)**2 * np.sum(np.abs(female_distance_matrix))
    male_norm = 1/len(male_distance_matrix)**2 * np.sum(np.abs(male_distance_matrix))
    print("Norm Female", female_norm)
    print("Norm Male", male_norm)

    # Compute affinity matrices and do cross contextual analysis
    aff_male = 1 - male_distance_matrix/np.max(male_distance_matrix)
    aff_female = 1 - female_distance_matrix/np.max(female_distance_matrix)
    field_consistency = np.abs(aff_male - aff_female)

    # Plot Male distance matrix
    plt.matshow(aff_male)
    plt.colorbar()
    plt.title("Male_distance_matrix")
    plt.show()

    # Plot female distance matrix
    plt.matshow(aff_female)
    plt.colorbar()
    plt.title("Female_distance_matrix")
    plt.show()

    # Plot of consistency
    plt.matshow(field_consistency)
    plt.colorbar()
    plt.title("Consistency_Matrix")
    plt.show()

    # Compute consistency scores for each sport
    anomaly_scores = []
    for i in range(len(events_list_m)):
        anomaly_i = np.sum(field_consistency[i,:])
        anomaly_scores.append([anomaly_i, events_list_m[i]])

    # Make it an array Anomaly scores
    anomaly_scores = np.array(anomaly_scores)
    anomaly_ordered = anomaly_scores[anomaly_scores[:, 0].argsort()]
    print(anomaly_ordered)
