import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt

path = '/Users/tassjames/Desktop/Olympic_data/olympic_data/field' # use your path
all_files = glob.glob(path + "/*.csv")

model = "variance" #l1_best, variance

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

if model == "l1_best":
    # Grab best performance of each year
    # selecting best male performances
    best_performance = frame_sp[(frame_sp['Rank'] == 1)]
    men_best_performance = frame_sp[(frame_sp['Rank'] == 1) & (frame_sp['gender'] == "men")]
    women_best_performance = frame_sp[(frame_sp['Rank'] == 1) & (frame_sp['gender'] == "women")]

    # Get event lists
    events_list_m = men_best_performance["event"].unique()
    events_list_w = women_best_performance["event"].unique()
    events_list = ['discus', 'high jump', 'shot put', 'triple jump', 'pole vault', 'long jump', 'javelin', 'hammer throw']

    # Drop duplicates of best performance
    normalized_trajectories = []
    for i in range(len(events_list_m)):
        # Compute L^1 norm of event i
        event_i = men_best_performance[(men_best_performance["event"] == events_list_m[i])]
        event_i.drop_duplicates(subset=['Date'], keep='first', inplace=True)
        mark_i = np.array(event_i["Mark"])
        norm_mark_i = mark_i/np.sum(np.abs(mark_i))
        normalized_trajectories.append(norm_mark_i)

    # Plot normalized trajectories
    for i in range(len(normalized_trajectories)):
        plt.plot(normalized_trajectories[i])
    plt.show()

    # Loop over all events, get time series L^1 normalize and determine distance trajectory
    male_distance_matrix = np.zeros((len(events_list_m),len(events_list_m)))
    normalized_trajectories = []
    for i in range(len(events_list_m)):
        for j in range(len(events_list_m)):
            # Compute L^1 norm of event i
            event_i = men_best_performance[(men_best_performance["event"] == events_list_m[i])]
            event_i.drop_duplicates(subset=['Date'], keep='first', inplace=True)
            mark_i = np.array(event_i["Mark"])
            norm_mark_i = mark_i/np.sum(np.abs(mark_i))

            # Compute L^1 norm of event j
            event_j = men_best_performance[(men_best_performance["event"] == events_list_m[j])]
            event_j.drop_duplicates(subset=['Date'], keep='first', inplace=True)
            mark_j = np.array(event_j["Mark"])
            norm_mark_j = mark_j / np.sum(np.abs(mark_j))
            print(event_j)
            print(len(mark_j))

            # Compute distance between vectors
            difference = np.sum(np.abs(norm_mark_i - norm_mark_j))
            male_distance_matrix[i,j] = np.sum(np.abs(norm_mark_i - norm_mark_j))

    # Plot Male distance matrix
    plt.matshow(male_distance_matrix)
    plt.show()

    # Loop over all events, get time series L^1 normalize and determine distance trajectory
    female_distance_matrix = np.zeros((len(events_list_w),len(events_list_w)))
    normalized_trajectories = []
    for i in range(len(events_list_w)):
        for j in range(len(events_list_w)):
            # Compute L^1 norm of event i
            event_i = women_best_performance[(women_best_performance["event"] == events_list_w[i])]
            event_i.drop_duplicates(subset=['Date'], keep='first', inplace=True)
            mark_i = np.array(event_i["Mark"])
            norm_mark_i = mark_i/np.sum(np.abs(mark_i))

            # Compute L^1 norm of event j
            event_j = women_best_performance[(women_best_performance["event"] == events_list_w[j])]
            event_j.drop_duplicates(subset=['Date'], keep='first', inplace=True)
            mark_j = np.array(event_j["Mark"])
            norm_mark_j = mark_j / np.sum(np.abs(mark_j))
            print(event_j)
            print(len(mark_j))

            # Compute distance between vectors
            difference = np.sum(np.abs(norm_mark_i - norm_mark_j))
            female_distance_matrix[i,j] = np.sum(np.abs(norm_mark_i - norm_mark_j))

    plt.matshow(female_distance_matrix)
    plt.show()

    # Total (normalized) norm for men/women
    female_norm = 1/len(female_distance_matrix)**2 * np.sum(np.abs(female_distance_matrix))
    male_norm = 1/len(male_distance_matrix)**2 * np.sum(np.abs(male_distance_matrix))
    print("Norm Female", female_norm)
    print("Norm Male", male_norm)

    # Compute affinity matrices and do cross contextual analysis
    aff_male = 1 - male_distance_matrix/np.max(male_distance_matrix)
    aff_female = 1 - female_distance_matrix/np.max(female_distance_matrix)
    field_consistency = np.abs(aff_male - aff_female)

    # Compute consistency scores for each sport
    anomaly_scores = []
    for i in range(len(field_consistency)):
        anomaly_i = np.sum(field_consistency[i,:])
        anomaly_scores.append([anomaly_i, events_list[i]])

    # Make it an array Anomaly scores
    anomaly_scores = np.array(anomaly_scores)
    anomaly_ordered = anomaly_scores[anomaly_scores[:, 0].argsort()]
    print(anomaly_ordered)


if model == "variance":
    years = np.linspace(2001,2021,21)
    variance_event_year_men = []
    for i in range(len(years)):
        men_year = frame_sp[(frame_sp['gender'] == "men") & (frame_sp['Date'] == years[i])]
        women_year = frame_sp[(frame_sp['gender'] == "women") & (frame_sp['Date'] == years[i])]
        events_list = men_year["event"].unique()
        events_name = ['discus', 'high jump', 'shot put', 'triple jump', 'pole vault', 'long jump', 'javelin',
                       'hammer throw']
        variance_list = []
        for j in range(len(events_list)): # For each year, append the variance of performance in each sport
            men_event = men_year[(men_year['event'] == events_list[j])]
            mark = men_event["Mark"]
            marks_var = np.var(mark)
            variance_list.append(marks_var)
        variance_event_year_men.append(variance_list)

    variance_event_array = np.array(variance_event_year_men)
    # Plot variance of all sports
    for i in range(len(variance_event_array[0])):
        plt.plot(years, variance_event_array[:,i])
        plt.title(events_list[i])
        plt.show()
