import statsmodels.formula.api as smf
import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from Utilities import generate_olympic_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import re
import statsmodels.api as sm

# Turn plots on/off with True/False
make_plots = True # True/False

# Top 10/100 athletes
top = 100 # 10/100

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

# Loop over years of analysis
years = np.linspace(2001,2019,19)

# Lists
basic_params_list = []
basic_pvalues_list = []

interaction_params_list = []
interaction_pvalues_list = []

for i in range(len(events_list_m)):
    # Slice a particular event
    event_m = genders[0][(genders[0]["event"] == events_list_m[i])]
    event_w = genders[1][(genders[1]["event"] == events_list_w[i])]

    # Mean/event/year for men and women
    means_m = []
    for j in range(len(years)):
        if top == 100:
            mean_year_event_m = event_m.loc[(event_m['Date'] == years[j]), 'Mark'].mean()
            means_m.append(mean_year_event_m)

    means_w = []
    for j in range(len(years)):
        if top == 100:
            mean_year_event_w = event_w.loc[(event_w['Date'] == years[j]), 'Mark'].mean()
            means_w.append(mean_year_event_w)

    # Generating data columns
    t1 = np.reshape(np.linspace(1,19,19), (len(years),1)) # Linear
    t2 = np.reshape(np.linspace(1, 19, 19), (len(years), 1))  # Linear
    o1 = np.reshape(generate_olympic_data(t1), (19,1)) # Olympic Indicator
    o2 = np.reshape(generate_olympic_data(t2), (19, 1))  # Olympic Indicator
    g1 = np.ones(len(years))
    g2 = np.zeros(len(years))

    # Flatten and combine all features
    y = np.array(means_m + means_w).reshape(38,1)
    intercept = np.ones(len(years)*2).reshape(38,1)
    t = np.concatenate((t1,t2),axis=0).reshape(38,1)
    o = np.concatenate((o1,o2),axis=0).reshape(38,1)
    g = np.concatenate((g1,g2),axis=0).reshape(38,1)

     # Full data matrix
    data_matrix = np.concatenate((y,intercept,t,o,g),axis=1)
    data_matrix_df = pd.DataFrame(data_matrix)
    # Generate dataframe
    data_matrix_df.columns = ['y', 'intercept', 'time', 'olympic', 'gender']
    model_basic = smf.ols(formula='y ~ time + olympic + gender', data=data_matrix_df)
    model_interaction = smf.ols(formula='y ~ time + olympic + gender + time : gender + gender : olympic', data=data_matrix_df)

    # Model results
    result_basic = model_basic.fit()
    result_interaction = model_interaction.fit()

    # Model results
    print(result_basic.summary())
    print(result_interaction.summary())

    # relabel
    label = re.sub('[!@#$\/]', '', events_list_m[i])

    # Append parameter values and p-values for each event: basic and interaction
    basic_params = result_basic.params
    basic_pvals = result_basic.pvalues
    basic_params_list.append(basic_params)
    basic_pvalues_list.append(basic_pvals)

    interaction_params = result_interaction.params
    interaction_pvals = result_interaction.pvalues
    interaction_params_list.append(interaction_params)
    interaction_pvalues_list.append(interaction_pvals)

    if make_plots:
        # Men
        plt.scatter(years, means_m, label="data", color='black')
        plt.plot(years, result_basic.fittedvalues[0:19], label="Basic model")
        plt.plot(years, result_interaction.fittedvalues[0:19], label="Interaction model")
        plt.ylabel("Distance in metres")
        plt.xlabel("Date")
        plt.legend()
        plt.locator_params(axis='x', nbins=4)
        plt.savefig("interaction_"+label + "_Men_")
        plt.show()

        # Women
        plt.scatter(years, means_w, color='black')
        plt.plot(years, result_basic.fittedvalues[19:38], label="basic model")
        plt.plot(years, result_interaction.fittedvalues[19:38], label="interaction model")
        plt.ylabel("Distance in metres")
        plt.xlabel("Date")
        plt.legend()
        plt.locator_params(axis='x', nbins=4)
        plt.savefig("interaction_"+label + "_Women")
        plt.show()

print(basic_params_list)
# # Print lists
basic_params_df = pd.DataFrame(basic_params_list)
basic_pvals_df = pd.DataFrame(basic_pvalues_list)
basic_params_df.to_csv("/Users/tassjames/Desktop/Olympic_data/Howard_interaction/basic_params_field.csv")
basic_pvals_df.to_csv("/Users/tassjames/Desktop/Olympic_data/Howard_interaction/basic_pvals_field.csv")

interaction_params_df = pd.DataFrame(interaction_params_list)
interaction_pvals_df = pd.DataFrame(interaction_pvalues_list)
interaction_params_df.to_csv("/Users/tassjames/Desktop/Olympic_data/Howard_interaction/interaction_params_field.csv")
interaction_pvals_df.to_csv("/Users/tassjames/Desktop/Olympic_data/Howard_interaction/interaction_pvals_field.csv")