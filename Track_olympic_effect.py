import pandas as pd
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import re
from Utilities import generate_olympic_data

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

# Model parameters
params_list = []
rmse_list = []
r2_list = []

for g in range(len(genders)):
    for i in range(len(events_list_m)):
        # Slice a particular event
        event_i = genders[g][(genders[g]["event"] == events_list_m[i])]

        # Mean/event/year
        means = []
        for j in range(len(years)):
            mean_year_event = event_i.loc[(event_i['Date_Y'] == years[j]), 'Mark_seconds'].mean()
            means.append(mean_year_event)

        # # Remove Years 2020 and 2021
        # index_2020 = event_i[(event_i['Date'] == 2020)].index
        # index_2021 = event_i[(event_i['Date'] == 2021)].index
        # # Delete these row indexes from dataFrame
        # event_i_2019 = event_i.drop([index_2020[0], index_2021[0]])

        # Slice response variable and two predictors
        y = np.array(means).reshape(-1,1)
        x1 = np.reshape(np.linspace(2001,2019,19), (len(years),1))
        x2 = np.reshape(generate_olympic_data(x1), (19,1))
        combined = np.concatenate((x1,x2),axis=1)

        # Model 1
        ols1 = LinearRegression(fit_intercept=True)
        ols1.fit(x1, y)
        int_1 = ols1.intercept_
        coeffs_1 = ols1.coef_
        m1_params = [int_1[0], coeffs_1[0][0]]
        y_pred_1 = ols1.predict(x1)

        # Model 2
        ols2 = LinearRegression(fit_intercept=True)
        ols2.fit(combined, y)
        int_2 = ols2.intercept_
        coeffs_2 = ols2.coef_
        m2_params = [int_2[0], coeffs_2[0][0], coeffs_2[0][1]]
        y_pred_2 = ols2.predict(combined)

        # relabel
        label = re.sub('[!@#$\/]', '', events_list_m[i])

        # plot of regression
        plt.scatter(x1, means)
        plt.plot(x1, y_pred_1, label="Model 1", color="red")
        plt.plot(x1, y_pred_2, label="Model 2", color="blue")
        plt.scatter(x1, y, color="black", alpha=0.25)
        plt.title(events_list_m[i]+"_"+gender_labels[g])
        plt.legend()
        plt.savefig(label+gender_labels[g])
        plt.show()

        # Compute RMSE and R2
        rmse_m1 = np.sqrt(mean_squared_error(means, y_pred_1))
        rmse_m2 = np.sqrt(mean_squared_error(means, y_pred_2))
        r2_m1 = r2_score(means, y_pred_1)
        r2_m2 = r2_score(means, y_pred_2)

        # Append to lists
        rmse_list.append([events_list_m[i]+"_"+gender_labels[g], rmse_m1, rmse_m2])
        r2_list.append([events_list_m[i]+"_"+gender_labels[g], r2_m1, r2_m2])
        params_list.append([m1_params, m2_params])

# Print RMSE and R2
print(rmse_list)
print(r2_list)
print(params_list)