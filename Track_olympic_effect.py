import pandas as pd
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import re
from Utilities import generate_olympic_data
import statsmodels.api as sm

# Top 10/100
top = 100 # 10/100

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
AIC_list = []
BIC_list = []
r2_list = []
pvals_list = []

for g in range(len(genders)):
    for i in range(len(events_list_m)):
        # Slice a particular event
        event_i = genders[g][(genders[g]["event"] == events_list_m[i])]

        # Mean/event/year
        means = []
        for j in range(len(years)):
            if top == 100:
                mean_year_event = event_i.loc[(event_i['Date_Y'] == years[j]), 'Mark_seconds'].mean()
                means.append(mean_year_event)
            if top == 10:
                year_event_top10 = event_i.loc[(event_i['Date_Y'] == years[j]), 'Mark_seconds'][:10]
                mean_year_event = year_event_top10.mean()
                means.append(mean_year_event)

        # # Remove Years 2020 and 2021
        # index_2020 = event_i[(event_i['Date'] == 2020)].index
        # index_2021 = event_i[(event_i['Date'] == 2021)].index
        # # Delete these row indexes from dataFrame
        # event_i_2019 = event_i.drop([index_2020[0], index_2021[0]])

        # Slice response variable and two predictors
        y = np.array(means).reshape(-1,1)
        x1 = np.reshape(np.linspace(2001,2019,19), (len(years),1))
        x1_ones = sm.tools.tools.add_constant(x1)
        x2 = np.reshape(generate_olympic_data(x1), (19,1))
        combined = np.concatenate((x1,x2),axis=1)
        combined_ones = sm.tools.tools.add_constant(combined)

        # Model 1 statsmodels
        model1 = sm.OLS(y, x1_ones)
        results1 = model1.fit()
        # AIC/BIC/Adjusted R2
        m1_aic = results1.aic
        m1_bic = results1.bic
        m1_r2a = results1.rsquared_adj
        m1_pvals = results1.pvalues

        # Model 2 statsmodels
        model2 = sm.OLS(y, combined_ones)
        results2 = model2.fit()
        # AIC/BIC/Adjusted R2
        m2_aic = results2.aic
        m2_bic = results2.bic
        m2_r2a = results2.rsquared_adj
        m2_pvals = results2.pvalues

        # relabel
        label = re.sub('[!@#$\/]', '', events_list_m[i])

        # Model 1 and Model 2 fit (statsmodels)
        plt.plot(x1, results1.fittedvalues, label="model 1")
        plt.plot(x1, results2.fittedvalues, label="model 2")
        plt.scatter(x1, y, label="data")
        plt.legend()
        plt.title(events_list_m[i] + "_" + gender_labels[g] + "_" + str(top))
        plt.savefig(label + gender_labels[g] + "_" + str(top))
        plt.show()

        # Append AIC/BIC/Adjusted R^2/p values to list
        AIC_list.append([events_list_m[i] + "_" + gender_labels[g], m1_aic, m2_aic])
        BIC_list.append([events_list_m[i] + "_" + gender_labels[g], m1_bic, m2_bic])
        r2_list.append([events_list_m[i] + "_" + gender_labels[g], m1_r2a, m2_r2a])
        pvals_list.append([events_list_m[i] + "_" + gender_labels[g], m1_pvals, m2_pvals])

# Print RMSE and R2
print("AIC", AIC_list)
print("BIC", BIC_list)
print("R^2", r2_list)
print("p-values",pvals_list)