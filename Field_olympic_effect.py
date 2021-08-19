import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fftshift, fft
from Utilities import generate_olympic_data
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import re

path = '/Users/tassjames/Desktop/Olympic_data/olympic_data/field' # use your path
all_files = glob.glob(path + "/*.csv")

model = "l1_best" #l1_best, mean_variance

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

# Get event lists
events_list_m = men_best_performance["event"].unique()
events_list_w = women_best_performance["event"].unique()
# events_list = ['discus', 'high jump', 'shot put', 'triple jump', 'pole vault', 'long jump', 'javelin', 'hammer throw']
events_list_m.sort()
events_list_w.sort()

# Loop over years of analysis
years = np.linspace(2001,2019,19)
for i in range(len(events_list_m)):
    # Slice a particular event
    event_i = men_best_performance[(men_best_performance["event"] == events_list_m[i])]

    # Mean/event/year
    means = []
    for j in range(len(years)):
        mean_year_event = event_i.loc[(event_i['Date'] == years[j]), 'Mark'].mean()
        means.append(mean_year_event)

    # # Remove Years 2020 and 2021
    # index_2020 = event_i[(event_i['Date'] == 2020)].index
    # index_2021 = event_i[(event_i['Date'] == 2021)].index
    # # Delete these row indexes from dataFrame
    # event_i_2019 = event_i.drop([index_2020[0], index_2021[0]])

    # Slice response variable and two predictors
    y = np.array(means).reshape(-1,1)
    x1 = np.reshape(np.linspace(2001,2019,19), (19,1))
    x2 = np.reshape(generate_olympic_data(x1), (19,1))
    combined = np.concatenate((x1,x2),axis=1)
    ols = LinearRegression()
    ols.fit(combined, y)
    y_pred_1 = ols.predict(combined)

    # # Fit linear regression in Python
    # ols1 = LinearRegression(fit_intercept=True)
    # ols2 = LinearRegression(fit_intercept=True)
    #
    # # Fit models
    # lr_fit_1 = ols1.fit(X=X1, y=y)
    # lr_fit_2 = ols2.fit(X=X2, y=y)
    #
    # # Prediction
    # y_pred_1 = ols1.predict(X1)
    # y_pred_2 = ols2.predict(X2)
    #
    # # Get coefficients
    # print(events_list_m[i], ols1.)
    #
    # relabel
    label = re.sub('[!@#$\/]', '', events_list_m[i])

    # plot of regression
    plt.scatter(x1, means)
    plt.plot(x1, y_pred_1)
    plt.title(events_list_m[i])
    plt.scatter(x1, y)
    # plt.savefig(label)
    plt.show()


