import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from Utilities import generate_olympic_data, indicator_function
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import re
import statsmodels.api as sm

# Turn plots on/off with True/False
make_plots = False # True/False

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
                mean_year_event = event_i.loc[(event_i['Date'] == years[j]), 'Mark'].mean()
                means.append(mean_year_event)
            if top == 10:
                year_event_top10 = event_i.loc[(event_i['Date'] == years[j]), 'Mark'][:10]
                mean_year_event = year_event_top10.mean()
                means.append(mean_year_event)

        # Slice response variable and two predictors
        y = np.array(means).reshape(-1,1)
        x1 = np.reshape(np.linspace(2001,2019,19), (len(years),1)) # Linear # Indicator
        # Combinations of features
        x2_indicator = np.reshape([0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0], (19,1))
        x3_indicator = np.reshape([0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1], (19,1))
        x4_indicator = np.reshape([0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,0,0,0], (19,1))
        multiple_indicator = np.concatenate((x1, x2_indicator, x3_indicator, x4_indicator), axis=1)  # linear + indicato
        multiple_indicator_ones = sm.tools.tools.add_constant(multiple_indicator)

        # Model 1 statsmodels: linear
        model1 = sm.OLS(y, multiple_indicator_ones) # Linear term
        results1 = model1.fit()
        # AIC/BIC/Adjusted R2
        m1_aic = results1.aic
        m1_bic = results1.bic
        m1_r2a = results1.rsquared_adj
        m1_pvals = results1.pvalues

        # relabel
        label = re.sub('[!@#$\/]', '', events_list_m[i])

        # Model 1, Model 2, Model 3, Model 4 fit (statsmodels)
        plt.plot(x1, results1.fittedvalues, label="model 1", alpha=0.4)
        plt.scatter(x1, y, label="data")
        plt.legend()
        plt.title(events_list_m[i]+"_"+gender_labels[g] + "_" + str(top))
        plt.savefig(label + gender_labels[g] + "_" + str(top))
        plt.show()

        # Append AIC/BIC/Adjusted R^2/p values to list
        AIC_list.append([events_list_m[i] + "_" + gender_labels[g], m1_aic])
        BIC_list.append([events_list_m[i] + "_" + gender_labels[g], m1_bic])
        r2_list.append([events_list_m[i] + "_" + gender_labels[g], m1_r2a])
        pvals_list.append([events_list_m[i] + "_" + gender_labels[g], m1_pvals])

        # Print results from each model
        print("Model 1", results1.summary())

# Print RMSE and R2
print(AIC_list)
print(BIC_list)
print(r2_list)
print(pvals_list)


