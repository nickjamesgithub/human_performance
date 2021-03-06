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

# Top 1/10/100
top = 10 # 1/10/100

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
params_list = []

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
            if top == 1:
                year_event_top1 = event_i.loc[(event_i['Date_Y'] == years[j]), 'Mark_seconds'][:1]
                mean_year_event = year_event_top1
                means.append(mean_year_event)

        # Slice response variable and two predictors
        y = np.array(means).reshape(-1,1)
        x1 = np.reshape(np.linspace(1,19,19), (len(years),1)) # Linear
        x1_ones = sm.tools.tools.add_constant(x1)
        x2 = np.reshape(generate_olympic_data(x1), (19,1)) # Indicator
        x3 = np.cos(np.pi / 2 * x1) # Periodic cos wave

        # Combinations of features
        linear_indicator = np.concatenate((x1,x2),axis=1) # linear + indicator
        linear_periodic = np.concatenate((x1,x3),axis=1) # Linear + periodic
        linear_indicator_periodic = np.concatenate((x1,x2,x3),axis=1) # Linear + periodic + indicator

        # Add column of ones
        linear_indicator_ones = sm.tools.tools.add_constant(linear_indicator)
        linear_periodic_ones = sm.tools.tools.add_constant(linear_periodic)
        linear_indicator_periodic_ones = sm.tools.tools.add_constant(linear_indicator_periodic)
        periodic_ones = sm.tools.tools.add_constant(x3)

        # relabel
        label = re.sub('[!@#$\/]', '', events_list_m[i])

        # Model 1 statsmodels: linear
        model1 = sm.OLS(y, x1_ones)  # Linear term
        results1 = model1.fit()
        # AIC/BIC/Adjusted R2
        m1_aic = results1.aic
        m1_bic = results1.bic
        m1_r2a = results1.rsquared_adj
        m1_pvals = results1.pvalues

        # Model 2 statsmodels: linear + indicator
        model2 = sm.OLS(y, linear_indicator_ones)  # Linear + indicator
        results2 = model2.fit()
        # AIC/BIC/Adjusted R2
        m2_aic = results2.aic
        m2_bic = results2.bic
        m2_r2a = results2.rsquared_adj
        m2_pvals = results2.pvalues
        m2_params = results2.params

        # Residual vs predictor
        y_pred = results2.fittedvalues
        residuals = np.reshape(y, (19,1)) - np.reshape(y_pred, (19,1))

        # Residuals plot
        years = years.astype("int")
        plt.scatter(years, residuals, label='residuals', color='red')
        m, b = np.polyfit(years, residuals, 1) # Fit line of best fit parameters
        plt.plot(years, m*years + b, label='trend') # Annotate line of best fit
        plt.xlabel("Years")
        plt.ylabel("Model residuals")
        plt.ylim(-6,6)
        plt.locator_params(axis='x', nbins=4)
        plt.legend()
        # plt.savefig(label + gender_labels[g] + "_" + str(top)+"_Residuals")
        # plt.show()

        # Homoscedasticity plot
        years = years.astype("int")
        plt.scatter(y_pred, residuals, color='red', label='residuals')
        m, b = np.polyfit(years, residuals, 1) # Fit line of best fit parameters
        plt.plot(y_pred, m*years + b, label='trend') # Annotate line of best fit
        plt.xlabel("Fitted value")
        plt.ylabel("Model residuals")
        plt.ylim(-6,6)
        plt.locator_params(axis='x', nbins=4)
        plt.legend()
        # plt.savefig(label + gender_labels[g] + "_" + str(top)+"_Homoscedasticity")
        # plt.show()

        # Model 3 statsmodels: linear + periodic
        model3 = sm.OLS(y, linear_periodic_ones)  # linear + periodic
        results3 = model3.fit()
        # AIC/BIC/Adjusted R2
        m3_aic = results3.aic
        m3_bic = results3.bic
        m3_r2a = results3.rsquared_adj
        m3_pvals = results3.pvalues
        #
        # # Model 4 statsmodels: linear + periodic
        # model4 = sm.OLS(y, linear_indicator_periodic_ones)  # Linear + periodic + indicator
        # results4 = model4.fit()
        # # AIC/BIC/Adjusted R2
        # m4_aic = results4.aic
        # m4_bic = results4.bic
        # m4_r2a = results4.rsquared_adj
        # m4_pvals = results4.pvalues
        #
        # # Model 5 statsmodels: linear + periodic
        # model5 = sm.OLS(y, periodic_ones)  # Periodic function
        # results5 = model5.fit()
        # # AIC/BIC/Adjusted R2
        # m5_aic = results5.aic
        # m5_bic = results5.bic
        # m5_r2a = results5.rsquared_adj
        # m5_pvals = results5.pvalues

        # Summary of results
        print(label + gender_labels[g], results1.summary())

        # Plotting grid
        # Model 1, Model 2, Model 3, Model 4 fit (statsmodels)
        x1 = x1.astype("int")
        fig, ax = plt.subplots()
        plt.plot(years, results2.fittedvalues, label="model 2", alpha=0.6)
        # plt.plot(x1, results3.fittedvalues, label="model 3", alpha=0.4)
        # plt.plot(x1, results4.fittedvalues, label="model 4", alpha=0.4)
        # plt.plot(x1, results5.fittedvalues, label="model 5", alpha=0.4)
        plt.scatter(years, y, label="data", color='red')
        plt.locator_params(axis='x', nbins=4)
        plt.ylabel("Time in seconds")
        plt.xlabel("Date")
        plt.legend()
        # plt.title(events_list_m[i] + "_" + gender_labels[g] + "_" + str(top))
        plt.savefig(label + gender_labels[g] + "_" + str(top))
        plt.show()

        # Append AIC/BIC/Adjusted R^2/p values to list
        AIC_list.append([events_list_m[i] + "_" + gender_labels[g], m2_aic])
        BIC_list.append([events_list_m[i] + "_" + gender_labels[g], m2_bic])
        # r2_list.append([events_list_m[i] + "_" + gender_labels[g], m1_r2a, m2_r2a])
        # pvals_list.append([events_list_m[i] + "_" + gender_labels[g], m1_pvals, m2_pvals, m3_pvals, m4_pvals, m5_pvals])
        params_list.append([events_list_m[i] + "_" + gender_labels[g], m2_params])

        # Print results from each model
        # print("Model 1", results1.summary())
        # print("Model 2", events_list_m[i] + "_" + gender_labels[g], results2.summary())
        # print("Model 3", results3.summary())
        # print("Model 4", results4.summary())
        # print("Model 5", results5.summary())

# Print RMSE and R2
print(AIC_list)
print(BIC_list)
print(params_list)
# print(r2_list)
# print(pvals_list)