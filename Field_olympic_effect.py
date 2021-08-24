import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt
from Utilities import generate_olympic_data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import re
import statsmodels.api as sm

# Top 10/100
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
rmse_list = []
r2_list = []
params_list = []
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
        x1 = np.reshape(np.linspace(2001,2019,19), (len(years),1))
        x1_ones = sm.tools.tools.add_constant(x1)
        x2 = np.reshape(generate_olympic_data(x1), (19,1))
        combined = np.concatenate((x1,x2),axis=1)
        combined_ones = sm.tools.tools.add_constant(combined)

        # Model 1
        ols1 = LinearRegression(fit_intercept=True)
        ols1.fit(x1, y)
        int_1 = ols1.intercept_
        coeffs_1 = ols1.coef_
        m1_params = [int_1[0], coeffs_1[0][0]]
        y_pred_1 = ols1.predict(x1)

        # Model 1 statsmodels
        model1 = sm.OLS(y, x1_ones)
        results1 = model1.fit()
        print(results1.summary())
        print("results parameters", results1.params)
        print("AIC", results1.aic)
        print("BIC", results1.bic)
        print("Adjusted R2", results1.rsquared_adj)

        # Model 2
        ols2 = LinearRegression(fit_intercept=True)
        ols2.fit(combined, y)
        int_2 = ols2.intercept_
        coeffs_2 = ols2.coef_
        m2_params = [int_2[0], coeffs_2[0][0], coeffs_2[0][1]]
        y_pred_2 = ols2.predict(combined)

        # Model 2 statsmodels
        model2 = sm.OLS(y, combined_ones)
        results2 = model2.fit()
        print(results2.summary())
        print("results parameters", results2.params)
        print("AIC", results2.aic)
        print("BIC", results2.bic)
        print("Adjusted R2", results2.rsquared_adj)

        # relabel
        label = re.sub('[!@#$\/]', '', events_list_m[i])

        # Model 1 and Model 2 fit (statsmodels)
        plt.plot(x1, results1.fittedvalues, label="model 1")
        plt.plot(x1, results2.fittedvalues, label="model 2")
        plt.scatter(x1, y, label="data")
        plt.legend()
        plt.savefig(label + gender_labels[g] + "_" + str(top)+"_SM")
        plt.show()

        # plot of regression
        plt.scatter(x1, means)
        plt.plot(x1, y_pred_1, label="Model 1", color="red")
        plt.plot(x1, y_pred_2, label="Model 2", color="blue")
        plt.scatter(x1, y, color="black", alpha=0.25)
        plt.title(events_list_m[i]+"_"+gender_labels[g]+"_"+str(top))
        plt.legend()
        plt.savefig(label+gender_labels[g]+"_"+str(top))
        plt.show()

        # Compute RMSE and R2
        rmse_m1 = np.sqrt(mean_squared_error(means, y_pred_1))
        rmse_m2 = np.sqrt(mean_squared_error(means, y_pred_2))
        r2_m1 = r2_score(means, y_pred_1)
        r2_m2 = r2_score(means, y_pred_2)

        # Append to lists
        rmse_list.append([events_list_m[i] + "_" + gender_labels[g], rmse_m1, rmse_m2])
        r2_list.append([events_list_m[i] + "_" + gender_labels[g], r2_m1, r2_m2])
        params_list.append([events_list_m[i] + "_" + gender_labels[g], m1_params, m2_params])

# Print RMSE and R2
print(rmse_list)
print(r2_list)
print(params_list)


