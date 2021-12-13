import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Import datasets
data_200 = pd.read_csv("/Users/tassjames/Desktop/Olympic_data/scatter_plot/200m.csv")
data_shotput = pd.read_csv("/Users/tassjames/Desktop/Olympic_data/scatter_plot/shotput.csv")

# Plot scatterplot 200 metres
grid = np.linspace(2001,2019,19)
men_counts_200 = np.array(data_200.iloc[0:19,2])
women_counts_200 = np.array(data_200.iloc[21:40,2])

# Plot scatterplot shotput
men_counts_sp = np.array(data_shotput.iloc[0:19,2])
women_counts_sp = np.array(data_shotput.iloc[21:40,2])

date_grid = np.linspace(2001,2019,19)
men_200_m, men_200_b = np.polyfit(grid, men_counts_200, 1)
women_200_m, women_200_b = np.polyfit(grid, women_counts_200, 1)

# Men/women scatter 200 metres
plt.scatter(grid, men_counts_200, label="Men", color="blue")
plt.plot(grid, men_200_m * date_grid + men_200_b, color='blue', alpha=0.7)
plt.scatter(grid, women_counts_200, label="Women", color="red")
plt.plot(grid, women_200_m * date_grid + women_200_b, color='red', alpha=0.7)
plt.ylabel("Counts")
plt.xlabel("Date")
plt.locator_params(axis='x', nbins=5)
plt.legend()
plt.savefig("Scatter_200_metres_USA")
plt.show()

men_sp_m, men_sp_b = np.polyfit(grid, men_counts_sp, 1)
women_sp_m, women_sp_b = np.polyfit(grid, women_counts_sp, 1)

# Men/women scatter 200 metres
plt.scatter(grid, men_counts_sp, label="Men", color="blue")
plt.plot(grid, men_sp_m * date_grid + men_sp_b, color='blue', alpha=0.7)
plt.scatter(grid, women_counts_sp, label="Women", color="red")
plt.plot(grid, women_sp_m * date_grid + women_sp_b, color='red', alpha=0.7)
plt.ylabel("Counts")
plt.xlabel("Date")
plt.locator_params(axis='x', nbins=5)
plt.legend()
plt.savefig("Scatter_shotput_USA")
plt.show()
