import pandas as pd
import glob
import datetime
import numpy as np
import matplotlib.pyplot as plt

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
# frame_sp = frame_sp.dropna(subset=['Date']) # Drop N/As on Date

# Change date format to just year %YYYY
# frame_sp['Date'] = frame_sp['Date'].astype(str).str.extract('(\d{2})').astype(int)
# frame_sp['Date'] = frame_sp['Date'].datetime.strptime("%d/%m/%Y").strftime("%Y")
frame_sp['Date_Y'] = frame_sp['Date'].str[-2:]
# lastconnection = datetime.strptime("21/12/2008", "%d/%m/%Y").strftime('%Y-%m-%d')
block = 1