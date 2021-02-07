# Hide warnings
import warnings
warnings.filterwarnings('ignore')

# Import libraries
import pickle
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly 
import plotly.express as px
import math
import numpy as np
import time

plt.style.use('fivethirtyeight')

def get_pickle_data(data_folder):
    data_dict = {}
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".pickle"):
                with open(data_folder+file, "rb") as f:
                    data = pickle.load(f)
                data_dict[file] = data
        
    print(f"data files in {data_folder}: {data_dict.keys()} \n")
    print(f"Total files: {len(data_dict.keys())} \n")
    return data_dict

historical_dict = {}

# Get data from historical data folder
historical_folder = "Data/historical_data/"
historical_dict = get_pickle_data(historical_folder)

''' Processing race information in RA file '''
race_detail_df = historical_dict["RA.pickle"]

# combine "id$Year", "id$MonthDay", "id$JyoCD", "id$Kaiji", "id$Nichiji", "id$RaceNum" cols to have "id" column in the dataset
race_id_cols = ["id$Year", "id$MonthDay", "id$JyoCD", "id$Kaiji", "id$Nichiji", "id$RaceNum"]
race_id_df = race_detail_df[race_id_cols]
race_id_df = race_id_df.astype("str")
race_id_df["id$MonthDay"] = race_id_df["id$MonthDay"].apply(lambda x: "0"+x if len(x) == 3 else x)
race_id_df["id$Nichiji"] = race_id_df["id$Nichiji"].apply(lambda x: "0"+x if len(x) == 1 else x)
race_id_df["id$RaceNum"] = race_id_df["id$RaceNum"].apply(lambda x: "0"+x if len(x) == 1 else x)

# add race_id to race_detail dataset
race_detail_df["race_id"] = race_id_df["id$Year"] + race_id_df["id$MonthDay"] + race_id_df["id$JyoCD"] \
                    + race_id_df["id$Kaiji"] + race_id_df["id$Nichiji"] + race_id_df["id$RaceNum"]

# Get useful features for prediction
# id: race id
# Kyori: total distance (meter)
# TrackCD: track code, type of racetrack
# TenkoBaba$SibaBabaCD: turf condition. 0: unknown, 1: firm, 2: good, 3: yielding, 4: soft
# TenkoBaba$DirtBabaCD: dirt condition. 0: unknown, 1: standard, 2: good, 3: muddy, 4: sloppy

race_info_cols = ["race_id", "Kyori", "TrackCD", "TenkoBaba$SibaBabaCD", "TenkoBaba$DirtBabaCD"]
race_df = race_detail_df[race_info_cols]
race_df.rename(columns={"Kyori": "distance(m)", "TrackCD": "racetrack_type",
                        "TenkoBaba$SibaBabaCD": "turf_condition", "TenkoBaba$DirtBabaCD": "dirt_condition"}, inplace=True)

''' Processing horse information in SE file '''
horse_info_df = historical_dict["SE.pickle"]