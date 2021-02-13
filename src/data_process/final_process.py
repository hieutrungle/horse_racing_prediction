# Hide warnings
import warnings
warnings.filterwarnings('ignore')

# Import libraries
import pickle
from os.path import dirname, abspath
import os
import pandas as pd
import math
import numpy as np
import time
import json

def get_pickle_data(data_folder):
    # Read pickle files
    # return a dictionary that contains all given pickle files
    # dict.key: file name in the dicrectory
    # dict.value: all values in each file
    data_dict = {}
    for root, dirs, files in os.walk(data_folder):
        for file in files:
            if file.endswith(".pickle"):
                with open(data_folder+file, "rb") as f:
                    data = pickle.load(f)
                data_dict[file] = data
        
    print(f"data files in {data_folder}: {data_dict.keys()}")
    print(f"Total files in the given data folder: {len(data_dict.keys())}\n")
    return data_dict

# Calculate actual horse weight
def weight_calc(original_weight, sign, weight_change):
    if sign == "+":
        weight = original_weight + weight_change
    elif sign == "-":
        weight = original_weight - weight_change
    else:
        weight = original_weight
    return weight

def create_json(filepath, data):
    # Create json file
    with open(filepath, 'w') as f:
        json.dump(data, f)

def stringtify_key(dictionary):
    d = {int(k):v for k,v in dictionary.items()}
    return d

if __name__ == '__main__':
    historical_dict = {}

    # Get data from historical data folder
    current_path = os.getcwd()

    parent_path = dirname(dirname(current_path))
    print(parent_path)
    historical_folder = parent_path + "/data/historical_data/"
    historical_dict = get_pickle_data(historical_folder)

    """''''''''''''''''''''''''''''''''''''''''''''"""
    ''' Processing race information in RA file '''
    print("Process race information...")
    race_detail_df = historical_dict["RA.pickle"]

    #   *** Aggregate data to have race_id ***
    race_id_cols = ["id$Year", "id$MonthDay", "id$JyoCD", "id$Kaiji", "id$Nichiji", "id$RaceNum"]
    race_id_df = race_detail_df[race_id_cols]
    race_id_df = race_id_df.astype("str")
    race_id_df["id$MonthDay"] = race_id_df["id$MonthDay"].apply(lambda x: "0"+x if len(x) == 3 else x)
    race_id_df["id$Nichiji"] = race_id_df["id$Nichiji"].apply(lambda x: "0"+x if len(x) == 1 else x)
    race_id_df["id$RaceNum"] = race_id_df["id$RaceNum"].apply(lambda x: "0"+x if len(x) == 1 else x)

    # add race_id to race_detail dataset
    race_detail_df["race_id"] = race_id_df["id$Year"] + race_id_df["id$MonthDay"] + race_id_df["id$JyoCD"] \
                        + race_id_df["id$Kaiji"] + race_id_df["id$Nichiji"] + race_id_df["id$RaceNum"]

    #   *** Get useful features for prediction ***
    # id: race id
    # Kyori: total distance (meter)
    # TrackCD: track code, type of racetrack
    # TenkoBaba$SibaBabaCD: turf condition. 0: unknown, 1: firm, 2: good, 3: yielding, 4: soft
    # TenkoBaba$DirtBabaCD: dirt condition. 0: unknown, 1: standard, 2: good, 3: muddy, 4: sloppy

    race_info_cols = ["race_id", "Kyori", "TrackCD", "TenkoBaba$SibaBabaCD", "TenkoBaba$DirtBabaCD"]
    race_df = race_detail_df[race_info_cols]
    race_df.rename(columns={"Kyori": "distance(m)", "TrackCD": "racetrack_type",
                            "TenkoBaba$SibaBabaCD": "turf_condition", "TenkoBaba$DirtBabaCD": "dirt_condition"}, inplace=True)
    print("Finished processing race information!\n")


    """''''''''''''''''''''''''''''''''''''''''''''"""
    ''' Processing horse information in SE file '''
    print("Process horse information...")
    horse_info_df = historical_dict["SE.pickle"]

    #   *** Horse Weight Handling ***
    print("  Manipulate horse weight...")
    print("  Handle missing values of horse_weight")
    # Change type of BaTaijyu (original weight) and ZogenSa (weight change) from object to int32
    horse_info_df["ZogenSa"] = horse_info_df["ZogenSa"].apply(lambda x: "000" if x=="   " else x)
    horse_info_df[["ZogenSa", "BaTaijyu"]] = horse_info_df[["ZogenSa", "BaTaijyu"]].astype("int32")
    # Replace missing weights with median
    horse_info_df["BaTaijyu"] = horse_info_df["BaTaijyu"].apply(lambda x: horse_info_df["BaTaijyu"].median() if (x == 999 or x == 0) else x)
    # Test if the weight has been completely handled
    original_num_horse = horse_info_df["BaTaijyu"].count()
    print("  Test if missing values of horse weight has been completely handled")
    print("  The number of data in the original horse information dataset: ", original_num_horse)
    # Eliminate horses which do not have weight information
    horse_info_df = horse_info_df[(horse_info_df["BaTaijyu"] != 999) & (horse_info_df["BaTaijyu"] != 0)]
    filtered_num_horse = horse_info_df["BaTaijyu"].count()
    print("  A number of data filtered out: ", original_num_horse - filtered_num_horse)
    print(f"  The number of data in the filtered horse information dataset: {filtered_num_horse}")
    print(f"  Finished handling missing values.")


    horse_info_df["horseweight"] = 0
    horse_info_df["horseweight"] = horse_info_df.apply(
                    lambda x: weight_calc(x["BaTaijyu"], x["ZogenFugo"], x["ZogenSa"]), axis=1)
    print("Finished handling horse weight!\n")


    #   *** Aggregate data to have race_id, horse_id ***
    horse_id_cols = ["id$Year", "id$MonthDay", "id$JyoCD", "id$Kaiji", "id$Nichiji", "id$RaceNum"]
    horse_id_df = horse_info_df[horse_id_cols]
    # cast raceetrack_num from str to int32 then back to str
    horse_id_df["id$JyoCD"] = horse_id_df["id$JyoCD"].astype("int32")
    horse_id_df = horse_id_df.astype("str")
    horse_id_df["id$MonthDay"] = horse_id_df["id$MonthDay"].apply(lambda x: "0"+x if len(x) == 3 else x)
    horse_id_df["id$JyoCD"] = horse_id_df["id$JyoCD"].apply(lambda x: "0"+x if len(x) == 1 else x)
    horse_id_df["id$Nichiji"] = horse_id_df["id$Nichiji"].apply(lambda x: "0"+x if len(x) == 1 else x)
    horse_id_df["id$RaceNum"] = horse_id_df["id$RaceNum"].apply(lambda x: "0"+x if len(x) == 1 else x)

    horse_info_df["race_id"] = horse_id_df["id$Year"] + horse_id_df["id$MonthDay"] + horse_id_df["id$JyoCD"] \
                        + horse_id_df["id$Kaiji"] + horse_id_df["id$Nichiji"] + horse_id_df["id$RaceNum"]

    #   *** Create a horse_df that contains only specific data for prediction ***
    # Umaban: horse number in a race
    # id$JyoCD: racetrack code
    # Futan: carry weight
    # SexCD: horse gender. 1: male, 2: female, 3: castrated
    # Barei: horse age
    # HinsyuCD: product code, types of horses
    # KisyuCode: jockeys' code
    # BanusiCode: owners' code
    # ChokyosiCode: trainers' code
    # KakuteiJyuni: result

    # Get data that is suitable for prediction
    horse_info_cols = ["Umaban", "race_id", "KettoNum", "KisyuCode", "BanusiCode", "ChokyosiCode", 
                    "horseweight", "Futan", "SexCD", "Barei", "HinsyuCD", "KakuteiJyuni"]
    horse_df = horse_info_df[horse_info_cols]
    horse_df.rename(columns={"Umaban": "horse_num", "KettoNum": "horse_id",
                            "KisyuCode": "jockey_code", "BanusiCode": "owner_code", "ChokyosiCode": "trainer_code", 
                            "Futan": "carry_weight", "Barei": "horse_age",
                            "HinsyuCD": "horse_type", "KakuteiJyuni": "result"}, inplace=True)

    # Add a column that indicates which horse were in the top 3 winners
    horse_df["horse_rank_top3"] = -1
    horse_df["horse_rank_top3"] = horse_df["result"].apply(lambda x: 1 if x <= 3 else 0)

    ''' Merge race_df and horse_df. Key value: race_id '''
    # There are many races that are not in the race_detail dataset. Use inner join to remove that data
    print("Merging race_df and horse_df...")
    print("number of unknown race_ids: ", len(horse_df["race_id"].unique()) - len(race_df["race_id"].unique()))
    print("Finished merging!\n")

    # Make a new dataframe which is the result of the merge between horse_df and race_df
    df = horse_df
    df = df.merge(race_df, left_on="race_id", right_on="race_id", suffixes=(False, False))


    """''''''''''''''''''''''''''''''''''''''''''''"""
    ''' Manipulate and calculate meaningful statistics of features in df'''
    print("Calculate meaningful statistics of jockeys, trainers, owners, and horses...")
    #   *** Calc win rate of jockeys ***
    df["jockey_winrate_top3"] = -1
    df["jockey_avg_rank"] = -1

    # Get all jockey code
    jockeys = df["jockey_code"].unique()
    # Get the number of races each jockey participate in and calc the top winrate
    jockey_num_races = {}
    jockey_winrate_top3 = {}
    jockey_avg_rank = {}

    for jockey in jockeys:
        # Get the number of races each jockey participate in and calc the top1 winrate of jockey
        jockey_num_races[jockey] = df[df["jockey_code"] == jockey].shape[0]
        # Calc top-3 winrate of jockey (position from 1 to 3)
        jockey_winrate_top3_count = df[(df["jockey_code"] == jockey) & (df["result"] <= 3)].shape[0]
        jockey_winrate_top3[jockey] = jockey_winrate_top3_count / jockey_num_races[jockey]
        # Calc avg rank
        jockey_avg_rank[jockey] = df[df["jockey_code"] == jockey]["result"].mean()

    # Smoothing
    # Get means of jockey winrate top-3
    jockey_winrate_top3_mean = np.array(list(jockey_winrate_top3.values())).mean()
    # Smoothing pamerater
    alpha = 0.2
    for jockey in jockeys:
        jockey_winrate_top3[jockey] = (1 - np.exp(-alpha * jockey_num_races[jockey]))*jockey_winrate_top3[jockey] \
            + np.exp(-alpha * jockey_num_races[jockey]) * jockey_winrate_top3_mean

    # Append value of winrate top3 to horse info dataframe
    df["jockey_winrate_top3"] = df["jockey_code"].apply(lambda x: jockey_winrate_top3[x])
    # Create avg rank of jockey
    df["jockey_avg_rank"] = df["jockey_code"].apply(lambda x: jockey_avg_rank[x])


    #   *** Calc win rate of owners ***
    df["owner_winrate_top3"] = -1

    # Get all owner code
    owners = df["owner_code"].unique()
    # Get the number of races each jockey participate in and calc the top1 winrate of owner
    owner_num_races = {}
    owner_winrate_top3 = {}
    owner_avg_rank = {}

    for owner in owners:
        # Get the number of races each owner participate in and calc the top1 winrate of owner
        owner_num_races[owner] = df[df["owner_code"] == owner].shape[0]
        # Calc top-3 winrate of owner (position from 1 to 3)
        owner_winrate_top3_count = df[(df["owner_code"] == owner) & (df["result"] <= 3)].shape[0]
        owner_winrate_top3[owner] = owner_winrate_top3_count / owner_num_races[owner]
        # Calc avg rank
        owner_avg_rank[owner] = df[df["owner_code"] == owner]["result"].mean()

    # Smoothing
    # Get means of owner winrate top-3
    owner_winrate_top3_mean = np.array(list(owner_winrate_top3.values())).mean()
    # Smoothing pamerater
    alpha = 0.2
    for owner in owners:
        owner_winrate_top3[owner] = (1 - np.exp(-alpha * owner_num_races[owner]))*owner_winrate_top3[owner] \
            + np.exp(-alpha * owner_num_races[owner]) * owner_winrate_top3_mean

    # Append value of winrate top3 to horse info dataframe
    df["owner_winrate_top3"] = df["owner_code"].apply(lambda x: owner_winrate_top3[x])
    # Create avg rank of owner
    df["owner_avg_rank"] = df["owner_code"].apply(lambda x: owner_avg_rank[x])


    #   *** Calc win rate of trainer ***
    df["trainer_winrate_top3"] = -1

    # Get all trainer code
    trainers = df["trainer_code"].unique()
    # Get the number of races each trainer participate in and calc the top1 winrate of trainer
    trainer_num_races = {}
    trainer_winrate_top3 = {}
    trainer_avg_rank = {}

    for trainer in trainers:
        # Get the number of races each trainer participate in and calc the top1 winrate of trainer
        trainer_num_races[trainer] = df[df["trainer_code"] == trainer].shape[0]
        # Calc top-3 winrate of trainers (position from 1 to 3)
        trainer_winrate_top3_count = df[(df["trainer_code"] == trainer) & (df["result"] <= 3)].shape[0]
        trainer_winrate_top3[trainer] = trainer_winrate_top3_count / trainer_num_races[trainer]
        # Calc avg rank
        trainer_avg_rank[trainer] = df[df["trainer_code"] == trainer]["result"].mean()

    # Smoothing
    # Get means of trainer winrate top-3
    trainer_winrate_top3_mean = np.array(list(trainer_winrate_top3.values())).mean()
    # Smoothing pamerater
    alpha = 0.2
    for trainer in trainers:
        trainer_winrate_top3[trainer] = (1 - np.exp(-alpha * trainer_num_races[trainer]))*trainer_winrate_top3[trainer] \
            + np.exp(-alpha * trainer_num_races[trainer]) * trainer_winrate_top3_mean

    # Append value of winrate top3 to horse info dataframe
    df["trainer_winrate_top3"] = df["trainer_code"].apply(lambda x: trainer_winrate_top3[x])
    # Create avg rank of trainer
    df["trainer_avg_rank"] = df["trainer_code"].apply(lambda x: trainer_avg_rank[x])

    #   *** Add recent average rank of 6 most recent races of each horse to the dataset ***
    horse_ids = df["horse_id"].unique()
    print(f"number of horses in the dataset: {len(horse_ids)}")

    df["recent_avg_rank"] = df["result"]
    df["recent_avg_rank"] = df["recent_avg_rank"].astype("float")
    count = 0
    start = time.time()
    recent_avg_rank_dict = {}
    for horse_id in horse_ids:
        if count == 0:
            start_loop = time.time()
        count += 1
        recent_pos = []
        for index, pos in zip(df[df["horse_id"] == horse_id].index, df[df["horse_id"] == horse_id]["recent_avg_rank"]):
            recent_pos.append(pos)
            # Calc 6 most recent average rank
            recent_avg_rank = sum(map(float, recent_pos)) / float(len(recent_pos))
            df.loc[index, "recent_avg_rank"] = recent_avg_rank
            if len(recent_pos) > 6:
                recent_pos = recent_pos[1:]

        # Get the latest average rank for each horse to prepare for prediction
        recent_avg_rank_dict[horse_id] = recent_avg_rank
        # Get the elapse time after 5000 iterations
        if count % 5000 == 0:        
            end_loop = time.time()
            print("elapse time of recent 5000 iterations: ", end_loop - start_loop)
            start_loop = end_loop

    end = time.time()
    print("total elapse time: ", end - start)
    print("Finished calculating statistics\n")

    recent_avg_rank_dict = stringtify_key(recent_avg_rank_dict)
    trainer_winrate_top3 = stringtify_key(trainer_winrate_top3)
    trainer_avg_rank = stringtify_key(trainer_avg_rank)
    owner_winrate_top3 = stringtify_key(owner_winrate_top3)
    owner_avg_rank = stringtify_key(owner_avg_rank)
    jockey_winrate_top3 = stringtify_key(jockey_winrate_top3)
    jockey_avg_rank = stringtify_key(jockey_avg_rank)


    # Save extracted features for later use.
    print("Creating containers for all ranking features.")
    feature_folder = parent_path + "/src/extract_feature/"
    create_json(feature_folder + "horse_recent_rank.json", recent_avg_rank_dict)
    create_json(feature_folder + "trainer_winrate_top3.json", trainer_winrate_top3)
    create_json(feature_folder + "trainer_avg_rank.json", trainer_avg_rank)
    create_json(feature_folder + "owner_winrate_top3.json", owner_winrate_top3)
    create_json(feature_folder + "owner_avg_rank.json", owner_avg_rank)
    create_json(feature_folder + "jockey_winrate_top3.json", jockey_winrate_top3)
    create_json(feature_folder + "jockey_avg_rank.json", jockey_avg_rank)

    print ("\nCreating complete_data.csv...")
    df.to_csv(feature_folder+'complete_data.csv')
    print ("Complete!")