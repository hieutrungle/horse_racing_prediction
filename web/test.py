import pandas as pd
import os

current_path = os.path.dirname(__file__)
parent_path = os.path.dirname(current_path)
data_path = parent_path + "/data/training.csv"
print(data_path)

df = pd.read_csv(data_path)
print(df.info())