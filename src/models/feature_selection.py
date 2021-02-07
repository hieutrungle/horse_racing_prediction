# Hide warnings
import warnings
warnings.filterwarnings('ignore')

# Import libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
from os.path import dirname, abspath
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
import time
import json

# Import data
current_path = dirname(dirname(abspath(__file__)))
parent_path = dirname(current_path)
df_train = pd.read_csv(parent_path +"/data/training.csv")
df_test = pd.read_csv(parent_path +"./data/testing.csv")

# Get data for classification
drop_cols = ["result", "horse_rank_top3", "trainer_winrate_top3", "owner_winrate_top3"]
X_train = df_train.iloc[:,7:].drop(drop_cols, axis=1)
y_train_top3 = df_train["horse_rank_top3"]
X_test = df_test.iloc[:,7:].drop(drop_cols, axis=1)
y_test_top3 = df_test["horse_rank_top3"]

# This dict store what features will be selected and how many times they are chosen
feature_dict = {}

""" Feature selection """
num_features = 8
# Pearson Correlation
t0 = time.time()
print("Peason Correlation...")
def cor_selector(X, y,num_feats):
    cor_list = []
    feature_name = X.columns.tolist()
    # calculate the correlation with y for each feature
    for i in feature_name:
        cor = np.corrcoef(X[i], y)[0, 1]
        cor_list.append(cor)
    # replace NaN with 0
    cor_list = [0 if np.isnan(i) else i for i in cor_list]
    # feature name
    cor_feature = X.iloc[:,np.argsort(np.abs(cor_list))[-num_feats:]].columns.tolist()
    # feature selection? 0 for not select, 1 for select
    cor_support = [True if i in cor_feature else False for i in feature_name]
    return cor_support, cor_feature
cor_support, cor_features = cor_selector(X_train, y_train_top3, num_features)
print(f"run time: {time.time() - t0:0.5f} sec")
print(str(len(cor_features)), 'selected features')
print(f"selected features: {cor_features}\n")
for feature in cor_features:
    feature_dict[feature] = feature_dict.get(feature, 0) + 1

# SelectFromModel tree-based
print("SelecFromModel tree-based...")
t0 = time.time()
embeded_rf_selector = SelectFromModel(RandomForestClassifier(n_estimators=100))
embeded_rf_selector.fit(X_train, y_train_top3)
embeded_rf_support = embeded_rf_selector.get_support()
embeded_rf_features = X_train.loc[:,embeded_rf_support].columns.tolist()
print(f"run time: {time.time() - t0:0.5f} sec")
print(str(len(cor_features)), 'selected features')
print(f"selected features: {embeded_rf_features}\n")
for feature in embeded_rf_features:
    feature_dict[feature] = feature_dict.get(feature, 0) + 1

# Chi-squared
print("Chi-square...")
t0 = time.time()
X_norm = MinMaxScaler().fit_transform(X_train)
chi_selector = SelectKBest(chi2, k=num_features)
chi_selector.fit(X_norm, y_train_top3)
chi_support = chi_selector.get_support()
chi_features = X_train.loc[:,chi_support].columns.tolist()
print(f"run time: {time.time() - t0:0.5f} sec")
print(str(len(chi_features)), 'selected features')
print(f"selected features: {chi_features}\n")
for feature in chi_features:
    feature_dict[feature] = feature_dict.get(feature, 0) + 1

# Recursive Feature Elimination
print("Recursive Feature Elimination...")
t0 = time.time()
rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=num_features, step=30, verbose=5)
rfe_selector.fit(X_norm, y_train_top3)
rfe_support = rfe_selector.get_support()
rfe_features = X_train.loc[:,rfe_support].columns.tolist()
print(f"run time: {time.time() - t0:0.5f} sec")
print(str(len(rfe_features)), 'selected features')
print(f"selected features: {rfe_features}\n")
for feature in rfe_features:
    feature_dict[feature] = feature_dict.get(feature, 0) + 1

print("Keys: selected features")
print("Values: number of times each feature has been chosen")
print(feature_dict)

json = json.dumps(feature_dict)
with open("useful_features.json","w") as f:
    f.write(json)