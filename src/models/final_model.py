
# Hide warnings
import warnings
warnings.filterwarnings('ignore')

# Import libraries
import json
import os
from os.path import dirname, abspath
from sklearn.metrics import f1_score
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
import catboost 
#from imblearn.over_sampling import SMOTE, ADASYN
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.utils import class_weight
import numpy as np
from sklearn.ensemble import VotingClassifier

"""## Import data"""

parent_dir_path = dirname(dirname(abspath(__file__)))
print(parent_dir_path)
df_train = pd.read_csv(parent_dir_path +"/extract_feature/training.csv")
df_test = pd.read_csv(parent_dir_path +"/extract_feature/testing.csv")

df = pd.concat([df_train, df_test], ignore_index=True)

"""## Select features based upon those algorithm and personal analysis"""

print("Select useful data...")
with open(parent_dir_path + '/models/useful_features.json') as f:
  useful_features = json.load(f)
# Remove SexCD attribute
useful_features.pop("SexCD")

X = df[useful_features]
y_top3 = df["horse_rank_top3"]
print("Finished selecting!\n")

"""# Normalization

Standard Normalization
"""

print("Normalization...")
std_scaler = StandardScaler()
std_scaler.fit(X)
X_std = std_scaler.transform(X)
print("Finished normalization!\n")

"""# Useful functions:

## Accuracy calculation and pickling models

- def calc_accuracy: calculate and show accuracy

- def create_model_pickle: create a pickle file that contains the model architecture
"""
def calc_accuracy(groundtruth, prediction):
    # input: groundtruth and prediction results
    # output: f1_score and comfussion matrix
    f1_score_model = f1_score(groundtruth, prediction, average="weighted")
    print(f"F1 score: {f1_score_model:0.5f}")
    cm = confusion_matrix(groundtruth, prediction)
    true_pos_win_rate = cm[1,1]/(cm[1,0]+cm[1,1])
    print(f"correct_pred_top3/total_actual_top3: {true_pos_win_rate:0.5f}")
    return f1_score_model, true_pos_win_rate

def create_model_pickle(model, filename):
    print(f"Pickling the model into the file {filename}")
    with open("./output_models/" + filename, 'wb') as f:
        pickle.dump(model, f)
    print("Finished pickling the model")


with open("./best_hyperparameters/catboost_best_hyperparams.json", 'r') as f:
    catboost_best_hyperparams = json.load(f)

with open("./best_hyperparameters/lgb_best_hyperparams.json", 'r') as f:
    lgb_best_hyperparams = json.load(f)

# CatBoost
catboost_model = catboost.CatBoostClassifier()
catboost_model.set_params(**catboost_best_hyperparams)
t0 = time.time()
catboost_model.fit(X_std, y_top3)
print(f"Training time for top3 CatBoost classifier: {time.time() - t0} sec")

# Prediction on training dataset
catboost_predict_top3 = catboost_model.predict(X_std)
# Round the float number
catboost_predict_top3 = catboost_predict_top3.round(0)
#converting from float to integer
catboost_predict_top3 = catboost_predict_top3.astype(int)

# Calculate accuracy
f1_score_train, true_pos_win_rate_train = calc_accuracy(y_top3, catboost_predict_top3)

# Create a pickle file that contains the model
create_model_pickle(catboost_model, 'catboost_final.pkl')
cm = confusion_matrix(y_top3, catboost_predict_top3)
print(cm)
print("CatBoost Done!\n")

# class_name = [0, 1]
# plot_confusion_matrix(catboost_predict_top3, y_top3, class_name, normalize=False)

# LightGBM
lgb_model = lgb.LGBMClassifier()
lgb_model.set_params(**lgb_best_hyperparams)
t0 = time.time()
lgb_model.fit(X_std, y_top3)
print(f"Training time for top3 CatBoost classifier: {time.time() - t0} sec")

# Prediction on training dataset
lgb_predict_top3 = lgb_model.predict(X_std)
# Round the float number
lgb_predict_top3 = lgb_predict_top3.round(0)
#converting from float to integer
lgb_predict_top3 = lgb_predict_top3.astype(int)

# Calculate accuracy
f1_score_train, true_pos_win_rate_train = calc_accuracy(y_top3, lgb_predict_top3)

# Create a pickle file that contains the model
create_model_pickle(lgb_model, 'lightGBM_final.pkl')
cm = confusion_matrix(y_top3, lgb_predict_top3)
print(cm)
print("LightGBM Done!\n")


"""## Ensemble model

### Soft Voting
"""
print("Soft Voting Ensemble...")
#create a dictionary of our models
estimator = [("catboost", catboost_model), 
             ("lightGBM", lgb_model)]

#create our voting classifier, inputting our models

vot_soft = VotingClassifier(estimators = estimator, voting ='soft')
t0 = time.time()
vot_soft.fit(X_std, y_top3)
print(f"Training time for top3 Soft Voting classifier: {time.time() - t0} sec")
yhat = vot_soft.predict(X_std)

# Calculate accuracy
f1_score, true_pos_win_rate = calc_accuracy(y_top3, yhat)

create_model_pickle(vot_soft, 'vot_soft_final.pkl')
cm = confusion_matrix(y_top3, lgb_predict_top3)
print(cm)
print("Soft Voting Ensemble Done!\n")


### Hard Voting
print("Hard Voting Ensemble...")
#create a dictionary of our models
estimator = [("catboost", catboost_model), 
             ("lightGBM", lgb_model)]

#create our voting classifier, inputting our models
vot_hard = VotingClassifier(estimators = estimator, voting ='hard')
t0 = time.time()
vot_hard.fit(X_std, y_top3)
print(f"Training time for top3 Hard Voting classifier: {time.time() - t0} sec")
yhat = vot_hard.predict(X_std)

# Calculate accuracy
# f1_score, true_pos_win_rate = calc_accuracy(y_top3, yhat)

create_model_pickle(vot_hard, 'vot_hard_final.pkl')
cm = confusion_matrix(y_top3, lgb_predict_top3)
print(cm)
print("Hard Voting Ensemble Done!")