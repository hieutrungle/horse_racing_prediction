# Hide warnings
import warnings
warnings.filterwarnings('ignore')

# Import libraries
import json
import os
from os.path import dirname, abspath
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pandas as pd
# import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import SMOTE, ADASYN
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from sklearn.model_selection import cross_val_score
import pickle

#plt.style.use('fivethirtyeight')

# Set the maximum columns displayed on the screen
pd.set_option("display.max_columns", 150)

# Import data
current_path = dirname(dirname(abspath(__file__)))
parent_path = dirname(current_path)
df_train = pd.read_csv(parent_path +"/data/training.csv")
df_test = pd.read_csv(parent_path +"./data/testing.csv")

# Select features based upon those algorithm and personal analysis
print("Select useful data...")
with open('./useful_features.json') as f:
  useful_features = json.load(f)
# Remove SexCD attribute
useful_features.pop("SexCD")

X_train = df_train[useful_features]
y_train_top3 = df_train["horse_rank_top3"]
X_test = df_test[useful_features]
y_test_top3 = df_test["horse_rank_top3"]
print("Finished selecting!\n")

# Synthetic Minority Oversampling Technique.
# SMOTE uses a nearest neighbors algorithm to generate new and 
# synthetic data we can use for training our model.
print("Handle imbalanced data...")
print("original ratio 1/0:", len(y_train_top3[y_train_top3 == 1])/ len(y_train_top3[y_train_top3 == 0]))
sm = ADASYN()
X_train, y_train_top3 = sm.fit_resample(X_train, y_train_top3)
print("after upsampling, ratio 1/0: ", len(y_train_top3[y_train_top3 == 1])/ len(y_train_top3[y_train_top3 == 0]))
print("Finished Handling Imbalance!\n")

# Standard Normalization
print("Normalization...")
std_scaler = StandardScaler()
std_scaler.fit(X_train)
X_train_std = std_scaler.transform(X_train)
X_test_std = std_scaler.transform(X_test)
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
    # create a pickle file that contains the model architecture
    print(f"Pickling the model into the file {filename}")
    with open("./output_models/" + filename, 'wb') as f:
        pickle.dump(model, f)
    print("Finished pickling the model")

"""
    Import f1_scores and true_pos_wining_rate if exist, otherwise create new dictionaries
"""

if os.path.exists('./score/f1_scores.json') == True:
    with open("./score/f1_scores.json", 'r') as f:
        f1_scores = json.load(f)
else:
    f1_scores = {'train':{'logis_reg': 0, 'randomForest': 0, 'catboost': 0, 'lightGBM': 0, 'EN_HARD': 0, 'EN_SOFT': 0},
             'test':{'logis_reg': 0, 'randomForest': 0, 'catboost': 0, 'lightGBM': 0, 'EN_HARD': 0, 'EN_SOFT': 0}}
    true_pos_wining_rates = {'train':{'logis_reg': 0, 'randomForest': 0, 'catboost': 0, 'lightGBM': 0, 'EN_HARD': 0, 'EN_SOFT': 0},
                                'test':{'logis_reg': 0, 'randomForest': 0, 'catboost': 0, 'lightGBM': 0, 'EN_HARD': 0, 'EN_SOFT': 0}}

if os.path.exists('./score/true_pos_wining_rates.json') == True:
    with open("./score/true_pos_wining_rates.json", 'r') as f:
        true_pos_wining_rates = json.load(f)
else:
    true_pos_wining_rates = {'train':{'logis_reg': 0, 'randomForest': 0, 'catboost': 0, 'lightGBM': 0, 'EN_HARD': 0, 'EN_SOFT': 0},
                            'test':{'logis_reg': 0, 'randomForest': 0, 'catboost': 0, 'lightGBM': 0, 'EN_HARD': 0, 'EN_SOFT': 0}}

""" Prediction """

# *** Logistic Regression ***
print("Logistic Regression...")
fit_intercept = {0: True, 1: False}
solver = {0: 'newton-cg', 1: 'lbfgs', 2: 'sag', 3: 'saga'}
lr_space = {'tol': hp.uniform('tol', 1e-6, 1e-3),
        'C': hp.quniform('C', 1, 1000, 10),
        'fit_intercept': hp.choice('fit_intercept', fit_intercept.values()),
        'solver': hp.choice('solver', solver.values())
    }

def lr_objective(params):
    model = linear_model.LogisticRegression(**params)
    acc = cross_val_score(model, X_train_std, y_train_top3, cv = 4, scoring='f1').mean()

    # We aim to maximize accuracy, therefore we return it as a negative value
    return {'loss': -acc, 'status': STATUS_OK }

t_tuning = time.time()
lr_trials = Trials()
lr_best_hyperparams = fmin(fn= lr_objective,
                space= lr_space,
                algo= tpe.suggest,
                max_evals = 100,
                trials= lr_trials)
print(f"elapse time of tuning hyperparameter process: {time.time()-t_tuning} sec")
lr_best_hyperparams['fit_intercept']= fit_intercept[lr_best_hyperparams['fit_intercept']]
lr_best_hyperparams['solver']= solver[lr_best_hyperparams['solver']]
print("best hyperparameters for logistic regression: ", lr_best_hyperparams)
# Dump the best parameters to json file for later use
with open('./best_hyperparameters/logis_best_hyperparams.json', 'w') as fp:
    json.dump(lr_best_hyperparams, fp)

# lr_model = linear_model.LogisticRegression(C=lr_best_hyperparams['C'],
#                                         tol=lr_best_hyperparams['tol'],
#                                         fit_intercept=lr_best_hyperparams['fit_intercept'],
#                                         solver=lr_best_hyperparams['solver'])
lr_model = linear_model.LogisticRegression()
t0 = time.time()
lr_model.set_params(**lr_best_hyperparams)
lr_model.fit(X_train_std, y_train_top3)
print(f"Training time for top3 logistic regression classifier: {time.time() - t0} sec")

# Prediction on training dataset
lr_predict_top3_train = lr_model.predict(X_train_std)

# Prediction on testing dataset
lr_predict_top3_test = lr_model.predict(X_test_std)

# Calculate accuracy
print("Accuracy for training set:")
f1_score_train, true_pos_win_rate_train = calc_accuracy(y_train_top3, lr_predict_top3_train)
print("Accuracy for testing set:")
f1_score_test, true_pos_win_rate_test = calc_accuracy(y_test_top3, lr_predict_top3_test)

# Append values to accuracy dictionary
f1_scores["train"]["logis_reg"] = round(f1_score_train, 4)
true_pos_wining_rates["train"]["logis_reg"] = round(true_pos_win_rate_train, 4)
f1_scores["test"]["logis_reg"] = round(f1_score_test, 4)
true_pos_wining_rates["test"]["logis_reg"] = round(true_pos_win_rate_test, 4)

create_model_pickle(lr_model, 'logis_model.pkl')
print("Logistic Regression Done!\n")



# *** Random Forest ***
print("Random Forest...")
crit = {0: 'entropy', 1: 'gini'}
feat = {0: 'sqrt', 1: 'log2'}
est = {0: 50, 1: 150, 2: 300}
rf_space = {
    "n_estimators": hp.choice("n_estimators", est.values()),
    "max_depth": hp.quniform("max_depth", 5, 400,10),
    "criterion": hp.choice("criterion", crit.values()),
    'max_features': hp.choice('max_features', feat.values()),
    'min_samples_split': hp.randint('min_samples_split', 500),
    'min_samples_leaf': hp.randint('min_samples_leaf', 300)
}

def rf_objective(params):
    model = RandomForestClassifier(**params, n_jobs=-1)
    acc = cross_val_score(model, X_train_std, y_train_top3, cv = 4, scoring='roc_auc').mean()
    # We aim to maximize accuracy, therefore we return it as a negative value
    return {'loss': -acc, 'status': STATUS_OK }

t_tuning = time.time()
rf_trials = Trials()
rf_best_hyperparams = fmin(fn= rf_objective,
                space= rf_space,
                algo= tpe.suggest,
                max_evals = 20,
                trials= rf_trials)
print(f"elapse time of tuning hyperparameter process: {time.time()-t_tuning} sec")
print("best hyperparameters for Random Forest: ", rf_best_hyperparams)

rf_best_hyperparams['criterion'] = crit[rf_best_hyperparams['criterion']]
rf_best_hyperparams['max_features'] = feat[rf_best_hyperparams['max_features']]
rf_best_hyperparams['n_estimators'] = est[rf_best_hyperparams['n_estimators']]
rf_best_hyperparams['min_samples_split'] = int(rf_best_hyperparams['min_samples_split'])
rf_best_hyperparams['min_samples_leaf'] = int(rf_best_hyperparams['min_samples_leaf'])
print("best hyperparameters for Random Forest: ", rf_best_hyperparams)
# Dump the best parameters to json file for later use
with open('./best_hyperparameters/ranforest_best_hyperparams.json', 'w') as fp:
    json.dump(rf_best_hyperparams, fp)

rf_model = RandomForestClassifier(n_jobs=-1)
rf_model.set_params(**rf_best_hyperparams)
t0 = time.time()
rf_model.fit(X_train_std,y_train_top3)
print(f"Training time for top3 random forest classifier: {time.time() - t0} sec")

# Prediction on training dataset
rf_predict_top3_train = rf_model.predict(X_train_std)
# Prediction on testing dataset
rf_predict_top3_test = rf_model.predict(X_test_std)
# Calculate accuracy
print("Accuracy for training set:")
f1_score_train, true_pos_win_rate_train = calc_accuracy(y_train_top3, rf_predict_top3_train)
print("Accuracy for testing set:")
f1_score_test, true_pos_win_rate_test = calc_accuracy(y_test_top3, rf_predict_top3_test)

# Append values to accuracy dictionary
f1_scores["train"]["randomForest"] = round(f1_score_train, 4)
true_pos_wining_rates["train"]["randomForest"] = round(true_pos_win_rate_train, 4)
f1_scores["test"]["randomForest"] = round(f1_score_test, 4)
true_pos_wining_rates["test"]["randomForest"] = round(true_pos_win_rate_test, 4)

with open('./score/f1_scores.json', 'w') as fp:
    json.dump(f1_scores, fp)
with open('./score/true_pos_wining_rates.json', 'w') as fp:
    json.dump(true_pos_wining_rates, fp)

create_model_pickle(rf_model, 'random_forest_model.pkl')
print("Random Forest Done!\n")