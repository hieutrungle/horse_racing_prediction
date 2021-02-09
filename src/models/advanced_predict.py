# Hide warnings
import warnings
warnings.filterwarnings('ignore')

# Import libraries
import json
from os.path import dirname, abspath
from sklearn.metrics import f1_score
import pandas as pd
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
import catboost 
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

def show_accuracy(groundtruth, prediction):
    # input: groundtruth and prediction results
    # output: f1_score and comfussion matrix
    f1_score_model = f1_score(groundtruth, prediction, average="weighted")
    print(f"F1 score: {f1_score_model:0.5f}")
    cm = confusion_matrix(groundtruth, prediction)
    print(f"correct_pred_top3/total_actual_top3: {cm[1,1]/(cm[1,0]+cm[1,1]):0.5f}")
    return f1_score, cm

def create_model_pickle(model, filename):
    print(f"Pickling the model into the file {filename}")
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print("Finished pickling the model")

""" Prediction """
# *** CatBoost ***
print("Catboost...")
# Bayesian Optimization for tuning hyperparameters
print("Bayesian Optimization for tuning hyperparameters")
iterations = {0: 100, 1: 250, 2: 450, 3: 700, 4: 1000, 5: 2000}
catboost_space = {
    "iterations": hp.choice("iterations", iterations.values()),
    "l2_leaf_reg": hp.uniform("l2_leaf_reg", 1, 1000),
    "depth": hp.quniform("depth", 1, 10, 1),
    'learning_rate': hp.uniform('learning_rate', 0.005, 0.3)
}

def catboost_objective(params):
    model = catboost(**params, eval_metric="AUC", early_stopping_rounds=150, task_type="GPU", silent=True)
    acc = cross_val_score(model, X_train_std, y_train_top3, cv = 4, scoring='roc_auc').mean()
    # We aim to maximize accuracy, therefore we return it as a negative value
    return {'loss': -acc, 'status': STATUS_OK }

t_tuning = time.time()
catboost_trials = Trials()
catboost_best_hyperparams = fmin(fn= catboost_objective,
                space= catboost_space,
                algo= tpe.suggest,
                max_evals = 100,
                trials= catboost_trials)
print(f"elapse time of tuning hyperparameter process: {time.time()-t_tuning} sec")

catboost_best_hyperparams = { 'iterations': iterations[catboost_best_hyperparams['iterations']], 
                        'l2_leaf_reg': catboost_best_hyperparams['l2_leaf_reg'], 
                        'depth': catboost_best_hyperparams['depth'],
                        'learning_rate': catboost_best_hyperparams['learning_rate']
}
print("best hyperparameters for Random Forest: ", catboost_best_hyperparams)

catboost_model = catboost(iterations = catboost_best_hyperparams['iterations'], 
                                l2_leaf_reg = catboost_best_hyperparams['l2_leaf_reg'], 
                                depth = catboost_best_hyperparams['depth'],
                                learning_rate = catboost_best_hyperparams['learning_rate'],
                                silent=True, 
                                eval_metric="AUC",
                                early_stopping_rounds=250,
                                task_type="GPU",)

t0 = time.time()
catboost_model.fit(X_train_std,y_train_top3)
print(f"Training time for top3 CatBoost classifier: {time.time() - t0} sec")
catboost_predict_top3 = catboost_model.predict(X_test_std)
# Round the float number
catboost_predict_top3 = catboost_predict_top3.round(0)
#converting from float to integer
catboost_predict_top3 = catboost_predict_top3.astype(int)
show_accuracy(y_test_top3, catboost_predict_top3)
create_model_pickle(catboost_model, 'catboost.pkl')
print("CatBoost Done!\n")


# *** LightGBM *** , "is_unbalance": True
print("LightGBM...")
lg_model = lgb.LGBMClassifier(silent=False)

t0 = time.time()
d_train = lgb.Dataset(X_train_std, label=y_train_top3)
params = {"max_depth": 50, "learning_rate" : 0.1, "num_leaves": 300,  "n_estimators": 300,
          'objective': 'binary', "metrics": "auc"}
lg_model = lgb.train(params, d_train)
t5 = time.time() - t0
print(f"Training time for top3 LightGBM classifier: {time.time() - t0} sec")
lg_predict_top3 = lg_model.predict(X_test_std)
# Round the float number
lg_predict_top3 = lg_predict_top3.round(0)
#converting from float to integer
lg_predict_top3 = lg_predict_top3.astype(int)
show_accuracy(y_test_top3, lg_predict_top3)
create_model_pickle(lg_model, 'lightgbm.pkl')
print("LightGBM Done!\n")