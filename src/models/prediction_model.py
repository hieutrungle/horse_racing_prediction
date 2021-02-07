# Hide warnings
import warnings
warnings.filterwarnings('ignore')

# Import libraries
import json
from os.path import dirname, abspath
from sklearn import linear_model
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
import pandas as pd
# import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import lightgbm as lgb
import catboost as cb

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

# Standard Normalization
print("Normalization...")
std_scaler = StandardScaler()
std_scaler.fit(X_train)
X_train_std = std_scaler.transform(X_train)
X_test_std = std_scaler.transform(X_test)
print("Finished normalization!\n")
# # Function to plot confusion matrix
# def plot_confusion_matrix(predictions, 
#                           groundtruth, 
#                           class_names,
#                           normalize=True):

#     import matplotlib.pyplot as plt
#     import numpy as np
#     import itertools
#     from sklearn.metrics import confusion_matrix
    
#     cm = confusion_matrix(groundtruth, predictions)
    
                          
#     accuracy = np.trace(cm) / float(np.sum(cm))
#     misclass = 1 - accuracy

#     cmap = plt.get_cmap('Blues')
#     title='Confusion matrix'
#     plt.figure(figsize=(8, 6))
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()

#     if class_names is not None:
#         tick_marks = np.arange(len(class_names))
#         plt.xticks(tick_marks, class_names, rotation=45)
#         plt.yticks(tick_marks, class_names)

#     if normalize:
#         cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


#     thresh = cm.max() / 1.5 if normalize else cm.max() / 2
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         if normalize:
#             plt.text(j, i, "{:0.4f}".format(cm[i, j]),
#                      horizontalalignment="center",
#                      color="white" if cm[i, j] > thresh else "black")
#         else:
#             plt.text(j, i, "{:,}".format(cm[i, j]),
#                      horizontalalignment="center",
#                      color="white" if cm[i, j] > thresh else "black")


#     plt.tight_layout()
#     plt.ylabel('Ground Truth label')
#     plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
#     plt.grid(False)
#     print("correct_pred_top3/total_actual_top3: ", cm[1,1]/(cm[1,0]+cm[1,1]))
#     plt.show()

def show_accuracy(groundtruth, prediction):
    # input: groundtruth and prediction results
    # output: f1_score and comfussion matrix
    f1_score_model = f1_score(groundtruth, prediction, average="weighted")
    print(f"F1 score: {f1_score_model:0.5f}")
    cm = confusion_matrix(groundtruth, prediction)
    print(f"correct_pred_top3/total_actual_top3: {cm[1,1]/(cm[1,0]+cm[1,1]):0.5f}\n")
    return f1_score, cm

""" Prediction """
# *** Logistic Regression ***
print("Logistic Regression...")
lr_model = linear_model.LogisticRegression()

t0 = time.time()
lr_model.fit(X_train_std, y_train_top3)
print(f"Training time for top3 logistic regression classifier: {time.time() - t0} sec\n")
lr_predict_top3 = lr_model.predict(X_test_std)
show_accuracy(y_test_top3, lr_predict_top3)


# *** Random Forest ***
print("Random Forest...")
rf_model = RandomForestClassifier()

t0 = time.time()
rf_model.fit(X_train_std, y_train_top3)
t3 = time.time() - t0
print(f"Training time for top3 random forest classifier: {time.time() - t0} sec\n")
rf_predict_top3 = rf_model.predict(X_test_std)
show_accuracy(y_test_top3, rf_predict_top3)


# *** CatBoost ***
print("Catboost...")
cb_model = cb.CatBoostClassifier(eval_metric="AUC", depth=10, iterations= 500, 
                                  l2_leaf_reg= 9, learning_rate= 0.15, class_weights = (1, 3.5),
                                  silent=True)

t0 = time.time()
cb_model.fit(X_train_std,y_train_top3)
print(f"Training time for top3 CatBoost classifier: {time.time() - t0} sec\n")
cb_predict_top3 = cb_model.predict(X_test_std)
# Round the float number
cb_predict_top3 = cb_predict_top3.round(0)
#converting from float to integer
cb_predict_top3 = cb_predict_top3.astype(int)
show_accuracy(y_test_top3, cb_predict_top3)


# *** LightGBM ***
print("LightGBM...")
lg_model = lgb.LGBMClassifier(silent=False)

t0 = time.time()
d_train = lgb.Dataset(X_train_std, label=y_train_top3)
params = {"max_depth": 50, "learning_rate" : 0.1, "num_leaves": 300,  "n_estimators": 300,
          'objective': 'binary', "metrics": "auc", "is_unbalance": True}
lg_model = lgb.train(params, d_train)
t5 = time.time() - t0
print(f"Training time for top3 LightGBM classifier: {time.time() - t0} sec\n")
lg_predict_top3 = lg_model.predict(X_test_std)
# Round the float number
lg_predict_top3 = lg_predict_top3.round(0)
#converting from float to integer
lg_predict_top3 = lg_predict_top3.astype(int)
show_accuracy(y_test_top3, lg_predict_top3)
