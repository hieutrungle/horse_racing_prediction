# Horse Racing Prediction
The goal of this project is to predict which horse in a race can be the first 3 winners.


## Data
The data is given and can be downloaded [here](https://drive.google.com/file/u/2/d/18EdiC515lnr7NDKJK_EdELRCmk5t2z0T/view)

## Domain Expertise
One should have at least basic understanding of horse racing in order to extract information from the given dataset.

Basic information about Japanese horse racing can be found [here](http://japanracing.jp/en/racing/go_racing/guide/)

## Library Installation
All dependencies are listed in the requirements.txt

### Steps to install all denpendencies
download requirements.txt

In the terminal, type: 
`pip install -r requirements.txt`

## File Description
- notebook_data_processing: preprocess and select meaningful data to build model
- classification: horse top3 prediction based on preprocessed data
- Input file: data is given in the compressed file given in the Data section. After unziping it, data in historical_data folder will be used.

# Steps to predict top3 horses

## Step 1: Preparation
- Download the data in the link given above.
- extract and place file in folder data/ in the main directory

## Step 2: Data Processing
- In ./src/data_process, run data_process.py file to process raw data.
- The outputs are training.csv and testing.csv in extract_feature folder

## Step 3: Feature Selection
- In the newly created training and testing files, there are still many features for modeling. Therefore, several methods are applied to choose suitable features.
- Run the file feature_selection.py then it will output useful_features.json in the same directory.

## Step 4: Model Tuning and Training
- Logistic Regression and Random Forest models are built and tuned in the basic_prediction_model.py.
- CatBoost and LightGBM models are built and tuned in advanced_pred_model.py. Moreover, soft and hard ensemble models are also created based on CatBoost and LightGBM.
- Scores of all models are recorded in src/models/score dicrectory.
- Optimal parameters of each model are place in separated json files in src/models/best_hyperparameters directory.
- After having tuned all parameters, training and testing datasets are combined and used to build soft voting ensemble in final_model.py. It output *_final.pkl models.

## Step 5: Deplot model on Flask website
- Working on it


# GitHub URL
**[https://github.com/GarlicSoup/horse_racing_prediction](https://github.com/GarlicSoup/horse_racing_prediction)**

# License
This program is created by [Hieu Le](https://github.com/GarlicSoup)
