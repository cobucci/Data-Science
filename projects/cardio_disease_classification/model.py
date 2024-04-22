# Importing libraries
import sys, os
path = os.path.dirname(os.path.realpath('__file__')).split("cardio_disease_classification")[0] + 'utils\\classification'
sys.path.insert(1, path)

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from decision_tree import decision_tree_classifier
from random_forest import random_forest_classifier
from bagging import bagging_classifier
from ada_boosting import ada_boosting_classifier
from gradient_boosting import gradient_boosting_classifier
from logistic_regression import logistic_regression_classifier
from voting import voting_classifier
from xgboosting import xgboosting_classifier

import pandas as pd
import numpy as np

def main():
    # Importing dataset
    dataset = pd.read_csv("./datasets/heart_data.csv")


    #Data Cleaning
    dataset.drop(columns=['index', 'id'], axis=1, inplace=True)

    cardio = dataset.drop(['cardio'], axis=1)
    cat_attribs = ['gender','cholesterol', 'gluc', 'smoke', 'alco', 'active']
    cardio_num = cardio.drop(cat_attribs, axis=1)
    num_attribs = list(cardio_num)

    num_pipeline = Pipeline([('std_scaler', StandardScaler())])
    full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs), #num_pipeline
    ("cat", OneHotEncoder(), cat_attribs), #one hot encoder
    ])
    cardio_prepared = full_pipeline.fit_transform(cardio)


    #Split into train and test set
    y = dataset['cardio'].values
    X = cardio_prepared.copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_type = 1
    
    #Train and fit the models
    print("\n####################################")
    logistic_regression_clf = logistic_regression_classifier(model_type, X_train, X_test, y_train, y_test)

    print("\n####################################")
    tree_clf = decision_tree_classifier(model_type, X_train, X_test, y_train, y_test)

    print("\n####################################")
    rf_clf = random_forest_classifier(model_type, X_train, X_test, y_train, y_test)
    
    print("\n####################################")
    bagging_clf = bagging_classifier(model_type, X_train, X_test, y_train, y_test)
  
    print("\n####################################")
    ada_clf = ada_boosting_classifier(model_type, X_train, X_test, y_train, y_test)
    
    print("\n####################################")
    gb_clf = gradient_boosting_classifier(model_type, X_train, X_test, y_train, y_test)
    
    print("\n####################################")
    voting_clf = voting_classifier(model_type, X_train, X_test, y_train, y_test, logistic_regression_clf, tree_clf, rf_clf, gb_clf)
    
    print("\n####################################")
    xgb_clf = xgboosting_classifier(model_type, X_train, X_test, y_train, y_test)
    
    models = [logistic_regression_clf, tree_clf, rf_clf, bagging_clf, ada_clf, gb_clf, voting_clf, xgb_clf]
    df_result_models = pd.DataFrame(data=models, columns=["Classifier", "Accuracy", "Precision", "Recall", "F1", "AUC"])
    df_result_models.sort_values(by=["Accuracy", "Precision", "Recall", "F1", "AUC"], inplace=True, ascending=False)
    print(df_result_models)
    
    
if __name__ == "__main__":
  main()