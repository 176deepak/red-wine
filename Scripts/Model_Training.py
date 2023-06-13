import Data_Processing
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import r2_score, accuracy_score
import pickle
import yaml

def train_test():
    train = pd.read_csv(r"Data\Splits\train.csv")
    test = pd.read_csv(r"Data\Splits\test.csv")

    X_train = train.drop(['quality'], axis=1)
    y_train = train['quality']
    X_test = test.drop(['quality'], axis=1)
    y_test = test['quality']

    return (X_train, X_test, y_train, y_test)


def models_report(X_train, X_test, y_train, y_test):
    accuracy_report = {}
    params_report = {}

    models = {
        "Logistic Classifier":LogisticRegression(), 
        "Random Forest Classifier":RandomForestClassifier(), 
        "Support Vector Classifier":SVC(), 
        "K Neighbors Classifier":KNeighborsClassifier(), 
        "XGB Classifier":XGBClassifier()
    }
    params = {
        "Logistic Classifier":{
            'tol':[1e-2, 1e-3, 1e-4, 1e-5],
            'C':[0.5, 0.75, 1, 1.5, 2],
            'solver':['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
        },
        "Random Forest Classifier":{
            'n_estimators':[90, 100, 110, 120],
            'criterion':['gini', 'entropy', 'log_loss'],
            'max_features':['sqrt', 'log2', None]
        },
        "Support Vector Classifier":{
            'C':[1,1.5,2,2.5,3],
            'gamma':['scale', 'auto'],
        },
        "K Neighbors Classifier":{
            'n_neighbors':[4,5,6,7],
            'weights':['uniform', 'distance'],
            'p':[1,2],
        },
        "XGB Classifier":{
            'n_estimators':[2,3,4,5],
            'learning_rate':[0.5,1,1.5,2],
        }
    }   

    for i in range(len(list(models))):
        model = list(models.values())[i]
        param = params[list(models.keys())[i]]
    
        gs = GridSearchCV(model, param, cv=3)
        gs.fit(X_train, y_train)
    
        model.set_params(**gs.best_params_)
        model.fit(X_train, y_train)
    
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)
    
        train_model_score = accuracy_score(y_train, y_train_pred)
        test_model_score = accuracy_score(y_test, y_test_pred)
    
        accuracy_report[list(models.keys())[i]] = [test_model_score]
        params_report[list(models.keys())[i]] = gs.best_params_
        
        with open('Config\Parameters.yaml', 'w') as f:
            yaml.dump(params_report, f)

    return (models, accuracy_report, params_report)

def model_trainer(models, accuracy_report, params_report):
    best_accuracy = max(sorted(accuracy_report.values()))
    best_model_name = list(accuracy_report.keys())[
        list(accuracy_report.values()).index(best_accuracy)
    ]   
    best_model = models[best_model_name]
    params = params_report[best_model_name]
    model = best_model
    model.set_params(**params)

    model.fit(X_train, y_train)

    with open('Models\Saved_Models\model.pkl', 'wb') as file:
        pickle.dump(model, file)

    print("Model trained successfully with parameters {}".format(params))



X_train, X_test, y_train, y_test = train_test()

models, accuracy_report, params_report = models_report(X_train, X_test, y_train, y_test)

model_trainer(models, accuracy_report, params_report)