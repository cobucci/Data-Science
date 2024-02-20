from performance_metrics import perfomance_metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time

def ada_boosting_classifier(type, X_train, X_test, y_train, y_test):

    start_time = time.time()

    parameters = {  'n_estimators': [10, 25, 50, 75, 100, 200],
                    'learning_rate': [0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 1],
                    'algorithm' : 'SAMME, SAMME.R'}
    
    if type == 0:
        ada_clf = AdaBoostClassifier(DecisionTreeClassifier())
    if type == 1:
        ada_clf = RandomizedSearchCV(DecisionTreeClassifier(), parameters, cv=5, n_iter=100, random_state=42, n_jobs=-1)
    if type == 2:
        ada_clf = GridSearchCV(DecisionTreeClassifier(), parameters, cv=5, n_jobs=-1)
    

    ada_clf.fit(X_train, y_train)
    y_pred = ada_clf.predict(X_test)
    print("Best Estimator: ", ada_clf.best_params_)
    
    duration = time.time() - start_time
    print("Computation Time = ", duration)

    return perfomance_metrics(ada_clf, y_test, y_pred)


        


