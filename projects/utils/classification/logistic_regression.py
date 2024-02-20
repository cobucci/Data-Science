from performance_metrics import perfomance_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
import numpy as np

def logistic_regression_classifier(type, X_train, X_test, y_train, y_test):

    start_time = time.time()

    parameters = {  'penalty': ['l1', 'l2', 'elasticnet'],
                    'C' : [0.1, 0.3, 0.5,0.7, 1],
                    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga'],
                    'max_iter': [100, 120, 150, 200],
                    'n_jobs': -1
                }
    
    if type == 0:
        logistic_regression_clf = LogisticRegression()
    if type == 1:
        logistic_regression_clf = RandomizedSearchCV(LogisticRegression(), parameters, cv=5, n_iter=100, random_state=42, n_jobs=-1)
    if type == 2:
        logistic_regression_clf = GridSearchCV(LogisticRegression(), parameters, cv=5, n_jobs=-1)
    

    logistic_regression_clf.fit(X_train, y_train)
    y_pred = logistic_regression_clf.predict(X_test)
    print("Best Estimator: ", logistic_regression_clf.best_params_)
    
    duration = time.time() - start_time
    print("Computation Time = ", duration)

    return perfomance_metrics(logistic_regression_clf, y_test, y_pred)        


