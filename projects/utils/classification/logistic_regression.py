from performance_metrics import perfomance_metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
import numpy as np

def logistic_regression_classifier(type, X_train, X_test, y_train, y_test):

    start_time = time.time()

    #lbfgs - [l2, None]
    #liblinear - [l1, l2]
    #newton-cg - [l2, None]
    #newton-cholesky - [l2, None]
    #sag - [l2, None]
    #saga - [elasticnet, l1, l2, None]

    parameters = {'penalty': ['l2'],
                    'C' : [0.01, 0.1, 0.3, 0.5,0.7, 1],
                    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'saga']
    }
    
    if type == 0:
        logistic_regression_clf = LogisticRegression()
    if type == 1:
        logistic_regression_clf = RandomizedSearchCV(LogisticRegression(), parameters, cv=3, random_state=42, n_jobs=-1, error_score=0, n_iter=20)
        logistic_regression_clf.__class__.__name__ = "LogisticRegression"
    if type == 2:
        logistic_regression_clf = GridSearchCV(LogisticRegression(), parameters, cv=3, n_jobs=-1)
        logistic_regression_clf.__class__.__name__ = "LogisticRegression"
    

    logistic_regression_clf.fit(X_train, y_train)
    y_pred = logistic_regression_clf.predict(X_test)
    
    duration = time.time() - start_time
    print("Computation Time = ", duration)

    return perfomance_metrics(logistic_regression_clf, y_test, y_pred)        


