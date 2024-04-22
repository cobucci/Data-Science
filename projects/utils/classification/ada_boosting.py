from performance_metrics import perfomance_metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
from sklearn.ensemble import RandomForestClassifier

def ada_boosting_classifier(type, X_train, X_test, y_train, y_test):

    start_time = time.time()

    parameters = {  'estimator': [DecisionTreeClassifier(),RandomForestClassifier()],
                    #'n_estimators':np.arange(1,1000).tolist()[0::50],
                    'learning_rate': [0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 1],
                    'algorithm' : ['SAMME', 'SAMME.R']}
    
    if type == 0:
        ada_clf = AdaBoostClassifier(AdaBoostClassifier())
    if type == 1:
        ada_clf = RandomizedSearchCV(AdaBoostClassifier(), parameters, cv=3, n_iter=15, random_state=42, n_jobs=-1)
        ada_clf.__class__.__name__ = "AdaBoostClassifier"
    if type == 2:
        ada_clf = GridSearchCV(AdaBoostClassifier(), parameters, cv=3, n_jobs=-1)
        ada_clf.__class__.__name__ = "AdaBoostClassifier"
    

    ada_clf.fit(X_train, y_train)
    y_pred = ada_clf.predict(X_test)
    
    duration = time.time() - start_time
    print("Computation Time = ", duration)

    return perfomance_metrics(ada_clf, y_test, y_pred)


        


