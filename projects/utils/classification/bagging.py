from sklearn.ensemble import BaggingClassifier
from performance_metrics import perfomance_metrics
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
from sklearn.ensemble import RandomForestClassifier

def bagging_classifier(type, X_train, X_test, y_train, y_test):

    start_time = time.time()

    parameters = {  'estimator': [DecisionTreeClassifier(), RandomForestClassifier()],
                    'n_estimators': [10, 15, 20, 30, 50, 100],
                    'max_samples' : np.arange(2,30).tolist()[1::2],
                    'max_features': np.arange(2,30).tolist()[1::2],
                    'bootstrap': [True, False],
                    'bootstrap_features': [True, False],
                    'oob_score': [True, False],
                    'warm_start': [True, False],
                    'n_jobs': -1}
    
    if type == 0:
        bagging_clf = BaggingClassifier(DecisionTreeClassifier())
    if type == 1:
        bagging_clf = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=5, n_iter=100, random_state=42, n_jobs=-1)
    if type == 2:
        bagging_clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5, n_jobs=-1)
    

    bagging_clf.fit(X_train, y_train)
    y_pred = bagging_clf.predict(X_test)
    print("Best Estimator: ", bagging_clf.best_params_)
    
    duration = time.time() - start_time
    print("Computation Time = ", duration)

    return perfomance_metrics(bagging_clf, y_test, y_pred)


