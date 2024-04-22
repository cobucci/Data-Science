from sklearn.ensemble import BaggingClassifier
from performance_metrics import perfomance_metrics
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time
from sklearn.ensemble import RandomForestClassifier

def bagging_classifier(type, X_train, X_test, y_train, y_test):

    start_time = time.time()

    parameters = { 'estimator': [DecisionTreeClassifier()
                                ,RandomForestClassifier()],
                    'n_estimators': np.arange(1,1000).tolist()[0::100],
                    'max_samples' : np.arange(2,30).tolist()[1::2],
                    'bootstrap_features': [True, False],
                    'oob_score': [True, False],
    }
    
    if type == 0:
        bagging_clf = BaggingClassifier(RandomForestClassifier())
    if type == 1:
        bagging_clf = RandomizedSearchCV(BaggingClassifier(), parameters, cv=3, n_iter=15, random_state=42, n_jobs=-1)
        bagging_clf.__class__.__name__ = "BaggingClassifier"
    if type == 2:
        bagging_clf = GridSearchCV(BaggingClassifier(), parameters, cv=3, n_jobs=-1)
        bagging_clf.__class__.__name__ = "BaggingClassifier"
    

    bagging_clf.fit(X_train, y_train)
    y_pred = bagging_clf.predict(X_test)

    duration = time.time() - start_time
    print("Computation Time = ", duration)

    return perfomance_metrics(bagging_clf, y_test, y_pred)


