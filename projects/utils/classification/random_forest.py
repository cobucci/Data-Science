from sklearn.ensemble import RandomForestClassifier
from performance_metrics import perfomance_metrics
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time

def random_forest_classifier(type, X_train, X_test, y_train, y_test):

    start_time = time.time()

    parameters = {
                'n_estimators':np.arange(1,1000).tolist()[0::50],
                'max_samples' : np.arange(2,30).tolist()[1::2],
                'max_features': np.arange(2,30).tolist()[1::2],
                'warm_start': [True, False]
            }
    if type == 0:
        random_forest_clf = RandomForestClassifier()  
    if type == 1:
        random_forest_clf = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=3, n_iter=15, random_state=42, n_jobs=-1)
        random_forest_clf.__class__.__name__ = "RandomForestClassifier"
    if type == 2:
        random_forest_clf = GridSearchCV(RandomForestClassifier(), parameters, cv=3, n_jobs=-1)
        random_forest_clf.__class__.__name__ = "RandomForestClassifier"

    random_forest_clf.fit(X_train, y_train)
    y_pred = random_forest_clf.predict(X_test)

    duration = time.time() - start_time
    print("Computation Time = ", duration)

    return perfomance_metrics(random_forest_clf, y_test, y_pred)

        


