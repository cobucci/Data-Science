from performance_metrics import perfomance_metrics
import xgboost as xgb 
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import numpy as np

def xgboosting_classifier(type, X_train, X_test, y_train, y_test):
    
    start_time = time.time()

    parameters = {
        'eta': [0.001, 0.05, 0.1, 0.3, 0.5, 0.7, 0.9],
        'max_depth': np.arange(2,20).tolist()[1::2]
    }
    
    if type == 0:
        xgb_clf = xgb.XGBClassifier()
    if type == 1:
        xgb_clf = RandomizedSearchCV(xgb.XGBClassifier(), parameters, cv=3, n_iter=15, random_state=42, n_jobs=-1)
        xgb_clf.__class__.__name__ = "XGBClassifier"
    if type == 2:
        xgb_clf = GridSearchCV(xgb.XGBClassifier(), parameters, cv=5, n_jobs=-1)
        xgb_clf.__class__.__name__ = "XGBClassifier"

    xgb_clf.fit(X_train, y_train)
    y_pred = xgb_clf.predict(X_test)

    duration = time.time() - start_time
    print("Computation Time = ", duration)

    return perfomance_metrics(xgb_clf, y_test, y_pred)
        


