from sklearn.ensemble import RandomForestClassifier
from performance_metrics import perfomance_metrics
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time

def random_forest_classifier(type, X_train, X_test, y_train, y_test):

    start_time = time.time()

    parameters = {  'n_estimators':np.arange(1,1000).tolist()[0::50],
                    'max_depth':np.arange(1,30).tolist()[0::2],
                    'min_samples_split':np.arange(2,30).tolist()[1::2],
                    'min_samples_leaf': np.arange(1,30).tolist()[0::2],
                    'max_leaf_nodes':np.arange(3,30).tolist()[0::2],
                    'max_features':  ['sqrt', 'log2', None] }
    if type == 0:
        random_forest_clf = RandomForestClassifier(max_depth=2)  
    if type == 1:
        random_forest_clf = RandomizedSearchCV(RandomForestClassifier(), parameters, cv=5, n_iter=100, random_state=42, n_jobs=-1)
    if type == 2:
        random_forest_clf = GridSearchCV(RandomForestClassifier(), parameters, cv=5, n_jobs=-1)
    

    random_forest_clf.fit(X_train, y_train)
    y_pred = random_forest_clf.predict(X_test)
    print("Best Estimator: ", random_forest_clf.best_params_)
    
    duration = time.time() - start_time
    print("Computation Time = ", duration)

    return perfomance_metrics(random_forest_clf, y_test, y_pred)

        


