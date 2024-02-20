from performance_metrics import perfomance_metrics
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import time
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

def gradient_boosting_classifier(type, X_train, X_test, y_train, y_test):

    start_time = time.time()

    parameters = {  'loss': {'log_loss', 'exponential'},
                    'learning_rate': [0.0001, 0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.07, 1],
                    'n_estimators':np.arange(1,1000).tolist()[0::50],
                    'criterion': ['friedman_mse', 'squared_error'],
                    'min_samples_split':np.arange(2,30).tolist()[1::2],
                    'min_samples_leaf': np.arange(1,30).tolist()[0::2],
                    'max_depth':np.arange(1,30).tolist()[0::2],
                    'max_features':  ['sqrt', 'log2', None],
                    'n_jobs': -1}
    
    if type == 0:
        gradient_boosting_clf = GradientBoostingClassifier()
    if type == 1:
        bagging_clf = RandomizedSearchCV(GradientBoostingClassifier(), parameters, cv=5, n_iter=100, random_state=42, n_jobs=-1)
    if type == 2:
        bagging_clf = GridSearchCV(GradientBoostingClassifier(), parameters, cv=5, n_jobs=-1)

   
    gradient_boosting_clf.fit(X_train, y_train)
    y_pred = gradient_boosting_clf.predict(X_test)
    print("Best Estimator: ", gradient_boosting_clf.best_params_)
    
    duration = time.time() - start_time
    print("Computation Time = ", duration)

    return perfomance_metrics(bagging_clf, y_test, y_pred)

        


