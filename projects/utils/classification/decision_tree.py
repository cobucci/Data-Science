from sklearn.tree import DecisionTreeClassifier
from performance_metrics import perfomance_metrics
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import time

def decision_tree_classifier(type, X_train, X_test, y_train, y_test):
    
    start_time = time.time()

    parameters = {  'criterion':['gini','entropy'],
                    'splitter': ['best', 'random'],
                    'max_depth':np.arange(1,30).tolist()[0::2],
                    'min_samples_split':np.arange(1,30).tolist()[1::2],
                    'min_samples_leaf': np.arange(1,30).tolist()[0::2],
                    'max_leaf_nodes':np.arange(3,30).tolist()[0::2] }

    if type == 0:
        tree_clf = DecisionTreeClassifier()   
    if type == 1:
        tree_clf = RandomizedSearchCV(DecisionTreeClassifier(), parameters, cv=3, n_iter=15, random_state=42, n_jobs=-1)
        tree_clf.__class__.__name__ = "DecisionTreeClassifier"
    if type == 2:
        tree_clf = GridSearchCV(DecisionTreeClassifier(), parameters, cv=3, n_jobs=-1)
        tree_clf.__class__.__name__ = "DecisionTreeClassifier"

    tree_clf.fit(X_train, y_train)
    y_pred = tree_clf.predict(X_test)
    
    duration = time.time() - start_time
    print("Computation Time = ", duration)

    return perfomance_metrics(tree_clf, y_test, y_pred)
        


