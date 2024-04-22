from performance_metrics import perfomance_metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

def voting_classifier(type, X_train, X_test, y_train, y_test, logistic_regression_clf, tree_clf, random_forest_clf, gradient_boosting_clf):

    if type == 0:
        vt_lr_clf = LogisticRegression()

        vt_dt_clf = DecisionTreeClassifier()

        vt_rf_clf = RandomForestClassifier()

        vt_gb_clf = GradientBoostingClassifier()

    if type >= 1:
        vt_lr_clf = LogisticRegression(solver='newton-cg', 
                                                penalty='l2', 
                                                C=1)

        vt_dt_clf = DecisionTreeClassifier(splitter='best', min_samples_split= 28, min_samples_leaf=9, max_leaf_nodes=27, max_depth=29, criterion='gini')

        vt_rf_clf = RandomForestClassifier(n_estimators=601, min_samples_split= 17, min_samples_leaf= 25, max_leaf_nodes= 19, max_features= None, max_depth=11)
        
        vt_gb_clf = GradientBoostingClassifier(min_samples_split=27, min_samples_leaf= 19, max_features= None, max_depth= 7, loss= 'exponential', learning_rate= 0.02, criterion='friedman_mse')

    voting_clf = VotingClassifier(estimators=[('lr', vt_lr_clf), ('tree', vt_dt_clf), ('rf', vt_rf_clf), ('gb', vt_gb_clf)], voting='hard', n_jobs=-1)
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)

    return perfomance_metrics(voting_clf, y_test, y_pred)

        


