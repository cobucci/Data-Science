from performance_metrics import perfomance_metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

def voting_classifier(type, X_train, X_test, y_train, y_test):

    if type == 0:
        log_clf = LogisticRegression()
        rnd_clf = RandomForestClassifier()
        svm_clf = SVC()
        tree_clf = DecisionTreeClassifier()
        gb_clf = GradientBoostingClassifier()

        voting_clf = VotingClassifier(estimators=[('lr', log_clf), ('rf', rnd_clf), ('svm', svm_clf), ('tree', tree_clf), ('gb', GradientBoostingClassifier)], voting='hard')
        voting_clf.fit(X_train, y_train)
        y_pred = voting_clf.predict(X_test)

    return perfomance_metrics(voting_clf, y_test, y_pred)

        


