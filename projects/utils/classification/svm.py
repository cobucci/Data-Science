from performance_metrics import perfomance_metrics
from sklearn.svm import SVC

def svm_classifier(type, X_train, X_test, y_train, y_test):

    if type == 0:
        svm_clf = SVC()
        svm_clf.fit(X_train, y_train)
        y_pred = svm_clf.predict(X_test)

    return perfomance_metrics(svm_clf, y_test, y_pred)

        


