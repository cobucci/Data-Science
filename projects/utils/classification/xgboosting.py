from performance_metrics import perfomance_metrics
import xgboost as xgb 

def xgboosting_classifier(type, X_train, X_test, y_train, y_test):

    if type == 0:
        xgb_clf = xgb.XGBClassifier()
        xgb_clf.fit(X_train, y_train)
        y_pred = xgb_clf.predict(X_test)

    return perfomance_metrics(xgb_clf, y_test, y_pred)

        


