#!/usr/bin/env python3
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import numpy as np
def perfomance_metrics(classifier, y_, y_pred):
    
    accuracy = round(accuracy_score(y_, y_pred), 5)
    precision = round(precision_score(y_, y_pred), 5)
    recall = round(recall_score(y_, y_pred), 5)
    f1 = round(f1_score(y_, y_pred), 5)
    auc = round(roc_auc_score(y_, y_pred), 5)
    
    print("\n",classifier.__class__.__name__)
    if hasattr(classifier, 'best_params_'):
        print("Best Model: ", classifier.best_params_)
    print("\nConfusion Matrix:\n", confusion_matrix(y_, y_pred))
    print("\nAccuracy: " , accuracy)
    print("\nPrecision: ", precision)
    print("\nRecall: ", recall)
    print("\nF1: ", f1)
    print("\nAUC: ", auc)

    return np.array([classifier.__class__.__name__, accuracy, precision, recall, f1, auc])
