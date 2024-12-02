import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pyod.utils import standardizer
from pyod.utils.data import generate_data
import scipy.io
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from sklearn.metrics import confusion_matrix, roc_auc_score
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, balanced_accuracy_score
from pyod.utils.data import generate_data

# Exercise 1

def plot_3d_data(X_train, X_test, y_train, y_test, y_train_pred, y_test_pred, title_suffix):
    fig = plt.figure(figsize=(14, 10))

    ax1 = fig.add_subplot(221, projection='3d')
    ax1.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], X_train[y_train == 0, 2], c='blue', label='Inliers')
    ax1.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], X_train[y_train == 1, 2], c='red', label='Outliers')
    ax1.set_title(f'Ground Truth - Training Data {title_suffix}')
    ax1.legend()

    ax2 = fig.add_subplot(222, projection='3d')
    ax2.scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], X_test[y_test == 0, 2], c='blue', label='Inliers')
    ax2.scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], X_test[y_test == 1, 2], c='red', label='Outliers')
    ax2.set_title(f'Ground Truth - Test Data {title_suffix}')
    ax2.legend()

    ax3 = fig.add_subplot(223, projection='3d')
    ax3.scatter(X_train[y_train_pred == 0, 0], X_train[y_train_pred == 0, 1], X_train[y_train_pred == 0, 2], c='blue', label='Inliers')
    ax3.scatter(X_train[y_train_pred == 1, 0], X_train[y_train_pred == 1, 1], X_train[y_train_pred == 1, 2], c='red', label='Outliers')
    ax3.set_title(f'Predicted Labels - Training Data {title_suffix}')
    ax3.legend()

    ax4 = fig.add_subplot(224, projection='3d')
    ax4.scatter(X_test[y_test_pred == 0, 0], X_test[y_test_pred == 0, 1], X_test[y_test_pred == 0, 2], c='blue', label='Inliers')
    ax4.scatter(X_test[y_test_pred == 1, 0], X_test[y_test_pred == 1, 1], X_test[y_test_pred == 1, 2], c='red', label='Outliers')
    ax4.set_title(f'Predicted Labels - Test Data {title_suffix}')
    ax4.legend()

    plt.tight_layout()
    plt.show()

def compute_metrics(y_test, y_test_pred, y_test_scores):
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    balanced_acc = (tpr + tnr) / 2
    roc_auc = roc_auc_score(y_test, y_test_scores)
    return balanced_acc, roc_auc

def ex1():
    X_train, X_test, y_train, y_test = generate_data(n_train=300, n_test=200, n_features=3, contamination=0.15)

    # OCSVM with linear kernel
    clf_linear = OCSVM(kernel='linear', contamination=0.15)
    clf_linear.fit(X_train)
    y_train_pred_linear = clf_linear.labels_
    y_test_pred_linear = clf_linear.predict(X_test)
    y_test_scores_linear = clf_linear.decision_function(X_test)
    balanced_acc_linear, roc_auc_linear = compute_metrics(y_test, y_test_pred_linear, y_test_scores_linear)
    print(f"OCSVM Linear Kernel - Balanced Accuracy: {balanced_acc_linear}, ROC AUC: {roc_auc_linear}")
    plot_3d_data(X_train, X_test, y_train, y_test, y_train_pred_linear, y_test_pred_linear, "OCSVM Linear")

    # OCSVM with RBF kernel
    clf_rbf = OCSVM(kernel='rbf', contamination=0.15)
    clf_rbf.fit(X_train)
    y_train_pred_rbf = clf_rbf.labels_
    y_test_pred_rbf = clf_rbf.predict(X_test)
    y_test_scores_rbf = clf_rbf.decision_function(X_test)
    balanced_acc_rbf, roc_auc_rbf = compute_metrics(y_test, y_test_pred_rbf, y_test_scores_rbf)
    print(f"OCSVM RBF Kernel - Balanced Accuracy: {balanced_acc_rbf}, ROC AUC: {roc_auc_rbf}")
    plot_3d_data(X_train, X_test, y_train, y_test, y_train_pred_rbf, y_test_pred_rbf, "OCSVM RBF")

    # DeepSVDD
    clf_deepsvdd = DeepSVDD(contamination=0.15, n_features=3)
    clf_deepsvdd.fit(X_train)
    y_train_pred_deepsvdd = clf_deepsvdd.labels_
    y_test_pred_deepsvdd = clf_deepsvdd.predict(X_test)
    y_test_scores_deepsvdd = clf_deepsvdd.decision_function(X_test)
    balanced_acc_deepsvdd, roc_auc_deepsvdd = compute_metrics(y_test, y_test_pred_deepsvdd,y_test_scores_deepsvdd)
    print(f"DeepSVDD - Balanced Accuracy: {balanced_acc_deepsvdd}, ROC AUC: {roc_auc_deepsvdd}")
    plot_3d_data(X_train, X_test, y_train, y_test, y_train_pred_deepsvdd, y_test_pred_deepsvdd,"DeepSVDD")


# Exercise 2

def convert_labels_sklearn_to_pyod(labels):
    return (-1 * labels + 1) // 2

def balanced_accuracy_scorer(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return (tpr + tnr) / 2


def ex2():
    data = scipy.io.loadmat('cardio.mat')
    X = data['X']
    y = data['y'].ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.4, test_size=0.6, random_state=None, shuffle=True,stratify=None)
    y_train = y_train.reshape(-1, 1)
    y_test = y_test.reshape(-1, 1)

    param_grid = {
        'ocsvm__kernel': ['linear', 'rbf', 'poly'],
        'ocsvm__gamma': ['scale', 'auto', 0.1, 1, 10],
        'ocsvm__nu': [0.1, 0.15, 0.2, 0.25]
    }

    pipeline = Pipeline([('scaler', StandardScaler()),
                         ('ocsvm', OneClassSVM())
                         ])

    grid_search = GridSearchCV(pipeline, param_grid, scoring=make_scorer(balanced_accuracy_scorer), cv=5)
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best parameters: {best_params}")
    print(f"Best cross-validated balanced accuracy: {best_score}")

    best_model = grid_search.best_estimator_
    y_test_pred = best_model.predict(X_test)
    y_test_pred_pyod = convert_labels_sklearn_to_pyod(y_test_pred)
    balanced_acc_test = balanced_accuracy_scorer(y_test, y_test_pred)

    print(f"Balanced accuracy on the test set: {balanced_acc_test}")


# Exercise 3

def ex3():
    data = scipy.io.loadmat('shuttle.mat')
    X = data['X']
    y = data['y'].ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.5, test_size=0.5, random_state=None, shuffle=True, stratify=None)
    X_train, X_test = standardizer(X_train, X_test)

    # OCSVM
    clf_ocsvm = OCSVM(kernel='rbf', contamination=0.15)
    clf_ocsvm.fit(X_train)
    y_test_pred_ocsvm = clf_ocsvm.predict(X_test)
    y_test_scores_ocsvm = clf_ocsvm.decision_function(X_test)
    balanced_acc_ocsvm = balanced_accuracy_scorer(y_test, y_test_pred_ocsvm)
    roc_auc_ocsvm = roc_auc_score(y_test, y_test_scores_ocsvm)
    print(f"OCSVM - Balanced Accuracy: {balanced_acc_ocsvm}, ROC AUC: {roc_auc_ocsvm}")

    # DeepSVDD
    architectures = [
        {'hidden_neurons': [32, 16]},
        {'hidden_neurons': [64, 32, 16]},
        {'hidden_neurons': [128, 64, 32, 16]}
    ]

    for arch in architectures:
        clf_deepsvdd = DeepSVDD(contamination=0.15, n_features=X_train.shape[1],hidden_neurons=arch['hidden_neurons'])
        clf_deepsvdd.fit(X_train)
        y_test_pred_deepsvdd = clf_deepsvdd.predict(X_test)
        y_test_scores_deepsvdd = clf_deepsvdd.decision_function(X_test)
        balanced_acc_deepsvdd = balanced_accuracy_scorer(y_test, y_test_pred_deepsvdd)
        roc_auc_deepsvdd = roc_auc_score(y_test, y_test_scores_deepsvdd)
        print(f"DeepSVDD {arch['hidden_neurons']} - Balanced Accuracy: {balanced_acc_deepsvdd}, ROC AUC: {roc_auc_deepsvdd}")

#ex1()
ex2()
#ex3()

