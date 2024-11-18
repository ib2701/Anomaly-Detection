import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA
from pyod.utils.utility import standardizer
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix



# Ex1
def generate_projections(n_projections, n_features):
    projections = np.random.multivariate_normal(((np.zeros(n_features))), np.identity(n_features), n_projections)
    return projections

def compute_histograms(projections, data, n_bins=10):
    histograms = []
    for projection in projections:
        projected_data = data.dot(projection)
        hist, bin_edges = np.histogram(projected_data, bins=n_bins, range=(-10, 10))
        histograms.append((hist, bin_edges, projection))
    return histograms

def compute_anomaly_scores(data, histograms):
    iforest = IForest(n_estimators=100, max_samples=256)
    iforest.fit(data)
    anomaly_scores = iforest.decision_function(data)
    return anomaly_scores

def ex1():
    X, _ = make_blobs(n_samples=500, n_features=2, centers=None, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=None, return_centers=False)
    X = standardizer(X)

    projections = generate_projections(5, 2)
    histograms = compute_histograms(projections, X)
    anomaly_scores = compute_anomaly_scores(X, histograms)
    test_data = np.random.uniform(-3, 3, (500, 2))
    test_data = standardizer(test_data)
    test_anomaly_scores = compute_anomaly_scores(test_data, histograms)

    plt.scatter(test_data[:, 0], test_data[:, 1], c=test_anomaly_scores, cmap='viridis')
    plt.colorbar(label='Anomaly Score')
    plt.title('Anomaly Scores for Test Dataset')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

    for n_bins in [5, 10, 20]:
        histograms = compute_histograms(projections, X, n_bins=n_bins)
        test_anomaly_scores = compute_anomaly_scores(test_data, histograms)
        plt.scatter(test_data[:, 0], test_data[:, 1], c=test_anomaly_scores, cmap='viridis')
        plt.colorbar(label='Anomaly Score')
        plt.title(f'Anomaly Scores for Test Dataset with {n_bins} Bins')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.show()

    model = IForest(n_estimators=100, max_samples=256, contamination=0.1, random_state=42)
    model.fit(X)
    deep_iforest_scores = model.decision_function(test_data)

    plt.scatter(test_data[:, 0], test_data[:, 1], c=deep_iforest_scores, cmap='viridis')
    plt.colorbar(label='Anomaly Score')
    plt.title('Anomaly Scores from Deep Isolation Forest')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


# Ex2

def fit_iforest(X, contamination=0.02):
    iforest = IForest(contamination=contamination, random_state=42)
    iforest.fit(X)
    return iforest

def fit_dif(X, contamination=0.02, hidden_neurons=[64, 32]):
    dif = DIF(contamination=contamination, hidden_neurons=hidden_neurons, random_state=42)
    dif.fit(X)
    return dif

def fit_loda(X, contamination=0.02, n_bins=10):
    loda = LODA(contamination=contamination, n_bins=n_bins)
    loda.fit(X)
    return loda

def plot_anomaly_scores(test_data, scores, title, subplot_position):
    plt.subplot(1, 3, subplot_position)
    plt.scatter(test_data[:, 0], test_data[:, 1], c=scores, cmap='viridis')
    plt.colorbar(label='Anomaly Score')
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')

def ex2():
    X, _ = make_blobs(n_samples=1000, centers=[(10, 0), (0, 10)], cluster_std=1.0, shuffle=True, random_state=None)
    test_data = np.random.uniform(-10, 20, (1000, 2))

    iforest = fit_iforest(X)
    iforest_scores = iforest.decision_function(test_data)

    dif = fit_dif(X)
    dif_scores = dif.decision_function(test_data)

    loda = fit_loda(X)
    loda_scores = loda.decision_function(test_data)

    plt.figure(figsize=(15, 5))
    plot_anomaly_scores(test_data, iforest_scores, 'IForest Anomaly Scores', 1)
    plot_anomaly_scores(test_data, dif_scores, 'DIF Anomaly Scores', 2)
    plot_anomaly_scores(test_data, loda_scores, 'LODA Anomaly Scores', 3)
    plt.tight_layout()
    plt.show()

    dif = fit_dif(X, hidden_neurons=[128, 64])
    dif_scores = dif.decision_function(test_data)

    loda = fit_loda(X, n_bins=20)
    loda_scores = loda.decision_function(test_data)

    plt.figure(figsize=(15, 5))
    plot_anomaly_scores(test_data, dif_scores, 'DIF Anomaly Scores with More Neurons', 1)
    plot_anomaly_scores(test_data, loda_scores, 'LODA Anomaly Scores with More Bins', 2)
    plt.tight_layout()
    plt.show()

# Ex2 3D
def plot_anomaly_scores_3d(test_data, scores, title, subplot_position):
    ax = plt.subplot(1, 3, subplot_position, projection='3d')
    sc = ax.scatter(test_data[:, 0], test_data[:, 1], test_data[:, 2], c=scores, cmap='viridis')
    plt.colorbar(sc, ax=ax, label='Anomaly Score')
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')

def ex2_3d():
    X, _ = make_blobs(n_samples=1000, centers=[(0, 10, 0), (10, 0, 10)], cluster_std=1.0, shuffle=True, random_state=None)
    test_data = np.random.uniform(-10, 20, (1000, 3))

    iforest = fit_iforest(X)
    iforest_scores = iforest.decision_function(test_data)

    dif = fit_dif(X)
    dif_scores = dif.decision_function(test_data)

    loda = fit_loda(X)
    loda_scores = loda.decision_function(test_data)

    fig = plt.figure(figsize=(15, 5))
    plot_anomaly_scores_3d(test_data, iforest_scores, 'IForest Anomaly Scores', 1)
    plot_anomaly_scores_3d(test_data, dif_scores, 'DIF Anomaly Scores', 2)
    plot_anomaly_scores_3d(test_data, loda_scores, 'LODA Anomaly Scores', 3)
    plt.tight_layout()
    plt.show()

    dif = fit_dif(X, hidden_neurons=[128, 64])
    dif_scores = dif.decision_function(test_data)

    loda = fit_loda(X, n_bins=20)
    loda_scores = loda.decision_function(test_data)

    fig = plt.figure(figsize=(15, 5))
    plot_anomaly_scores_3d(test_data, dif_scores, 'DIF Anomaly Scores with More Neurons', 1)
    plot_anomaly_scores_3d(test_data, loda_scores, 'LODA Anomaly Scores with More Bins', 2)
    plt.tight_layout()
    plt.show()

#Ex3
def compute_balanced_accuracy(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return (tpr + tnr) / 2

def fit_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train)
    y_pred = model.predict(X_test)
    y_scores = model.decision_function(X_test)
    ba = compute_balanced_accuracy(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_scores)
    return ba, roc_auc

def ex3():
    ba_iforest, roc_auc_iforest = [], []
    ba_dif, roc_auc_dif = [], []
    ba_loda, roc_auc_loda = [], []
    data = loadmat('shuttle.mat')
    X = data['X']
    y = data['y'].ravel()

    for i in range(10):
        print(i)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=None, shuffle=True, stratify=None)
        X_train, X_test = standardizer(X_train, X_test)

        iforest = IForest(contamination=0.02)
        ba, roc_auc = fit_and_evaluate_model(iforest, X_train, y_train, X_test, y_test)
        ba_iforest.append(ba)
        roc_auc_iforest.append(roc_auc)

        dif = DIF(contamination=0.02, hidden_neurons=[64, 32])
        ba, roc_auc = fit_and_evaluate_model(dif, X_train, y_train, X_test, y_test)
        ba_dif.append(ba)
        roc_auc_dif.append(roc_auc)

        loda = LODA(contamination=0.02, n_bins=10)
        ba, roc_auc = fit_and_evaluate_model(loda, X_train, y_train, X_test, y_test)
        ba_loda.append(ba)
        roc_auc_loda.append(roc_auc)

    print(f'IForest - Mean BA: {np.mean(ba_iforest):.4f}, Mean ROC AUC: {np.mean(roc_auc_iforest):.4f}')
    print(f'DIF - Mean BA: {np.mean(ba_dif):.4f}, Mean ROC AUC: {np.mean(roc_auc_dif):.4f}')
    print(f'LODA - Mean BA: {np.mean(ba_loda):.4f}, Mean ROC AUC: {np.mean(roc_auc_loda):.4f}')



ex1()
#ex2()
#ex2_3d()
#ex3()
