import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_blobs
from pyod.models.lof import LOF
from sklearn.model_selection import train_test_split
import scipy.io
from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization


# Ex 1
def generate_data(a, b, mu, sigma2, n=100):
    x1 = np.random.randn(n)
    epsilon = np.random.normal(mu, np.sqrt(sigma2), n)
    y = a * x1 + b + epsilon
    return x1, y

def leverage_scores(X):
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    leverage = np.sum(U[:, :X.shape[1]] ** 2, axis=1)
    return leverage

def plot_data(x1, y, leverage, title, ax):
    ax.scatter(x1, y, c=leverage, cmap='viridis', s=50)
    high_leverage_points = np.argsort(leverage)[-5:]
    ax.scatter(x1[high_leverage_points], y[high_leverage_points], color='red')
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('y')

def ex1():
    a, b = 2, 1
    mu_values = [0, 0, 0, 0]
    sigma2_values = [0.1, 1, 10, 100]
    titles = ['Low noise', 'High variance on x', 'High variance on y', 'High variance on both x and y']
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    for i, (mu, sigma2, title) in enumerate(zip(mu_values, sigma2_values, titles)):
        x1, y = generate_data(a, b, mu, sigma2)
        X = np.column_stack((np.ones_like(x1), x1))
        leverage = leverage_scores(X)
        plot_data(x1, y, leverage, title, axs[i])

    plt.tight_layout()
    plt.show()

# Ex 1 - 2D
def generate_data_2d(a, b, c, mu, sigma2, n=100):
    x1 = np.random.randn(n)
    x2 = np.random.randn(n)
    epsilon = np.random.normal(mu, np.sqrt(sigma2), n)
    y = a * x1 + b * x2 + c + epsilon
    return x1, x2, y

def plot_data_2d(x1, x2, y, leverage, title, ax):
    scatter = ax.scatter(x1, x2, c=leverage, cmap='viridis', s=50)
    high_leverage_points = np.argsort(leverage)[-5:]
    ax.scatter(x1[high_leverage_points], x2[high_leverage_points], color='red')
    ax.set_title(title)
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.colorbar(scatter, ax=ax, label='Leverage Score')

def ex1_2d():
    a, b, c = 2, 3, 1
    mu_values = [0, 0, 0, 0]
    sigma2_values = [0.1, 1, 10, 100]
    titles = ['Low noise', 'High variance on x1', 'High variance on x2', 'High variance on both x1 and x2']
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()
    for i, (mu, sigma2, title) in enumerate(zip(mu_values, sigma2_values, titles)):
        x1, x2, y = generate_data_2d(a, b, c, mu, sigma2)
        X = np.column_stack((np.ones_like(x1), x1, x2))
        leverage = leverage_scores(X)
        plot_data_2d(x1, x2, y, leverage, title, axs[i])

    plt.tight_layout()
    plt.show()


# Ex 2
def plot_results(X, y_true, y_pred, title, ax):
    ax.scatter(X[:, 0], X[:, 1], c=y_true, cmap='coolwarm', marker='o', label='Ground Truth')
    ax.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='coolwarm', marker='x', label='Predicted')
    ax.set_title(title)
    ax.legend()


def balanced_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    TN, FP, FN, TP = cm.ravel()
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    return (TPR + TNR) / 2


def ex2():
    X_train, X_test, y_train, y_test = generate_data_clusters(n_train=400, n_test=200, n_features=2, n_clusters=2, contamination=0.1)

    n_neighbors_values = [5, 10, 20]

    fig, axs = plt.subplots(len(n_neighbors_values), 4, figsize=(20, 15))

    for i, n_neighbors in enumerate(n_neighbors_values):
        model = KNN(n_neighbors=n_neighbors, contamination=0.1)
        model.fit(X_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_bal_acc = balanced_accuracy(y_train, y_train_pred)
        test_bal_acc = balanced_accuracy(y_test, y_test_pred)

        plot_results(X_train, y_train, y_train_pred, f'Train Ground Truth (n_neighbors={n_neighbors})', axs[i, 0])
        plot_results(X_train, y_train, y_train_pred, f'Train Predicted (n_neighbors={n_neighbors})', axs[i, 1])
        plot_results(X_test, y_test, y_test_pred, f'Test Ground Truth (n_neighbors={n_neighbors})', axs[i, 2])
        plot_results(X_test, y_test, y_test_pred, f'Test Predicted (n_neighbors={n_neighbors})', axs[i, 3])

        print(f'n_neighbors={n_neighbors}, Train Balanced Accuracy: {train_bal_acc:.2f}, Test Balanced Accuracy: {test_bal_acc:.2f}')

    plt.tight_layout()
    plt.show()

# Ex 3
def ex3():
    centers = [(-10, -10), (10, 10)]
    std_devs = [2, 6]
    X, _ = make_blobs(n_samples=[200, 100], centers=centers, cluster_std=std_devs, random_state=42)

    contamination = 0.07

    n_neighbors_values = [5, 10, 20]
    fig, axs = plt.subplots(len(n_neighbors_values), 2, figsize=(15, 15))

    for i, n_neighbors in enumerate(n_neighbors_values):
        knn = KNN(n_neighbors=n_neighbors, contamination=contamination)
        knn.fit(X)
        y_pred_knn = knn.labels_

        lof = LOF(n_neighbors=n_neighbors, contamination=contamination)
        lof.fit(X)
        y_pred_lof = lof.labels_

        axs[i, 0].scatter(X[y_pred_knn == 0][:, 0], X[y_pred_knn == 0][:, 1], c='blue', label='Inliers')
        axs[i, 0].scatter(X[y_pred_knn == 1][:, 0], X[y_pred_knn == 1][:, 1], c='red', label='Outliers')
        axs[i, 0].set_title(f'KNN (n_neighbors={n_neighbors})')
        axs[i, 0].legend()

        axs[i, 1].scatter(X[y_pred_lof == 0][:, 0], X[y_pred_lof == 0][:, 1], c='blue', label='Inliers')
        axs[i, 1].scatter(X[y_pred_lof == 1][:, 0], X[y_pred_lof == 1][:, 1], c='red', label='Outliers')
        axs[i, 1].set_title(f'LOF (n_neighbors={n_neighbors})')
        axs[i, 1].legend()

    plt.tight_layout()
    plt.show()

# Ex 4
def ex4():
    data = scipy.io.loadmat('cardio.mat')
    X = data['X']
    y = data['y'].ravel()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    X_train_norm, X_test_norm = standardizer(X_train, X_test)

    contamination = 0.1

    n_neighbors_values = range(30, 121, 10)
    train_scores = []
    test_scores = []

    for n_neighbors in n_neighbors_values:
        model = KNN(n_neighbors=n_neighbors, contamination=contamination)
        model.fit(X_train_norm)

        train_scores.append(model.decision_scores_)
        test_scores.append(model.decision_function(X_test_norm))

        y_train_pred = model.predict(X_train_norm)
        y_test_pred = model.predict(X_test_norm)
        train_bal_acc = balanced_accuracy(y_train, y_train_pred)
        test_bal_acc = balanced_accuracy(y_test, y_test_pred)

        print(f'n_neighbors={n_neighbors}, Train Balanced Accuracy: {train_bal_acc:.2f}, Test Balanced Accuracy: {test_bal_acc:.2f}')

    train_scores_norm, test_scores_norm = standardizer(np.array(train_scores).T, np.array(test_scores).T)

    avg_train_scores = average(train_scores_norm)
    avg_test_scores = average(test_scores_norm)

    max_train_scores = maximization(train_scores_norm)
    max_test_scores = maximization(test_scores_norm)

    threshold_avg = np.quantile(avg_train_scores, 1 - contamination)
    threshold_max = np.quantile(max_train_scores, 1 - contamination)

    y_train_pred_avg = (avg_train_scores > threshold_avg).astype(int)
    y_test_pred_avg = (avg_test_scores > threshold_avg).astype(int)
    y_train_pred_max = (max_train_scores > threshold_max).astype(int)
    y_test_pred_max = (max_test_scores > threshold_max).astype(int)

    train_bal_acc_avg = balanced_accuracy(y_train, y_train_pred_avg)
    test_bal_acc_avg = balanced_accuracy(y_test, y_test_pred_avg)
    train_bal_acc_max = balanced_accuracy(y_train, y_train_pred_max)
    test_bal_acc_max = balanced_accuracy(y_test, y_test_pred_max)

    print(
        f'Average Strategy - Train Balanced Accuracy: {train_bal_acc_avg:.2f}, Test Balanced Accuracy: {test_bal_acc_avg:.2f}')
    print(
        f'Maximization Strategy - Train Balanced Accuracy: {train_bal_acc_max:.2f}, Test Balanced Accuracy: {test_bal_acc_max:.2f}')

#ex1()
#ex1_2d()
#ex2()
#ex3()
ex4()