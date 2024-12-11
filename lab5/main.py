import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score
from pyod.models.pca import PCA
from pyod.models.kpca import KPCA
from pyod.utils.data import generate_data
from pyod.utils.utility import standardizer
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras import layers, models


# Exercise 1

def ex1():
    # 1. + 2. Generate a 3D dataset and PCA
    mean = [5, 10, 2]
    cov = [[3, 2, 2], [2, 10, 1], [2, 1, 2]]
    data = np.random.multivariate_normal(mean, cov, 500)

    # Plot dataset in 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2])
    plt.show()

    # Center data
    data_centered = data - np.mean(data, axis=0)

    # Covariance matrix
    cov_matrix = np.cov(data_centered, rowvar=False)

    # EVD
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvalues and eigenvectors
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    # Transform data
    data_transformed = np.dot(data_centered, eigenvectors)

    # Normalize data
    data_normalized = data_transformed / np.sqrt(eigenvalues)

    # 3. Plot cumulative explained variance and individual variances
    cumulative_explained_variance = np.cumsum(eigenvalues) / np.sum(eigenvalues)

    plt.figure()
    plt.step(range(1, len(eigenvalues) + 1), cumulative_explained_variance, where='mid')
    plt.bar(range(1, len(eigenvalues) + 1), eigenvalues / np.sum(eigenvalues), alpha=0.5)
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.show()

    # 4. Identify outliers based on the 3rd principal component
    third_component = data_transformed[:, 2]
    threshold = np.quantile(third_component, 0.9)
    outliers = third_component > threshold

    # Plot dataset with anomalies
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=outliers, cmap='coolwarm')
    plt.show()

    # Identify outliers based on the 2nd principal component
    second_component = data_transformed[:, 1]
    threshold = np.quantile(second_component, 0.9)
    outliers = second_component > threshold

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=outliers, cmap='coolwarm')
    plt.show()

    # Identify outliers based on normalized distance
    distances = np.linalg.norm(data_normalized, axis=1)
    threshold = np.quantile(distances, 0.9)
    outliers = distances > threshold

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=outliers, cmap='coolwarm')
    plt.show()

# Exercise 2

def balanced_accuracy(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return (tpr + tnr) / 2

def ex2():
    # Load shuttle dataset
    data = loadmat('shuttle.mat')
    X = data['X']
    y = data['y'].ravel()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

    # Standardize data
    X_train, X_test = standardizer(X_train, X_test)

    # PCA model
    pca = PCA(contamination=0.1)
    pca.fit(X_train)

    # Plot cumulative explained variance and individual variances
    explained_variance = pca.explained_variance_ratio_
    cumulative_explained_variance = np.cumsum(explained_variance)

    plt.figure()
    plt.step(range(1, len(explained_variance) + 1), cumulative_explained_variance, where='mid')
    plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5)
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.show()

    # Compute balanced accuracy for train and test sets
    y_train_pred = pca.predict(X_train)
    y_test_pred = pca.predict(X_test)

    train_balanced_accuracy = balanced_accuracy(y_train, y_train_pred)
    test_balanced_accuracy = balanced_accuracy(y_test, y_test_pred)

    print(f'PCA Train Balanced Accuracy: {train_balanced_accuracy}')
    print(f'PCA Test Balanced Accuracy: {test_balanced_accuracy}')

    # KPCA model
    kpca = KPCA(contamination=0.1)
    kpca.fit(X_train)

    # Compute balanced accuracy for train and test sets
    y_train_pred_kpca = kpca.predict(X_train)
    y_test_pred_kpca = kpca.predict(X_test)

    train_balanced_accuracy_kpca = balanced_accuracy_score(y_train, y_train_pred_kpca)
    test_balanced_accuracy_kpca = balanced_accuracy_score(y_test, y_test_pred_kpca)

    print(f'KPCA Train Balanced Accuracy: {train_balanced_accuracy_kpca}')
    print(f'KPCA Test Balanced Accuracy: {test_balanced_accuracy_kpca}')

# Exercise 3
def ex3():
    data = loadmat('shuttle.mat')
    X = data['X']
    y = data['y'].ravel()

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

    # Min-max normalization
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Design the Autoencoder class
    class Autoencoder(tf.keras.Model):
        def __init__(self):
            super(Autoencoder, self).__init__()
            self.encoder = models.Sequential([
                layers.Dense(8, activation='relu'),
                layers.Dense(5, activation='relu'),
                layers.Dense(3, activation='relu')
            ])
            self.decoder = models.Sequential([
                layers.Dense(5, activation='relu'),
                layers.Dense(8, activation='relu'),
                layers.Dense(9, activation='sigmoid')
            ])

        def call(self, x):
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            return decoded

    # Compile and train model
    autoencoder = Autoencoder()
    autoencoder.compile(optimizer='adam', loss='mse')

    history = autoencoder.fit(X_train, X_train, epochs=100, batch_size=1024, validation_data=(X_test, X_test))

    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Get reconstruction errors
    reconstructions = autoencoder.predict(X_train)
    train_errors = np.mean(np.square(X_train - reconstructions), axis=1)

    # Compute threshold
    threshold = np.quantile(train_errors, 1 - 0.1)

    # Classify data based on threshold
    y_train_pred = train_errors > threshold
    y_test_pred = np.mean(np.square(X_test - autoencoder.predict(X_test)), axis=1) > threshold

    # Compute balanced accuracy
    train_balanced_accuracy = balanced_accuracy(y_train, y_train_pred)
    test_balanced_accuracy = balanced_accuracy(y_test, y_test_pred)

    print(f'Train Balanced Accuracy: {train_balanced_accuracy}')
    print(f'Test Balanced Accuracy: {test_balanced_accuracy}')

#ex1()
#ex2()
ex3()