import numpy as np
import matplotlib.pyplot as plt
from pyod.utils.data import generate_data
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix, roc_curve, auc


def ex1ex2():
    # Exercise 1

    # generating the data
    X_train, X_test, y_train, y_test = generate_data(n_train=400, n_test=100, n_features=2, contamination=0.1)

    # plotting the training samples
    plt.scatter(X_train[y_train == 0][:, 0], X_train[y_train == 0][:, 1], label='Normal', c='blue')
    plt.scatter(X_train[y_train == 1][:, 0], X_train[y_train == 1][:, 1], label='Outliers', c='red')
    plt.legend()
    plt.title('Training Samples')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()


    # Exercise 2

    # fitting with the training data
    model = KNN(contamination=0.1)
    model.fit(X_train)

    print(X_train.shape)
    print(X_test.shape)

    # getting predictions for both training and testing data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    print(y_train.shape)
    print(y_train_pred.shape)

    print(y_test.shape)
    print(y_test_pred.shape)

    # computing the confusion matrix for training data
    cm_train = confusion_matrix(y_train, y_train_pred)
    tn_train, fp_train, fn_train, tp_train = cm_train.ravel()

    # computing the confusion matrix for testing data
    cm_test = confusion_matrix(y_test, y_test_pred)
    tn_test, fp_test, fn_test, tp_test = cm_test.ravel()

    # computing the balanced accuracy for training and testing data
    tpr_train = tp_train / (tp_train + fn_train)
    tnr_train = tn_train / (tn_train + fp_train)
    tpr_test = tp_test / (tp_test + fn_test)
    tnr_test = tn_test / (tn_test + fp_test)

    # calculating the balanced accuracy
    balanced_accuracy_train = (tpr_train + tnr_train) / 2
    balanced_accuracy_test = (tpr_test + tnr_test) / 2

    print(f"Training Data: TN={tn_train}, FP={fp_train}, FN={fn_train}, TP={tp_train}, Balanced Accuracy={balanced_accuracy_train:.2f}")
    print(f"Testing Data: TN={tn_test}, FP={fp_test}, FN={fn_test}, TP={tp_test}, Balanced Accuracy={balanced_accuracy_test:.2f}")

    # ROC curve
    y_test_scores = model.decision_function(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_test_scores)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.2f}')
    plt.legend()
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()


def ex3():
    contamination_rate = 0.1

    X_train, X_test, y_train, y_test = generate_data(n_train=1000, n_test=0, n_features=1, contamination=0.1)

    # computing Z-scores
    z_scores = (X_train - np.mean(X_train)) / np.std(X_train)

    # computing the Z-score threshold
    threshold = np.quantile(np.abs(z_scores), 1 - contamination_rate)

    # detecting anomalies based on the Z-score threshold
    y_train_pred = (np.abs(z_scores) > threshold).astype(int)

    # computing confusion matrix
    cm_train = confusion_matrix(y_train, y_train_pred)
    tn_train, fp_train, fn_train, tp_train = cm_train.ravel()

    # computing balanced accuracy
    tpr_train = tp_train / (tp_train + fn_train)
    tnr_train = tn_train / (tn_train + fp_train)
    balanced_accuracy_train = (tpr_train + tnr_train) / 2

    print(f"Training Data: TN={tn_train}, FP={fp_train}, FN={fn_train}, TP={tp_train}, Balanced Accuracy={balanced_accuracy_train:.2f}")


def ex4():
    contamination_rate = 0.1



ex1ex2()
ex3()
