#! /usr/bin/python

from sklearn import svm
from helpers import *
import joblib


DATA_DIR = 'data/'
TEST_DIR = 'test/'
DATASET = 'emnist'
TEST_DATA_FILENAME = DATA_DIR + DATASET + '/emnist-letters-test-images-idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + DATASET + '/emnist-letters-test-labels-idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + DATASET + '/emnist-letters-train-images-idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + DATASET + '/emnist-letters-train-labels-idx1-ubyte'


@timing
def ocr(X_train, y_train, X_test, kernel='linear'):
    model_name = f"svm-{len(X_train)}-{kernel}"
    import os
    if model_name in os.listdir():
        print("Model already found, using saved version")
        model = joblib.load(model_name)
        return model.predict(X_test)

    clf = svm.SVC(kernel=kernel).fit(X_train, y_train)
    print("Saving model", model_name)
    joblib.dump(clf, model_name)
    y_pred = clf.predict(X_test)

    return y_pred

def main():
    train_n = 60000
    test_n = 10000
    print("train_n =", train_n)
    print("test_n =", test_n)

    X_train = read_images(TRAIN_DATA_FILENAME, train_n)
    y_train = read_labels(TRAIN_LABELS_FILENAME, train_n)

    X_test = read_images(TEST_DATA_FILENAME, test_n)
    y_test = read_labels(TEST_LABELS_FILENAME, test_n)

    X_train = extract_features(X_train)
    X_test = extract_features(X_test)

    kernel = 'rbf' # ["linear", "rbf", "poly"]

    time1 = time.time()
    y_pred = ocr(X_train, y_train, X_test, kernel)
    time2 = time.time()

    accuracy = sum([int(y_pred_i) == int(y_test_i) for y_pred_i, y_test_i in zip(y_pred, y_test)])  / len(y_pred) * 100

    print("Predicted values =", [chr(i - 1 + 65) for i in y_pred])
    print("Actual values =", [chr(i - 1 + 65) for i in y_test])
    print(f"Accuracy = {accuracy}%")

    # Write analytics to  a text file
    with open("emnist-analytics.txt", "a") as f:
        f.writelines([
            f"train_n={train_n}, ",
            f"test_n={test_n}, ",
            f"accuracy={accuracy:.2f}%, ",
            f"kernel={kernel}, ",
            f"time={time2-time1:.3f} seconds"
            f"\n{'=' * 80}\n"
        ])

if __name__ == "__main__":
    main()