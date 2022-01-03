#! /usr/bin/python

from sklearn import svm
from helpers import *

DATA_DIR = 'data/'
TEST_DIR = 'test/'
DATASET = 'mnist'
TEST_DATA_FILENAME = DATA_DIR + DATASET + '/t10k-images-idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + DATASET + '/t10k-labels-idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + DATASET + '/train-images-idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + DATASET + '/train-labels-idx1-ubyte'


@timing
def ocr(X_train, y_train, X_test, kernel='linear'):
    clf = svm.SVC(kernel=kernel).fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return y_pred

def main():
    train_n = 15000
    test_n = 100
    print("train_n =", train_n)
    print("test_n =", test_n)
    X_train = read_images(TRAIN_DATA_FILENAME, train_n)
    y_train = read_labels(TRAIN_LABELS_FILENAME, train_n)
    X_test = read_images(TEST_DATA_FILENAME, test_n)
    y_test = read_labels(TEST_LABELS_FILENAME, test_n)

    X_train = extract_features(X_train)
    X_test = extract_features(X_test)

    time1 = time.time()
    y_pred = ocr(X_train, y_train, X_test)
    time2 = time.time()
    accuracy = sum([int(y_pred_i) == int(y_test_i) for y_pred_i, y_test_i in zip(y_pred, y_test)])  / len(y_pred)

    print("Predicted values =", y_pred)
    print("Actual values =", y_test)
    print(f"Accuracy = {accuracy * 100}%")

    # Write analytics to  a text file
    with open("analytics.txt", "a") as f:
        f.writelines([
            f"train_n={train_n}, ",
            f"test_n={test_n}, ",
            f"accuracy={accuracy}, ",
            f"time={time2-time1:.3f} seconds"
            f"\n{'=' * 60}\n"
        ])

if __name__ == "__main__":
    main()