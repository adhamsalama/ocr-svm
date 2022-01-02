#! /usr/bin/python

from sklearn import svm
import numpy as np

DEBUG = True

if DEBUG:
    # This code only exists to help us visually inspect the images.
    # It's in an `if DEBUG:` block to illustrate that we don't need it for our code to work.
    from PIL import Image
    import numpy as np

    def read_image(path):
        return np.asarray(Image.open(path).convert('L'))

    def write_image(image, path):
        img = Image.fromarray(np.array(image), 'L')
        img.save(path)

def timing(f):
    import time
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print(f'{f.__name__} function took {(time2-time1):.3f} s')

        return ret
    return wrap


DATA_DIR = 'data/'
TEST_DIR = 'test/'
DATASET = 'mnist'
TEST_DATA_FILENAME = DATA_DIR + DATASET + '/t10k-images-idx3-ubyte'
TEST_LABELS_FILENAME = DATA_DIR + DATASET + '/t10k-labels-idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + DATASET + '/train-images-idx3-ubyte'
TRAIN_LABELS_FILENAME = DATA_DIR + DATASET + '/train-labels-idx1-ubyte'

def bytes_to_int(byte_data):
    return int.from_bytes(byte_data, 'big')
@timing
def read_images(filename, n_max_images=None):
    images = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic number
        n_images = bytes_to_int(f.read(4))
        if n_max_images:
            n_images = n_max_images
        n_rows = bytes_to_int(f.read(4))
        n_columns = bytes_to_int(f.read(4))
        for image_idx in range(n_images):
            image = []
            for row_idx in range(n_rows):
                row = []
                for col_idx in range(n_columns):
                    pixel = bytes_to_int(f.read(1))
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return np.array(images)

@timing
def read_labels(filename, n_max_labels=None):
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # magic number
        n_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            n_labels = n_max_labels
        for label_idx in range(n_labels):
            label = bytes_to_int(f.read(1))
            labels.append(label)
    return np.array(labels)

def flatten_list(l):
    return np.array([pixel for sublist in l for pixel in sublist])
@timing
def extract_features(X):
    return np.array([flatten_list(sample) for sample in X])

@timing
def ocr(X_train, y_train, X_test, kernel='linear'):
    clf = svm.SVC(kernel=kernel).fit(X_train, y_train)
    # clf = svm.LinearSVC().fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    return y_pred



def main():
    train_n = 40000
    test_n = 4000
    print("train_n =", train_n)
    print("test_n =", test_n)
    X_train = read_images(TRAIN_DATA_FILENAME, train_n)
    y_train = read_labels(TRAIN_LABELS_FILENAME, train_n)
    X_test = read_images(TEST_DATA_FILENAME, test_n)
    y_test = read_labels(TEST_LABELS_FILENAME, test_n)

    X_train = extract_features(X_train)
    X_test = extract_features(X_test)

    y_pred = ocr(X_train, y_train, X_test)
    accuracy = sum([int(y_pred_i) == int(y_test_i) for y_pred_i, y_test_i in zip(y_pred, y_test)])

    print("Predicted values =", y_pred)
    print("Actual values =", y_test)
    print(f"Accuracy = {accuracy / len(y_pred) * 100}%")

if __name__ == "__main__":
    main()