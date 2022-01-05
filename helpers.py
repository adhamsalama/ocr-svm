import time
import numpy as np

def timing(f):
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        print(f'{f.__name__} function took {(time2-time1):.3f} s')

        return ret
    return wrap


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
                    pixel = f.read(1)
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
    return np.array([bytes_to_int(pixel) for sublist in l for pixel in sublist])

DEBUG = True

if DEBUG:
    # This code only exists to help us visually inspect the images.
    # It's in an `if DEBUG:` block to illustrate that we don't need it for our code to work.
    from PIL import Image

    def read_image(path):
        return np.asarray(Image.open(path).convert('L'))

    def write_image(image, path):
        img = Image.fromarray(np.array(image), 'L')
        img.save(path)

@timing
def extract_features(X):
    return np.array([flatten_list(sample) for sample in X])


