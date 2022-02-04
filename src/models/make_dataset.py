import os
import gzip
import numpy as np
from tqdm import main
import config as CONFIG


def load_data(data_content):

    """Load data from `path`"""
    if data_content == "train":
        labels_path = os.path.join(CONFIG.TRAIN_DATA_LABELS)
        images_path = os.path.join(CONFIG.TRAIN_DATA_IMGS)
        print("Loading Training Data...")
    elif data_content == "test":
        labels_path = os.path.join(CONFIG.TEST_DATA_LABELS)
        images_path = os.path.join(CONFIG.TEST_DATA_IMGS)
        print("Loading Testing Data...")

    with gzip.open(labels_path, "rb") as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8, offset=8)

    with gzip.open(images_path, "rb") as imgpath:
        images = np.frombuffer(
            imgpath.read(), dtype=np.uint8, offset=16
        ).reshape(len(labels), 784)

    print(type(images), type(labels))

    return images, labels


if __name__ == "__main__":
    train_img, train_labels = load_data(data_content="test")
