import os
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from tensorflow.keras.preprocessing import image
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, Flatten
from tensorflow.keras.models import load_model

# User-defined
import config as config
from make_dataset import load_data

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
model_path = r"C:\Rohan\Vector AI\repo\rohan_vai\src\models\trained_models\fmnist_model.h5"


def load_data():
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    return x_train, y_train, x_test, y_test


def get_model(img_width, img_height, no_of_dim):
    """[summary]"""
    model = Sequential()
    model.add(
        Conv2D(
            filters=32,
            kernel_size=(4, 4),
            activation="relu",
            input_shape=(img_width, img_height, no_of_dim),
        )
    )
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(10, activation="softmax"))

    model.compile(
        optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def draw_plots(model):
    metrics = pd.DataFrame(model.history.history)
    metrics[["loss", "val_loss"]].plot()
    plt.savefig(
        r"C:\Rohan\Vector AI\repo\rohan_vai\src\plots\loss-plot.png",
        dpi=300,
        bbox_inches="tight",
    )
    metrics[["accuracy", "val_accuracy"]].plot()
    plt.savefig(
        r"C:\Rohan\Vector AI\repo\rohan_vai\src\plots\acc-plot.png",
        dpi=300,
        bbox_inches="tight",
    )


def model_performance(model, x_test, y_test):
    predictions = np.argmax(model.predict(x_test), axis=1)
    report = classification_report(y_test, predictions, output_dict=True)
    df = pd.DataFrame(report).transpose()
    df.to_csv("classification_report.csv")


def main():
    # Step 1: Load datasets
    x_train, y_train, x_test, y_test = load_data()

    training_samples, img_width, img_height = x_train.shape
    testing_samples, _, _ = x_test.shape
    no_of_dim = 1
    num_classes = len(config.class_names)
    print(
        training_samples, img_width, img_height, testing_samples, num_classes
    )
    # Step 2: Normalize
    x_train = x_train / 255
    x_test = x_test / 255

    # Step 3: reshape
    x_train = x_train.reshape(
        training_samples, img_width, img_height, no_of_dim
    )
    x_test = x_test.reshape(testing_samples, img_width, img_height, no_of_dim)

    y_cat_train = to_categorical(y_train, num_classes)
    y_cat_test = to_categorical(y_test, num_classes)

    model = get_model(img_width, img_height, no_of_dim)

    model.fit(
        x_train,
        y_cat_train,
        validation_data=(x_test, y_cat_test),
        epochs=3,
        verbose=0,
    )

    model.save(model_path)
    draw_plots(model)
    model_performance(model, x_test, y_test)


if __name__ == "__main":
    main()
