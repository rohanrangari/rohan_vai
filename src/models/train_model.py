import os
from matplotlib import path
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


def load_data():
    """Loading the dataset"""
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    return x_train, y_train, x_test, y_test


def get_model(img_width, img_height, no_of_dim):
    """
    Defining the CNN architecture & return it as model object
    """
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
    """
    Uses the trained/saved model to draw the plots for Loss & Accuracy metrics

    """
    metrics = pd.DataFrame(model.history.history)
    metrics[["loss", "val_loss"]].plot()
    plt.savefig(
        model.loss_plot,
        dpi=300,
        bbox_inches="tight",
    )
    metrics[["accuracy", "val_accuracy"]].plot()
    plt.savefig(
        model.acc_plot,
        dpi=300,
        bbox_inches="tight",
    )


def model_performance(model, x_test, y_test):
    """

    Args:
        model : trained/saved model
        x_test : Test data/Images
        y_test : Test labels
    """
    predictions = np.argmax(model.predict(x_test), axis=1)
    report = classification_report(y_test, predictions, output_dict=True)
    df = pd.DataFrame(report).transpose()
    print("classification_report:")
    print(df)
    df.to_csv(path=config.classification_report, index=False)


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

    # Step 4: Define & BUild the model
    model = get_model(img_width, img_height, no_of_dim)

    # Step 5: Training Stage
    model.fit(
        x_train,
        y_cat_train,
        validation_data=(x_test, y_cat_test),
        epochs=3,
        verbose=0,
    )

    # Step 6: Saving the trained model
    model.save(config.model_path)
    # Step 7: Visualkization on the trained model
    draw_plots(model)
    # Step 8: model performance metrics on the trained model
    model_performance(model, x_test, y_test)


if __name__ == "__main":
    main()
