from tensorflow.keras.preprocessing import image
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import config as config


def health_check():
    print("Health check")
    return "Health check"


def preprocess_img():
    img = cv2.imread(config.test_img_path)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(config.path2, gray_image)
    my_img = image.load_img(config.path2, target_size=(28, 28))
    img = tf.io.read_file(config.path2)
    img = tf.image.decode_image(img)
    img = tf.image.resize(img, size=[28, 28])
    img = img / 255.0
    expanded_img = tf.expand_dims(img, axis=0)
    return expanded_img


def predict(expanded_img):
    model = load_model(config.model_path)
    test_predict = np.argmax(model.predict(expanded_img), axis=1)
    pred_class = config.class_names[int(test_predict)]
    return pred_class


def main():
    expanded_img = preprocess_img()
    pred_class = predict(expanded_img)
    # print(pred_class)
    return pred_class


if __name__ == "main":
    main()
