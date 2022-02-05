import os
import config as config
from google.cloud import pubsub_v1
from predict_model import preprocess_img, predict


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.credentials_path


def publish_msg():
    publisher = pubsub_v1.PublisherClient()
    data = "Hello Pub Sub"
    data = data.encode("utf-8")
    attributes = {"model": "CnnModel", "input": "image"}
    future = publisher.publish(config.topic_path, data, **attributes)
    print(f"Published message id : {future.result()}")


if __name__ == "__main__":
    publish_msg()
