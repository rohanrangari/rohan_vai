import os
from google.cloud import pubsub_v1
from concurrent.futures import TimeoutError
from predict_model import preprocess_img, predict, health_check, main
import config as config

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = config.credentials_path
timeout = 7.0


def callback(message):
    """
    message contains information & acknowledges on pub-sub

    """
    print(f"Received message: {message}")
    print(f"data: {message.data}")
    if message.attributes:
        print("attributes:")
        for key in message.attributes:
            value = message.attributes.get(key)
            print(f"{key}:{value}")
    print("Predictions:", main())
    message.ack()


def pull_message():
    """
    Pulls messages from published messages
    """
    subscriber = pubsub_v1.SubscriberClient()
    streaming_pull_future = subscriber.subscribe(
        config.subscriber_path, callback=callback
    )
    print(f"Listening for messages on {config.subscriber_path}")

    with subscriber:
        try:
            streaming_pull_future.result(timeout)
        except TimeoutError:
            streaming_pull_future.cancel()
            streaming_pull_future.result()


if __name__ == "__main__":
    pull_message()
