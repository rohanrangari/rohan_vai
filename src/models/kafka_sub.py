import json
from time import sleep
from kafka import KafkaConsumer, KafkaProducer
from predict_model import main


def consumer():
    consumer = KafkaConsumer("CnnnModel")
    for msg in consumer:
        print(msg)
        print("Predictions:", main())


if __name__ == "__main__":
    consumer()
