from kafka import KafkaProducer
from kafka import KafkaProducer


def publish_msg():

    producer = KafkaProducer(bootstrap_servers="localhost:1234")
    for _ in range(10):
        producer.send("CnnModel", b"Hello Pub Sub From Kafka ")


if __name__ == "__main__":
    publish_msg()
