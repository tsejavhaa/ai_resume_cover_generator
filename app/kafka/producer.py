"""
Kafka producer.

Publishes a JobMessage to the resume-jobs topic.
The FastAPI endpoint calls publish_job() and returns immediately
with a job_id — no waiting for LLM inference.
"""
import base64
import json
from loguru import logger
from aiokafka import AIOKafkaProducer

from app.core.config import get_settings
from app.models.schemas import JobMessage


async def publish_job(message: JobMessage) -> None:
    """Serialize and publish a job to Kafka. Fire-and-forget from the API layer."""
    settings = get_settings()
    producer = AIOKafkaProducer(
        bootstrap_servers=settings.kafka_bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        # Use job_id as partition key so same user's jobs go to same partition
        key_serializer=lambda k: k.encode("utf-8"),
    )
    await producer.start()
    try:
        payload = message.model_dump()
        await producer.send_and_wait(
            topic=settings.kafka_topic,
            key=message.job_id,
            value=payload,
        )
        logger.info(f"Published job {message.job_id} to topic '{settings.kafka_topic}'")
    finally:
        await producer.stop()


def encode_file(file_bytes: bytes) -> str:
    """Base64-encode file bytes for safe JSON serialization."""
    return base64.b64encode(file_bytes).decode("utf-8")


def decode_file(b64: str) -> bytes:
    """Decode base64 file bytes from Kafka message."""
    return base64.b64decode(b64.encode("utf-8"))