"""
Kafka producer.

Publishes a JobMessage to the resume-jobs topic.
The FastAPI endpoint calls publish_job() and returns immediately
with a job_id — no waiting for LLM inference.
"""
import asyncio
import base64
import json
from loguru import logger
from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaConnectionError

from app.core.config import get_settings
from app.models.schemas import JobMessage


async def publish_job(message: JobMessage) -> None:
    """Serialize and publish a job to Kafka. Fire-and-forget from the API layer."""
    settings = get_settings()

    # Retry with exponential backoff to give Kafka time to become ready.
    # Kafka healthcheck has a 60s start_period + 15 retries at 15s intervals,
    # so it can take up to ~4 minutes — retry for at least 5 minutes.
    last_error = None
    for attempt in range(50):
        producer = AIOKafkaProducer(
            bootstrap_servers=settings.kafka_bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            key_serializer=lambda k: k.encode("utf-8"),
        )
        try:
            await producer.start()
            payload = message.model_dump()
            await producer.send_and_wait(
                topic=settings.kafka_topic,
                key=message.job_id,
                value=payload,
            )
            logger.info(f"Published job {message.job_id} to topic '{settings.kafka_topic}'")
            return
        except KafkaConnectionError as e:
            last_error = e
            wait = min(2 ** attempt, 15)
            logger.warning(f"Kafka not ready (attempt {attempt+1}/50), retrying in {wait}s: {e}")
            await asyncio.sleep(wait)
        finally:
            await producer.stop()

    # All retries exhausted — log and re-raise so the caller sees a 503
    logger.error(f"Failed to publish job {message.job_id} after 50 attempts")
    raise last_error or RuntimeError("Kafka unavailable")


def encode_file(file_bytes: bytes) -> str:
    """Base64-encode file bytes for safe JSON serialization."""
    return base64.b64encode(file_bytes).decode("utf-8")


def decode_file(b64: str) -> bytes:
    """Decode base64 file bytes from Kafka message."""
    return base64.b64decode(b64.encode("utf-8"))