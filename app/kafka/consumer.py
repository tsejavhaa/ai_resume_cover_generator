"""
Kafka consumer worker pool.

Spawns KAFKA_NUM_WORKERS concurrent async workers.
Each worker:
  1. Pulls a JobMessage from the resume-jobs topic
  2. Updates Redis → processing
  3. Calls generator.generate() (the same Phase 1 pipeline)
  4. Updates Redis → done / failed
  5. Notifies connected WebSocket clients

Run standalone:
    python -m app.kafka.consumer
"""
import asyncio
import json
from loguru import logger
from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaConnectionError

from app.core.config import get_settings
from app.models.schemas import JobMessage, GenerateRequest, ToneEnum, BackendEnum
from app.services.generator import generate
from app.services.storage.redis_store import set_processing, set_done, set_failed
from app.kafka.producer import decode_file
from app.kafka.websocket_manager import ws_manager


async def _process_message(raw_value: dict) -> None:
    """Handle a single Kafka message end-to-end."""
    try:
        msg = JobMessage(**raw_value)
    except Exception as e:
        logger.error(f"Invalid Kafka message: {e} | raw={raw_value}")
        return

    job_id = msg.job_id
    logger.info(f"Worker picked up job: {job_id}")

    # Mark as processing in Redis + notify WebSocket clients
    await set_processing(job_id)
    await ws_manager.broadcast(job_id, {"status": "processing", "job_id": job_id})

    try:
        # Reconstruct request and decode file bytes
        file_bytes = decode_file(msg.resume_bytes_b64)
        request = GenerateRequest(
            job_description=msg.job_description,
            tone=ToneEnum(msg.tone),
            backend=BackendEnum(msg.backend) if msg.backend else None,
            cover_letter_length=msg.cover_letter_length,
        )

        # Run the full Phase 1 generation pipeline
        result = await generate(request, file_bytes, msg.filename)

        # Persist result + notify
        await set_done(job_id, result)
        await ws_manager.broadcast(job_id, {
            "status": "done",
            "job_id": job_id,
            "result": result.model_dump(),
        })
        logger.info(f"Job completed: {job_id}")

    except Exception as e:
        logger.exception(f"Job {job_id} failed: {e}")
        await set_failed(job_id, str(e))
        await ws_manager.broadcast(job_id, {
            "status": "failed",
            "job_id": job_id,
            "error": str(e),
        })


async def run_worker(worker_id: int, consumer: AIOKafkaConsumer) -> None:
    """A single consumer worker — runs forever pulling from Kafka."""
    logger.info(f"Worker-{worker_id} started")
    async for msg in consumer:
        try:
            raw = json.loads(msg.value.decode("utf-8"))
            logger.debug(f"Worker-{worker_id} received message: {raw.get('job_id')}")
            await _process_message(raw)
        except Exception as e:
            logger.error(f"Worker-{worker_id} error: {e}")


async def start_consumer_pool() -> None:
    """
    Start N concurrent worker coroutines sharing one Kafka consumer group.
    Called from app startup — runs as a background task.
    """
    settings = get_settings()
    logger.info(
        f"Starting Kafka consumer pool | "
        f"brokers={settings.kafka_bootstrap_servers} "
        f"topic={settings.kafka_topic} "
        f"group={settings.kafka_consumer_group} "
        f"workers={settings.kafka_num_workers}"
    )

    # Retry loop — Kafka may not be ready immediately at startup
    for attempt in range(10):
        try:
            consumer = AIOKafkaConsumer(
                settings.kafka_topic,
                bootstrap_servers=settings.kafka_bootstrap_servers,
                group_id=settings.kafka_consumer_group,
                auto_offset_reset="earliest",
                enable_auto_commit=True,
                value_deserializer=lambda v: v,  # raw bytes; decoded in worker
            )
            await consumer.start()
            logger.info("Kafka consumer connected")
            break
        except KafkaConnectionError as e:
            wait = 2 ** attempt
            logger.warning(f"Kafka not ready (attempt {attempt+1}/10), retrying in {wait}s: {e}")
            await asyncio.sleep(wait)
    else:
        logger.error("Could not connect to Kafka after 10 attempts. Consumer pool not started.")
        return

    try:
        # Spawn N workers all sharing the same consumer
        workers = [
            asyncio.create_task(run_worker(i, consumer))
            for i in range(settings.kafka_num_workers)
        ]
        await asyncio.gather(*workers)
    finally:
        await consumer.stop()
        logger.info("Kafka consumer stopped")


# ── Standalone entry point ─────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(start_consumer_pool())