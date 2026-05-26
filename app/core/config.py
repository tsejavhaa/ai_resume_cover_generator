from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    # LLM backend selection
    llm_backend: str = Field(default="ollama", pattern="^(ollama|deepseek)$")

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"

    # DeepSeek
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-chat"

    # NLP
    ner_model: str = "dslim/bert-base-NER"
    device: str = "cpu"

    # App
    app_env: str = "development"
    log_level: str = "INFO"

    # ── Phase 2: Kafka ───────────────────────────────────────
    kafka_bootstrap_servers: str = "kafka:9092"
    kafka_topic: str = "resume-jobs"
    kafka_consumer_group: str = "resume-workers"
    kafka_num_workers: int = 3

    # ── Phase 2: Redis ───────────────────────────────────────
    redis_url: str = "redis://redis:6379"
    redis_ttl_seconds: int = 3600  # results expire after 1 hour

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    return Settings()