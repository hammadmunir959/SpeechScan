from api.config import settings

# Configure Celery to use Redis as the broker and result backend
app = Celery(
    "speechscan",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["api.tasks"]
)

# Optional configuration
app.conf.update(
    task_track_started=True,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
)

if __name__ == "__main__":
    app.start()
