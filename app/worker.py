import asyncio

from app.config import get_settings
from app.main import get_durable_knowledge_service


async def run() -> None:
    settings = get_settings()
    service = get_durable_knowledge_service(settings)
    service.repository.recover_stale_jobs(settings.worker_job_lease_seconds)
    while True:
        job = await service.process_next()
        await asyncio.sleep(
            settings.worker_busy_poll_seconds if job else settings.worker_poll_seconds
        )


if __name__ == "__main__":
    asyncio.run(run())
