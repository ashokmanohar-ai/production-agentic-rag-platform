import asyncio

from app.main import get_durable_knowledge_service


async def run() -> None:
    service = get_durable_knowledge_service()
    while True:
        job = await service.process_next()
        await asyncio.sleep(0.2 if job else 2.0)


if __name__ == "__main__":
    asyncio.run(run())
