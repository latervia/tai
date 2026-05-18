from app.shared.logger import logger


async def startup_event():
    logger.info("Starting up...")


async def shutdown_event():
    logger.info("Shutting down...")
