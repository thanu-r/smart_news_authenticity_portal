import os
import logging
from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("smartnews.database")

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017")
MONGODB_DB_NAME = os.getenv("MONGODB_DB_NAME", "smartnews_db")

class MongoDBManager:
    client: Optional[AsyncIOMotorClient] = None
    db: Optional[AsyncIOMotorDatabase] = None

    @classmethod
    async def connect(cls):
        """Establish asynchronous connection to MongoDB and initialize indexes."""
        try:
            logger.info(f"Connecting to MongoDB at {MONGODB_URI}...")
            cls.client = AsyncIOMotorClient(
                MONGODB_URI,
                serverSelectionTimeoutMS=4000,
                connectTimeoutMS=4000
            )
            cls.db = cls.client[MONGODB_DB_NAME]
            # Ping database to verify connection
            await cls.client.admin.command("ping")
            logger.info(f"Successfully connected to MongoDB database '{MONGODB_DB_NAME}'")
            await cls.init_indexes()
        except Exception as e:
            logger.warning(f"MongoDB connection failed: {e}. Running in degraded offline mode if needed.")

    @classmethod
    async def init_indexes(cls):
        """Ensure necessary indexes exist on collections."""
        if cls.db is None:
            return
        try:
            # Users indexes
            await cls.db.users.create_index("email", unique=True)
            await cls.db.users.create_index("username", unique=True)
            
            # Verifications indexes
            await cls.db.verifications.create_index("user_id")
            await cls.db.verifications.create_index("created_at")
            await cls.db.verifications.create_index("prediction")
            
            # Live news indexes
            await cls.db.live_news.create_index("url", unique=True, sparse=True)
            await cls.db.live_news.create_index("published_at")
            
            # Evidence & Feedback indexes
            await cls.db.evidence.create_index("verification_id")
            await cls.db.feedback.create_index("verification_id")
            await cls.db.model_versions.create_index("version_id", unique=True)
            await cls.db.admin_logs.create_index("timestamp")
            logger.info("MongoDB indexes verified.")
        except Exception as e:
            logger.error(f"Error creating MongoDB indexes: {e}")

    @classmethod
    async def disconnect(cls):
        """Close MongoDB connection gracefully."""
        if cls.client:
            cls.client.close()
            logger.info("MongoDB connection closed.")

def get_db() -> AsyncIOMotorDatabase:
    """Dependency helper to get the database."""
    if MongoDBManager.db is None:
        raise RuntimeError("MongoDB is not initialized. Ensure connect() has run.")
    return MongoDBManager.db
