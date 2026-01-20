import os
import logging
from api.celery_app import app
from api.config import settings
from api.logging_config import setup_logging
from api.database import db
from api.storage import get_storage
# from predict import DysarthriaGenderPredictor # Moved to lazy load

# Initialize structured logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize persistence
db.initialize()
storage = get_storage()

# Global predictor instance (loaded only within the worker process)
_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        logger.info("🤖 Initializing DysarthriaGenderPredictor in worker...")
        try:
            from predict import DysarthriaGenderPredictor
            # Use settings for model directory
            _predictor = DysarthriaGenderPredictor(model_dir=settings.MODEL_DIR)
            logger.info("✅ Predictor initialized")
        except ImportError as e:
            logger.error(f"❌ Failed to import predictor: {e}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to initialize predictor: {e}")
            raise
    return _predictor

@app.task(bind=True, name="api.tasks.predict_task")
def predict_task(self, audio_path: str):
    """
    Background task for dysarthria and gender prediction
    """
    try:
        logger.info(f"🚀 Starting prediction task for: {audio_path}")
        predictor = get_predictor()
        
        # 1. Perform prediction
        result = predictor.predict(audio_path)
        
        # 2. Save result to persistence layer
        # Strip internal paths from result for security before saving
        db_result = result.copy()
        db_result.pop("audio_path", None)
        db.save_result(self.request.id, db_result)
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Prediction task failed: {e}")
        return {"error": str(e)}
    finally:
        # 3. Cleanup temporary file
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
                logger.info(f"🗑️ Cleaned up temp file: {audio_path}")
            except Exception as cleanup_err:
                logger.warning(f"⚠️ Failed to delete temp file {audio_path}: {cleanup_err}")

