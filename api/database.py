import logging
import firebase_admin
from firebase_admin import credentials, firestore
from api.config import settings

logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self._db = None
        self._initialized = False

    def initialize(self):
        """
        Initialize the Firebase Admin SDK.
        Requires GOOGLE_APPLICATION_CREDENTIALS environment variable or a local service account key.
        """
        if self._initialized:
            return

        try:
            # Check for service account key path in env
            cred_path = getattr(settings, "FIREBASE_CREDENTIALS_PATH", None)
            
            if cred_path and os.path.exists(cred_path):
                cred = credentials.Certificate(cred_path)
                firebase_admin.initialize_app(cred)
            else:
                # Attempt to initialize with default credentials (likely to work in GCP/Firebase environments)
                try:
                    firebase_admin.initialize_app()
                except ValueError:
                    # Already initialized or failed
                    pass

            self._db = firestore.client()
            self._initialized = True
            logger.info("✅ Firestore initialized")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize Firestore: {e}. Results will only be logged.")

    def save_result(self, job_id: str, result: dict):
        """
        Save prediction results to Firestore.
        """
        if not self._initialized or not self._db:
            logger.info(f"📝 Result for job {job_id} (No DB): {result}")
            return

        try:
            doc_ref = self._db.collection("analysis_history").document(job_id)
            doc_ref.set({
                "job_id": job_id,
                "timestamp": firestore.SERVER_TIMESTAMP,
                "result": result,
                "app_version": settings.APP_VERSION
            })
            logger.info(f"✅ Result saved to Firestore: {job_id}")
        except Exception as e:
            logger.error(f"❌ Error saving result to Firestore: {e}")

# Global database instance
db = Database()
