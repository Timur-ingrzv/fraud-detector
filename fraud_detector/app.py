import os
import sys
import pandas as pd
import time
import logging
import json
import joblib
from pathlib import Path
from confluent_kafka import Consumer, Producer
from sklearn.preprocessing import TargetEncoder

sys.path.append(os.path.abspath('./src'))
from data_preprocessing import preprocess_data
from scoring import get_preds

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/logs/service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

KAFKA_CONFIG = {
    "transactions_topic": os.getenv("KAFKA_TRANSACTIONS_TOPIC", "transactions"),
    "scoring_topic": os.getenv("KAFKA_SCORING_TOPIC", "scoring"),
    "bootstrap_servers": os.getenv("BOOTSTRAP_SERVERS", "kafka:9092"),
    "transactions_group": os.getenv("KAFKA_TRANSACTIONS_GROUP", 'ml-scorer')
}

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))

class ModelUpdater:
    def __init__(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        latest = MODELS_DIR / "latest.txt"
        if not latest.exists() or not latest.read_text(encoding="utf-8").strip():
            dest = self._model_path("initial")
            if not dest.exists():
                raise FileNotFoundError(
                    "Initial model doesn't exist"
                )
            latest.write_text("initial\n", encoding="utf-8")

        self.current_version = self._read_latest_version()
        self.model = self._load_model(self.current_version)
        
    def _model_path(self, version: str) -> Path:
        return MODELS_DIR / f"my_classifier_{version}.pkl"

    def _read_latest_version(self) -> str:
        version = (MODELS_DIR / 'latest.txt').read_text(encoding='utf-8').strip()
        if not version:
            raise RuntimeError('No model pointer in latest.txt')
        return version

    def _load_model(self, version):
        path = self._model_path(version)
        model = joblib.load(path)
        return model

    def _current_model(self):
        version = self._read_latest_version()
        if version != self.current_version:
            self.model = self._load_model(version)
            self.current_version = version
        return self.model, self.current_version

class EncoderUpdater:
    def __init__(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        latest = MODELS_DIR / "latest.txt"
        if not latest.exists() or not latest.read_text(encoding="utf-8").strip():
            dest = self._model_path("initial")
            if not dest.exists():
                raise FileNotFoundError(
                    "Initial model doesn't exist"
                )
            latest.write_text("initial\n", encoding="utf-8")

        self.current_version = self._read_latest_version()
        self.model = self._load_model(self.current_version)
        
    def _model_path(self, version: str) -> Path:
        return MODELS_DIR / f"target_encoder_{version}.pkl"

    def _read_latest_version(self) -> str:
        version = (MODELS_DIR / 'latest.txt').read_text(encoding='utf-8').strip()
        if not version:
            raise RuntimeError('No encoder pointer in latest_encoder.txt')
        return version

    def _load_model(self, version):
        path = self._model_path(version)
        model = joblib.load(path)
        return model

    def _current_model(self):
        version = self._read_latest_version()
        if version != self.current_version:
            self.model = self._load_model(version)
            self.current_version = version
        return self.model, self.current_version
        

class ProcessingService:
    def __init__(self):
        self.consumer = Consumer({
            'bootstrap.servers': KAFKA_CONFIG['bootstrap_servers'],
            'group.id': KAFKA_CONFIG['transactions_group'],
            'auto.offset.reset': 'earliest'
        })
        self.producer = Producer({
            'bootstrap.servers': KAFKA_CONFIG['bootstrap_servers']
        })
        self.consumer.subscribe([KAFKA_CONFIG['transactions_topic']])

        self.model_updater = ModelUpdater()
        self.encoder_updater = EncoderUpdater()

    def process_messages(self):
        while True:
            message = self.consumer.poll(1.0)
            if not message:
                time.sleep(0.2)
                continue
            if message.error():
                logger.error(f"Kafka error: {message.error()}")
                continue
            try:
                data = json.loads(message.value().decode('utf-8'))
                transaction_id = data['transaction_id']
                df = pd.DataFrame(data['data'])

                df = preprocess_data(df, mode='inference', encoder_updater=self.encoder_updater)
                prediction = get_preds(df, self.model_updater)
                prediction['transaction_id'] = transaction_id
                logger.info(f"Prediction for transaction {transaction_id} completed")

                record = prediction.to_json(orient='records')
                self.producer.produce(
                    KAFKA_CONFIG['scoring_topic'],
                    value=record.encode('utf-8')
                )

            except Exception as e:
                logger.error(f"Error processing message: {e}")

    def close(self):
        self.consumer.close()
        self.producer.flush(timeout=10)

if __name__ == "__main__":
    logger.info('Starting Kafka ML scoring service...')
    service = ProcessingService()
    try:
        service.process_messages()
    except KeyboardInterrupt:
        logger.info('Service stopped by user')
    finally:
        service.close()

