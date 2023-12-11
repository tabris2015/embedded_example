from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import cache


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    api_name: str = "Object Detection service"
    revision: str = "local"
    yolo_version: str = "yolov8n.pt"
    mediapipe_det_model: str = "efficientdet.tflite"
    log_level: str = "DEBUG"


@cache
def get_settings():
    print("getting settings...")
    return Settings()
