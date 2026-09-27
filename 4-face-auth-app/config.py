"""
Settings for the face access-control app, read from `.env` with Pydantic Settings.

The same pattern as ai-chat-guardrails/chatbot/config.py: every value that was hard-coded in the
notebook becomes a field with a default, and any of them can be overridden in `.env` (FIELD_NAME
in upper case) without touching the code.
"""

from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

APP_DIR = Path(__file__).resolve().parent


class FaceAuthConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=APP_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    image_path: Path = Field(default=Path("../resources/images/scene1.jpg"))
    db_path: Path = Field(default=Path("../artifacts/face_auth_db"))
    collection_name: str = Field(default="enrolled_faces")
    min_brightness: float = Field(default=40.0)
    min_sharpness: float = Field(default=100.0)
    verification_threshold: float = Field(default=0.80)
    yolo_model_path: Path = Field(default=Path("../resources/models/yolo26n.pt"))
    clip_model_name: str = Field(default="clip-ViT-B-32")

    @field_validator("image_path", "db_path", "yolo_model_path")
    @classmethod
    def resolve_from_app_dir(cls, value: Path) -> Path:
        # Relative paths are anchored to this folder, not to wherever the command is run from.
        return value if value.is_absolute() else (APP_DIR / value).resolve()
