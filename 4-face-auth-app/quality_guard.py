"""
Checks run before any identity decision: is the image usable, and is there a person in it?

Sourced from face_recognition_pipeline.ipynb: the quality checks come from
capture_faces_from_webcam() and the detection from detect_and_crop_face().
"""

from dataclasses import dataclass
from functools import cache
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from config import FaceAuthConfig

PERSON_CLASS_ID = 0  # "person" in the COCO classes YOLOv8 was trained on


@dataclass
class QualityReport:
    passed: bool
    reason: str
    brightness: float
    sharpness: float


def check_image_quality(image: np.ndarray, config: FaceAuthConfig) -> QualityReport:
    """Rejects images too dark or too blurred to give a reliable embedding."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    # Variance of the Laplacian: edges give large second derivatives, so a blurred image scores low.
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    if brightness < config.min_brightness:
        reason = f"too dark (brightness {brightness:.1f} < {config.min_brightness})"
        return QualityReport(False, reason, brightness, sharpness)
    if sharpness < config.min_sharpness:
        reason = f"too blurred (sharpness {sharpness:.1f} < {config.min_sharpness})"
        return QualityReport(False, reason, brightness, sharpness)
    return QualityReport(True, "acceptable", brightness, sharpness)


@cache
def load_detector(model_path: Path) -> YOLO:
    """Loaded once per path and reused. Downloads the weights on first use if the file is missing."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    return YOLO(model_path)


def detect_and_crop_person(image: np.ndarray, config: FaceAuthConfig, padding: int = 20) -> np.ndarray | None:
    """
    Returns the crop of the most confident person detection, or None if there is nobody.

    Change from the notebook: the notebook falls back to the whole image when nobody is detected.
    For access control that is a security hole (any picture would be compared against the
    enrolled faces), so here "nobody detected" means None, and the caller denies access.
    """
    results = load_detector(config.yolo_model_path)(image, verbose=False)

    best_box, best_confidence = None, 0.0
    for box in results[0].boxes:
        confidence = float(box.conf[0])
        if int(box.cls[0]) == PERSON_CLASS_ID and confidence > best_confidence:
            best_box, best_confidence = box.xyxy[0].cpu().numpy().astype(int), confidence

    if best_box is None:
        return None

    height, width = image.shape[:2]
    x1, y1, x2, y2 = best_box
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(width, x2 + padding), min(height, y2 + padding)
    return image[y1:y2, x1:x2]


def extract_person(image_path: Path, config: FaceAuthConfig) -> tuple[np.ndarray | None, str]:
    """
    The pipeline shared by enrolment and authentication: load → quality check → detect and crop.

    Returns (crop, "") on success, or (None, reason) at the first step that fails.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        return None, f"cannot read image {image_path}"

    quality = check_image_quality(image, config)
    print(f"  quality: brightness {quality.brightness:.1f}, sharpness {quality.sharpness:.1f} -> {quality.reason}")
    if not quality.passed:
        return None, f"image {quality.reason}"

    crop = detect_and_crop_person(image, config)
    if crop is None:
        return None, "no person detected"
    return crop, ""
