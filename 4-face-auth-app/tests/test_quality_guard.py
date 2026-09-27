"""Quality checks on synthetic images: no model or image file needed."""

import numpy as np

from config import APP_DIR, FaceAuthConfig
from quality_guard import check_image_quality


def make_config() -> FaceAuthConfig:
    # _env_file=None: tests must not depend on the developer's own .env
    return FaceAuthConfig(_env_file=None)


def checkerboard(square_size: int = 8, size: int = 128) -> np.ndarray:
    """Sharp black-and-white squares: bright enough and full of edges."""
    rows, columns = np.indices((size, size)) // square_size
    board = ((rows + columns) % 2 * 255).astype(np.uint8)
    return np.stack([board] * 3, axis=-1)


def test_dark_image_is_rejected():
    report = check_image_quality(np.zeros((128, 128, 3), dtype=np.uint8), make_config())

    assert not report.passed
    assert "too dark" in report.reason


def test_flat_image_is_rejected_as_blurred():
    # A uniform grey image has no edges at all, so the Laplacian variance is 0
    report = check_image_quality(np.full((128, 128, 3), 128, dtype=np.uint8), make_config())

    assert not report.passed
    assert "too blurred" in report.reason
    assert report.sharpness == 0.0


def test_sharp_bright_image_passes():
    report = check_image_quality(checkerboard(), make_config())

    assert report.passed
    assert report.sharpness > make_config().min_sharpness


def test_relative_paths_are_resolved_from_the_app_folder():
    config = make_config()

    assert config.yolo_model_path == (APP_DIR / "../resources/models/yolo26n.pt").resolve()
    assert config.image_path.is_file()
