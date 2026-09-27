# Computer Vision Module

A hands-on module covering computer vision from image fundamentals to face authentication, built with OpenCV, YOLO, CLIP, and ChromaDB.

The module is a path in five stages. Each stage builds on the previous ones, and the folder numbers give the order.

## 1. OpenCV: Images and Video

- [1-opencv/opencv_fundamentals.ipynb](1-opencv/opencv_fundamentals.ipynb): loading and manipulating images; BGR vs RGB, `uint8`, resizing, rotation, cropping, drawing, and saving.
- [1-opencv/opencv_image_processing.ipynb](1-opencv/opencv_image_processing.ipynb): classical image processing; filtering, convolution, edge detection, thresholding, morphology, contours, and CLAHE.
- [1-opencv/opencv_video.ipynb](1-opencv/opencv_video.ipynb): video processing from files and webcams; frame loops, codecs, background subtraction, and optical flow.

## 2. YOLO: Object Detection

- [2-yolo/object_detection_and_yolo.ipynb](2-yolo/object_detection_and_yolo.ipynb): detection concepts and inference with YOLO26, the current Ultralytics model; bounding boxes, IoU, NMS, metrics, pre-trained inference, segmentation, speed comparison, and export.
- [2-yolo/yolo_custom_training.ipynb](2-yolo/yolo_custom_training.ipynb): fine-tuning YOLO; annotation format, dataset validation, training configuration, plots, transfer learning, and catastrophic forgetting.

## 3. Embeddings, Vector Databases, and Recognition

- [3-embeddings/vectors_and_embeddings.ipynb](3-embeddings/vectors_and_embeddings.ipynb): image embeddings; vector arithmetic, cosine vs Euclidean distance, CLIP, PCA, and links to NLP and RAG.
- [3-embeddings/chromadb_intro.ipynb](3-embeddings/chromadb_intro.ipynb): vector databases; HNSW search, cosine distance, metadata filtering, and ephemeral vs persistent clients.
- [3-embeddings/face_recognition_pipeline.ipynb](3-embeddings/face_recognition_pipeline.ipynb): end-to-end face verification; capture, detect, embed, store, verify, webcam workflow, and a similarity heatmap.

The same embedding + vector search technique works for text: [ai-chat-guardrails](https://github.com/avidaldo/ai-chat-guardrails) uses it to block messages that mean the same as known attacks.

## 4. From Notebook to Application

- [4-face-auth-app/](4-face-auth-app/): the stage 3 pipeline rebuilt as a terminal app: one module per responsibility, settings in `.env` with Pydantic Settings, and access denied at every failing step. Its README explains each design change and reports what the app really achieves on the sample images, including where it falls short.

## 5. Your Turn: Suggested Improvements

The [app's README](4-face-auth-app/README.md#suggested-improvements) lists the next steps, from calibrating the threshold with real genuine and impostor pairs, to face-specific detectors and embeddings, liveness detection, webcam enrolment with several frames per person, video input and an audit trail.

The gap between the notebook prototype and a real deployment is mostly reliability engineering, not new algorithms. That makes it a good project scope for learning.

## How to Reuse This in Your Own Projects

The path is also a recipe for your own computer vision applications:

1. **Prototype in a notebook**, one step per cell, looking at every intermediate image and number. That is what notebooks are for.
2. **Measure before trusting.** Every threshold (brightness, sharpness, similarity) should come from examples of both outcomes, and be checked on examples you did not use to choose it. The app's README shows how misleading a demo can be without this.
3. **Pick models trained for your task.** YOLO pretrained on COCO detects persons, not faces; CLIP describes images, it does not identify people. The pipeline structure transfers to any task; the models must fit it.
4. **Move to an application** when the steps work: one module per responsibility, every setting in `.env` read with Pydantic Settings, a decision for every way each step can fail, and tests for the logic that does not need a model ([4-face-auth-app](4-face-auth-app/)).

## Repository Structure

```text
.
├── 1-opencv/             # stage 1 notebooks
├── 2-yolo/               # stage 2 notebooks
├── 3-embeddings/         # stage 3 notebooks
├── 4-face-auth-app/      # stage 4: terminal application + tests
├── resources/
│   ├── images/           # static images used by the notebooks and the app
│   └── models/           # gitignored; downloaded YOLO weights
└── artifacts/            # gitignored; generated images, videos, YOLO runs, ChromaDB data
```

## Environment

Python ≥ 3.13, managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # install all dependencies from pyproject.toml
uv run pytest    # tests of the stage 4 app (no model download needed)
```

Key dependencies: `ultralytics`, `opencv-python`, `sentence-transformers`, `chromadb`, `pydantic-settings`, `scikit-learn`, `matplotlib`.
