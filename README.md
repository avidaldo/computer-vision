# Computer Vision Module

A hands-on module covering computer vision from image fundamentals to face authentication, built with OpenCV, YOLO, CLIP, and ChromaDB.

The module is organized into three areas. Each area can be studied independently, though the progression from one to the next is intentional.

---

## OpenCV: Images and Video

[opencv/opencv_fundamentals.ipynb](opencv/opencv_fundamentals.ipynb)
Loading, displaying, and manipulating images. Covers colour spaces (BGR, RGB, HSV, LAB), basic transformations (resize, rotate, crop, flip), drawing, and saving. Explains why `uint8` matters and when to prefer PNG over JPEG.

[opencv/opencv_image_processing.ipynb](opencv/opencv_image_processing.ipynb)
Classical image processing. Filtering and convolution (Gaussian, median, bilateral, custom kernels), edge detection (Sobel, Laplacian, Canny with hysteresis threshold tuning), thresholding (fixed, Otsu, adaptive), morphological operations, contour detection, and histogram equalization including CLAHE.

[opencv/opencv_video.ipynb](opencv/opencv_video.ipynb)
From files and webcams. Reading and writing video, per-frame processing, codec selection, background subtraction with MOG2, and dense optical flow with Farneback.

---

## YOLO: Object Detection

[yolo/object_detection_and_yolo.ipynb](yolo/object_detection_and_yolo.ipynb)
Concepts and practice in one notebook. Part one covers the core building blocks of object detection — bounding box formats, IoU, Non-Maximum Suppression, detection metrics (precision, recall, mAP50, mAP50-95), and how YOLO is structured (backbone / neck / head). Part two is hands-on: loading a pre-trained model, running inference on images and video, filtering detections, instance segmentation, model comparison by speed, and export for deployment.

[yolo/yolo_custom_training.ipynb](yolo/yolo_custom_training.ipynb)
Adapting YOLO to new data. Covers the YOLO annotation format, dataset validation, training configuration, and reading training plots. Includes a rigorous explanation of transfer learning — the spectrum from feature extraction (frozen backbone) to full fine-tuning — and the concept of catastrophic forgetting.

---

## Embeddings, Databases, and Recognition

[vectors_and_embeddings.ipynb](vectors_and_embeddings.ipynb)
From pixel arrays to semantic vectors. Vector arithmetic, Euclidean vs cosine distance, and how CLIP maps images (and text) to a shared 512-D space using a Vision Transformer backbone. Includes PCA visualisation and a bridge to NLP: word embeddings, contextual embeddings, and how the same vector-similarity idea underlies retrieval-augmented generation (RAG).

[chromadb_intro.ipynb](chromadb_intro.ipynb)
Vector databases as infrastructure. How ChromaDB stores embeddings and retrieves them by approximate nearest-neighbour search (HNSW). The cosine distance formula, metadata filtering, ephemeral vs persistent clients, and choosing the right distance metric for normalised CLIP embeddings.

[face_recognition_pipeline.ipynb](face_recognition_pipeline.ipynb)
Putting it all together. A five-step pipeline: capture → detect (YOLO) → embed (CLIP) → store (ChromaDB) → verify. Each step is explained and demonstrated with real images. The notebook also shows the webcam capture pattern used in a production authentication system, and includes a similarity heatmap to visualise the separation between enrolled identities.

---

## Suggested Project: Face Authentication with a Webcam

The notebooks above provide all the pieces needed to build a small face authentication system — the kind used for attendance checking or room access.

**Goal**: enroll a set of people by capturing their face from a webcam, then verify their identity in real time.

**Suggested steps**:

1. Use the webcam pattern in [09_face_recognition_pipeline.ipynb](09_face_recognition_pipeline.ipynb) to capture several frames per person at enrollment.
2. For each frame, detect and crop the face using the YOLO pipeline shown in the same notebook.
3. Embed each crop with CLIP and store in a [ChromaDB persistent collection](08_chromadb_intro.ipynb).
4. At verification, capture a new frame, embed it, and query the database. Apply the 0.85 similarity threshold as a starting point — adjust based on your hardware and lighting.
5. Add a quality check (blur detection, brightness bounds) before embedding to avoid storing bad captures; the webcam section of [09_face_recognition_pipeline.ipynb](09_face_recognition_pipeline.ipynb) shows this pattern.

**Extensions to consider**:
- Use `yolov8n-face.pt` instead of the general model for tighter face crops.
- Enroll multiple frames per person and compare the query against all of them (soft voting).
- Store enrollment data with timestamps; flag embeddings older than N days as stale.
- Add a logging step for rejected attempts.

The gap between this notebook-based prototype and a real deployment is primarily reliability engineering, not algorithmic complexity, which makes it a useful project scope for learning.

---

## Repository Structure

```
.
├── opencv/
│   ├── opencv_fundamentals.ipynb
│   ├── opencv_image_processing.ipynb
│   └── opencv_video.ipynb
├── yolo/
│   ├── object_detection_and_yolo.ipynb
│   └── yolo_custom_training.ipynb
├── 07_vectors_and_embeddings.ipynb
├── 08_chromadb_intro.ipynb
├── 09_face_recognition_pipeline.ipynb
├── img/                  # static test images committed to the repo
│   ├── lenna.png
│   ├── baboon.png
│   ├── peppers.jpg
│   ├── bus.jpg           # YOLO demo image
│   ├── scene1.jpg
│   ├── scene2.jpg
│   └── scene3.jpg
├── outputs/              # gitignored; generated by notebooks
└── models/               # gitignored; downloaded YOLO weights
```

## Environment

Python ≥ 3.13, managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync          # install all dependencies from pyproject.toml
```

Key dependencies: `ultralytics`, `opencv-python`, `sentence-transformers`, `chromadb`, `scikit-learn`, `matplotlib`.
