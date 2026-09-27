# Face Access Control App

The pipeline from [`../3-embeddings/face_recognition_pipeline.ipynb`](../3-embeddings/face_recognition_pipeline.ipynb), rebuilt as a terminal application: one module per responsibility, settings in a `.env` file, and an explicit decision for every way the pipeline can fail.

A notebook is the right tool to *explore* an idea: each step is visible and can be rerun on its own. It is the wrong tool to *run* one: state lives in global variables, values are hard-coded, and a failed step just prints a message. This folder shows that transition.

## Run It

From the repository root, once `uv sync` has been run:

```bash
cd 4-face-auth-app
uv run python enroll.py                                    # phase 1: store Alice, Bob and Charlie
uv run python face_auth.py                                 # phase 2: check IMAGE_PATH (scene1.jpg)
uv run python face_auth.py ../resources/images/face2.jpg   # someone who is not enrolled
uv run python face_auth.py ../resources/images/peppers.jpg # no person at all
```

YOLO weights (~6 MB) and CLIP (~600 MB) download on the first run. To change a setting, copy `.env.example` to `.env` and edit it; relative paths are resolved from this folder.

Tests (no model needed): `uv run pytest` from the repository root.

## Structure

| File | Responsibility | From the notebook |
|---|---|---|
| [`config.py`](config.py) | Settings with Pydantic Settings; same pattern as [ai-chat-guardrails](https://github.com/avidaldo/ai-chat-guardrails/blob/main/chatbot/config.py) | the hard-coded values |
| [`quality_guard.py`](quality_guard.py) | Brightness and sharpness checks; YOLO person detection and crop | `capture_faces_from_webcam()`, `detect_and_crop_face()` |
| [`verification.py`](verification.py) | CLIP embedding; nearest enrolled user in ChromaDB; threshold decision | `embed_face()`, `verify_identity()` |
| [`enroll.py`](enroll.py) | Phase 1: store one embedding per known user | step 4 |
| [`face_auth.py`](face_auth.py) | Phase 2: grant or deny access to one image | steps 1–5 in order |

## What Changed from the Notebook, and Why

- **Nobody detected → access denied.** The notebook falls back to the whole image, which is convenient for a demo. In access control it is a security hole: any picture at all would be compared with the enrolled faces.
- **Every failure is a denial.** An unreadable image, poor quality, no person, an empty database or a low similarity all end in `ACCESS DENIED` with the reason. An access-control system must *fail closed*.
- **Models load once.** `load_detector()` and `load_clip()` are wrapped in `functools.cache`: the first call loads the model and later calls reuse it. The notebook kept them in global variables.
- **Results are dataclasses.** `QualityReport` and `VerificationResult` name their fields, instead of returning a tuple whose positions you have to remember.
- **Paths don't depend on where you run the command.** They are resolved from this folder, and the images and weights are shared with the notebooks in `../resources/`.

## What It Really Shows: Measured Results

With the default settings (threshold 0.80, minimum sharpness 100):

| Image | Who | Result |
|---|---|---|
| `scene1.jpg` | Alice, enrolled **from this very image** | granted, similarity 1.000 |
| `face2.jpg` | not enrolled | denied, closest Charlie at 0.704 |
| `peppers.jpg` | no person | denied, no person detected |
| `face1.jpg`, `face3.jpg` | not enrolled | denied as **"too blurred"** (sharpness 95 and 86) |

With the sharpness check relaxed to 50, the strangers score between 0.64 and 0.75: all below 0.80, but only by about 0.05.

Read those numbers critically:

- **The "granted" case proves nothing.** Enrolling and verifying with the same image always gives 1.000. The real test is a *different* photo of an enrolled person, and these images don't include one.
- **The sharpness check rejected sharp photos.** `face1` and `face3` are crisp studio portraits on plain backgrounds. The variance of the Laplacian counts *edges*, and a plain background has almost none. A fixed threshold that suits busy scenes rejects clean portraits. That is a false rejection caused by the metric, not by the camera.
- **The margin against strangers is thin.** A threshold 0.05 lower would have let `face1` in as Alice.

## Suggested Improvements

These close the gap between this prototype and something you could trust with a door. Most are small projects on their own.

### Measure before you trust

- **Enrol and verify with different photos.** Take several photos of yourself and your classmates. Enrol with some and verify with others. Only then does a "granted" mean anything.
- **Calibrate the threshold.** Collect *genuine* pairs (same person, different photo) and *impostor* pairs (different people). Plot both similarity distributions and choose the threshold from the trade-off between the false accept rate (FAR: strangers let in) and the false reject rate (FRR: users locked out). The same exercise for text is `calibrate_semantic_guard.py` in ai-chat-guardrails.
- **Make the sharpness check relative.** Compare against the enrolment photo, or compute the sharpness only inside the detected crop, instead of using one absolute number for every scene.

### Better models

- **Detect faces, not bodies.** YOLO class 0 is *person*: the crop includes clothes and background, and CLIP embeds all of it. A face-specific detector (`yolov8n-face.pt`, or OpenCV's DNN face detector) gives tight face crops.
- **Use a face-recognition embedding.** CLIP was trained to match images with captions, so it encodes *what the picture shows* (a man in a grey jumper) more than *who* it is. Models trained for identity, such as ArcFace (via `insightface`) or FaceNet (`facenet-pytorch`), separate people far better.
- **Liveness detection.** Today a printed photo of Alice gets in as Alice. Blink detection, small head movements or a depth camera address this *presentation attack*.

### A real enrolment and use flow

- **Webcam enrolment with several frames per person.** Capture 5–10 frames with the webcam loop from the notebook, keep only those that pass the quality checks, and store all of them. At verification, compare against all of them and vote.
- **Video input.** Process a video file or a live stream: skip frames for speed, and grant access only after several consecutive matches.
- **Audit trail.** Save a JSON record of every attempt (time, image, closest match, similarity, decision) and the face crop, so rejected attempts can be reviewed later.
- **Show the result.** Display the query crop next to the closest enrolled face and the similarity, for example with matplotlib or a small Gradio app.
- **Expire old enrolments.** Store the enrolment date (already in the metadata) and ask users to re-enrol after N days, since faces change.
