import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from pathlib import Path

import cv2
import torch
torch.set_num_threads(1)

from langchain_core.tools import tool
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "shuttle.pt"

BASE_URL = "http://localhost:8000"

OUTPUT_DIR = BASE_DIR / "runs" / "shuttle"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# MODEL
# ============================================================

shuttle_model: YOLO | None = None


def get_model() -> YOLO:
    global shuttle_model

    if shuttle_model is None:

        print(
            "1 - loading YOLO model",
            flush=True,
        )

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"YOLO model not found: {MODEL_PATH}"
            )

        shuttle_model = YOLO(
            str(MODEL_PATH)
        )

        print(
            "2 - YOLO model loaded",
            flush=True,
        )

    return shuttle_model


# ============================================================
# IMAGE
# ============================================================

@tool
def predict_image(file_path: str) -> dict:
    """
    Detect badminton shuttles in an image.
    """

    print(
        "YOLO IMAGE TOOL STARTED",
        flush=True,
    )

    model = get_model()

    print(
        "3 - starting image prediction",
        flush=True,
    )

    torch.set_num_threads(1)

    results = model.predict(
        source=file_path,
        conf=0.25,
        device="cpu",
        workers=0,
        verbose=False,
    )

    print(
        "4 - image prediction completed",
        flush=True,
    )

    result = results[0]

    detections = len(result.boxes)

    print(
        f"5 - detections: {detections}",
        flush=True,
    )

    # --------------------------------------------------------
    # Save annotated image
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / f"{Path(file_path).stem}_result.jpg"
    )

    annotated = result.plot()

    cv2.imwrite(
        str(output_path),
        annotated,
    )

    print(
        f"6 - saved: {output_path}",
        flush=True,
    )

    return {
        "status": "success",
        "detections": detections,
        "file_path": str(output_path),
        "media_type": "image",
    }


# ============================================================
# VIDEO
# ============================================================

@tool
def predict_video(file_path: str) -> dict:
    """
    Detect and track badminton shuttles in a video.
    """

    print(
        "YOLO VIDEO TOOL STARTED",
        flush=True,
    )

    # --------------------------------------------------------
    # Validate input
    # --------------------------------------------------------

    input_path = Path(file_path)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Video not found: {input_path}"
        )

    print(
        f"Input video: {input_path}",
        flush=True,
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = get_model()

    print(
        "3 - opening video",
        flush=True,
    )

    # --------------------------------------------------------
    # Open input video
    # --------------------------------------------------------

    cap = cv2.VideoCapture(
        str(input_path)
    )

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {input_path}"
        )

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if fps <= 0:
        fps = 30.0

    print(
        f"Video: {width}x{height} @ {fps:.2f} FPS",
        flush=True,
    )

    print(
        f"Total frames: {total_frames}",
        flush=True,
    )

    # --------------------------------------------------------
    # Output path
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / f"{input_path.stem}_shuttle.mp4"
    )

    # --------------------------------------------------------
    # Video writer
    # --------------------------------------------------------

    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        cap.release()

        raise RuntimeError(
            f"Could not create output video: {output_path}"
        )

    print(
        "4 - video writer ready",
        flush=True,
    )

    # --------------------------------------------------------
    # Process frames
    # --------------------------------------------------------

    frame_number = 0

    try:

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            # ------------------------------------------------
            # YOLO tracking
            # ------------------------------------------------

            results = model.track(
                frame,
                conf=0.25,
                device="cpu",
                persist=True,
                verbose=False,
            )

            result = results[0]

            # ------------------------------------------------
            # Draw detections
            # ------------------------------------------------

            annotated_frame = result.plot()

            # ------------------------------------------------
            # Write frame
            # ------------------------------------------------

            writer.write(
                annotated_frame
            )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                frame_number == 1
                or frame_number % 10 == 0
            ):

                if total_frames > 0:

                    percentage = (
                        frame_number
                        / total_frames
                        * 100
                    )

                    print(
                        f"Processed "
                        f"{frame_number}/"
                        f"{total_frames} "
                        f"({percentage:.1f}%)",
                        flush=True,
                    )

                else:

                    print(
                        f"Processed "
                        f"{frame_number} frames",
                        flush=True,
                    )

    finally:

        cap.release()

        writer.release()

    print(
        "5 - video processing completed",
        flush=True,
    )

    print(
        f"Frames processed: {frame_number}",
        flush=True,
    )

    # --------------------------------------------------------
    # Validate output
    # --------------------------------------------------------

    if not output_path.exists():

        raise RuntimeError(
            "YOLO finished but output "
            "video was not created."
        )

    file_size = output_path.stat().st_size

    if file_size == 0:

        raise RuntimeError(
            "YOLO created an empty output video."
        )

    print(
        f"6 - output created: {output_path}",
        flush=True,
    )

    print(
        f"Output size: {file_size / 1024 / 1024:.2f} MB",
        flush=True,
    )

    # --------------------------------------------------------
    # URL
    # --------------------------------------------------------

    relative_path = output_path.relative_to(
        BASE_DIR
    )

    video_url = (
        f"{BASE_URL}/"
        + str(relative_path).replace("\\", "/")
    )

    print(
        f"7 - video URL: {video_url}",
        flush=True,
    )

    return {
        "status": "success",
        "video_url": video_url,
        "file_path": str(output_path),
        "media_type": "video/mp4",
        "frames_processed": frame_number,
    }