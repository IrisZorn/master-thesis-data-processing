from pathlib import Path

from ultralytics import YOLO

MODEL_PATH = "../../resources/trained_models/yolov8s_forest_folliesv142/weights/best.pt"
VIDEO_PATH = "../../resources/custom_test_img/forest_follies/forest_follies_aql.mp4"

CONF = 0.5
IOU = 0.5
IMGSZ = 640


def run_detection(source, model_path=MODEL_PATH, conf=CONF, iou=IOU, imgsz=IMGSZ,
                   project=None, name="track", exist_ok=False):
    """Run YOLOv8 tracking on a video, saving detections (boxes, labels, conf) to disk."""
    model = YOLO(str(model_path))

    results = model.track(
        source=str(source),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        persist=True,
        tracker="botsort.yaml",
        save=True,
        save_txt=True,
        save_conf=True,
        stream=True,
        project=str(project) if project is not None else None,
        name=name,
        exist_ok=exist_ok,
    )

    for _ in results:
        pass


if __name__ == "__main__":
    run_detection(VIDEO_PATH)
