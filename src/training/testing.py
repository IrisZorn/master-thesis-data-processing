from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('../../resources/trained_models/yolov8n_forest_folliesv132/weights/best.pt')
    results = model.val(
        data='training_set/data.yaml',
        split='test',
        imgsz=640,
        batch=8,
        device=0
    )
