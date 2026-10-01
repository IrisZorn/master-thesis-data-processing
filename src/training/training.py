from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8s.pt")
    results = model.train(
        data='training_set/data.yaml',
        imgsz=640,
        epochs=350,
        batch=4,
        augment=True,
        close_mosaic=10,
        cache=True,
        name='yolov8s_wally_warblesv8',  # tuned for class confusion
        device=0)
