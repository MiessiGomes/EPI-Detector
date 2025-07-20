from config.settings import SETTINGS
from ultralytics import YOLO


def train_model():
    MODEL_PATH = "epi_monitor/model/yolo11m.pt"
    model = YOLO(MODEL_PATH)

    results = model.train(
        data=SETTINGS.DATA_PATH,
        epochs=SETTINGS.EPOCHS,
        imgsz=SETTINGS.IMG_SIZE,
        batch=SETTINGS.BATCH_SIZE,
        name="yolov11n_epi_custom",
    )


if __name__ == "__main__":
    train_model()
