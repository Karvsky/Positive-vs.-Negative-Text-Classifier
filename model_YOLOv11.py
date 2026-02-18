from ultralytics import YOLO

def train_model():
    model = YOLO("yolo11n.pt")

    #result = model.train(data="./Dataset_yolov11/data.yaml", epochs=40, imgsz=640, device=0, batch=16)

def model():

    model = YOLO("./runs/detect/train5/weights/best.pt")

    #metrics = model.val(data="./Dataset_yolov11/data.yaml", split='test', device=0)

    #print(f"Accuracy on a completely new data: {metrics.box.map50:.3f}")

    return model

def val_data(model):
    metrics = model.val(data="./Dataset_yolov11/data.yaml", split='test', device=0)

    print(f"Accuracy on a completely new data: {metrics.box.map50:.3f}")

if __name__ == '__main__':
    val_data(model())