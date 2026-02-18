import easyocr
import cv2
from ultralytics import YOLO
from model_YOLOv11 import model

def text():

    frame = model()

    reader = easyocr.Reader(['pl', 'en'], gpu=True)

    path = "./Test_data/test9.png"

    photo = frame(path)

    img = cv2.imread(path)

    ### ZMIANA START: Tworzymy pustą listę, by zbierać wszystkie teksty ###
    all_texts = []
    ### ZMIANA KONIEC ###

    for result in photo:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            crop = img[y1:y2, x1:x2]

            text_data = reader.readtext(crop, detail=0)
            final_text = " ".join(text_data)

            ### ZMIANA START: Dodajemy tekst do listy zamiast kończyć funkcję ###
            if final_text:  # Sprawdzamy, czy tekst nie jest pusty
                all_texts.append(final_text)
            ### ZMIANA KONIEC ###

    ### ZMIANA START: Zwracamy całą listę po sprawdzeniu wszystkich ramek ###
    return all_texts
    ### ZMIANA KONIEC ###