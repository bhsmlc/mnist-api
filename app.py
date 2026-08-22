from fastapi import FastAPI, HTTPException, UploadFile, File
import os, uvicorn
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers

app = FastAPI()
model = models.load_model("mnistmodel.keras")

@app.get("/")
def read_root():
    return {}

@app.post("/predict-frame")
def predict_frame(image: UploadFile = File(...)):
    file_bytes = image.file.read()
    nparr = np.frombuffer(file_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = 255 - gray_img
    resized_img = cv2.resize(gray_img, (280, 280), interpolation=cv2.INTER_AREA)
    _, thresholded_img = cv2.threshold(resized_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    x, y, w, h = cv2.boundingRect(thresholded_img)
    digit_crop = thresholded_img[y:y+h, x:x+w]

    if w > h:
        w = h
        digit_crop = cv2.resize(digit_crop, (int(w), int(h)))

    if h == 0:
        return {"prediction": "-1", "confidence": "-1"}

    scalar = 200/h
    new_h = h * scalar
    new_w = w * scalar
    new_x = (280-new_w) // 2
    new_y = (280-new_h) // 2

    digit_crop = cv2.resize(digit_crop, (int(new_w), int(new_h)), interpolation=cv2.INTER_AREA)
    blank = np.full(78400, 0)
    blank = blank.reshape(280, 280)
    blank[int(new_y) : int(new_y) + int(new_h), int(new_x) : int(new_x) + int(new_w)] = digit_crop
    resized_img = cv2.resize(blank, (28, 28), interpolation=cv2.INTER_AREA)
    normalized_img = resized_img / 255.0
    tensor = normalized_img.reshape(1, 28, 28, 1)

    predictions = model.predict(tensor)[0]
    idx = np.where(predictions > 0.5)
    if idx[0].size == 0:
        return {"prediction": "-1", "confidence": "-1"}
    prediction = idx[0][0]
    confidence = predictions[prediction]

    return {"prediction": str(prediction), "confidence": str(confidence)}

@app.post("/read-img")
def predict(img):
    pass

@app.get("/items/{item_id}")
def read_tem(item_id: int, q: str | None = None):
    return {"item_id": item_id, "q" : q}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port = 8000)