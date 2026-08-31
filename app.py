from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
import base64
from typing import Optional

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    


    # background = cv2.GaussianBlur(gray_img, (0, 0), 50) 

    # normalized = cv2.divide(gray_img, background, scale=255) # (gray_img / background) * 255

    # thresh = cv2.adaptiveThreshold(
    #     gray_img,
    #     255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     cv2.THRESH_BINARY,
    #     115,
    #     15
    # )

    # thresh = 255-thresh

    gray_img = 255 - gray_img
    low_val, _ = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    high_val = min(254, low_val * 1.5)
    _, thresh = cv2.threshold(gray_img, high_val, 255, cv2.THRESH_BINARY)


    #_, thresholded_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresholded_img = thresh.copy()

    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # for contour in contours:
    #     area = cv2.contourArea(contour)
    #     if area < 20:
    #         cv2.drawContours(thresholded_img, [contour], -1, 0, thickness=cv2.FILLED)

    

    height, width = thresholded_img.shape
    biggest = max(width, height)

    blank_canvas = np.full((biggest, biggest), 0, dtype=np.uint8)

    if width > height:
        y_offset = (width - height) // 2
        blank_canvas[y_offset:y_offset + height, 0:width] = thresholded_img
    else:
        x_offset = (height - width) // 2
        blank_canvas[0:height, x_offset:x_offset + width] = thresholded_img
    
    blank_canvas = cv2.resize(blank_canvas, (280, 280), interpolation=cv2.INTER_AREA)

    x, y, w, h = cv2.boundingRect(thresholded_img)
    digit_crop = thresholded_img[y:y+h, x:x+w]

    if w > h:
        w = h
        digit_crop = cv2.resize(digit_crop, (int(w), int(h)))

    if h == 0:
        return {"prediction": "-1", "confidence": "-1", "preprocessed-img": "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAcABwDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigAooooAKKKKACiiigD//2Q=="}

    scalar = 200/h
    new_h = h * scalar
    new_w = w * scalar
    new_x = (280-new_w) // 2
    new_y = (280-new_h) // 2

    digit_crop = cv2.resize(digit_crop, (int(new_w), int(new_h)), interpolation=cv2.INTER_AREA)
    blank = np.full((280, 280), 0, dtype=np.uint8)
    blank[int(new_y) : int(new_y) + int(new_h), int(new_x) : int(new_x) + int(new_w)] = digit_crop
    resized_img = cv2.resize(blank, (28, 28), interpolation=cv2.INTER_AREA)
    _, resized_img = cv2.threshold(resized_img, 0, 255, cv2.THRESH_BINARY)
    normalized_img = resized_img / 255.0

    tensor = normalized_img.reshape(1, 28, 28, 1)

    response_img = resized_img
    response_img = response_img.astype("uint8")
    _, encoded_img = cv2.imencode(".jpg", response_img)
    base64str = base64.b64encode(encoded_img.tobytes()).decode("utf-8")

    predictions = model.predict(tensor)[0]
    idx = np.where(predictions > 0.5)
    if idx[0].size == 0:
        return {"prediction": "-1", "confidence": "-1", "preprocessed-img": str(base64str)}
    prediction = idx[0][0]
    confidence = predictions[prediction]

    return {"prediction": str(prediction), "confidence": str(confidence), "preprocessed-img": str(base64str)}

@app.post("/read-img")
def predict(img):
    pass

@app.get("/items/{item_id}")
def read_tem(item_id: int, q: Optional[str]):
    return {"item_id": item_id, "q" : q}

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port = 8000)