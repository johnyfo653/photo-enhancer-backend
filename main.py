from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import io
import os

app = FastAPI()

# CORS (aby fungoval Netlify)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def enhance_image(image):
    img = np.array(image)

    # Sharpen
    kernel = np.array([[0, -1, 0],
                       [-1, 5,-1],
                       [0, -1, 0]])
    sharp = cv2.filter2D(img, -1, kernel)

    # Contrast + color
    lab = cv2.cvtColor(sharp, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0)
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    final = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)

    return final

@app.get("/")
def home():
    return {"message": "Photo Enhancer API běží 🚀"}

@app.post("/enhance/")
async def enhance(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    enhanced = enhance_image(image)

    output_path = "output.jpg"
    Image.fromarray(enhanced).save(output_path)

    return FileResponse(output_path)

