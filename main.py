from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import replicate
import shutil
import requests
import os
import replicate

client = replicate.Client(api_token=os.environ["REPLICATE_API_TOKEN"])
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"message": "🔥 AI enhancer běží"}

@app.post("/enhance/")
async def enhance(file: UploadFile = File(...)):
    input_path = "input.jpg"
    step1 = "face.png"
    step2 = "final.png"

    # save input
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🧠 STEP 1 — FACE AI
    face_output = replicate.run(
        "sczhou/codeformer",
        input={
            "image": open(input_path, "rb"),
            "face_upsample": True,
            "background_enhance": False,
            "codeformer_fidelity": 0.6
        }
    )

    img_data = requests.get(face_output).content
    with open(step1, "wb") as f:
        f.write(img_data)

    # 🔍 STEP 2 — UPSCALE
    upscale_output = replicate.run(
        "nightmareai/real-esrgan",
        input={
            "image": open(step1, "rb"),
            "scale": 4
        }
    )

    img_data2 = requests.get(upscale_output).content
    with open(step2, "wb") as f:
        f.write(img_data2)

    return FileResponse(step2)

