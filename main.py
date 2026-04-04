from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import replicate
import requests
import os
import uuid
from io import BytesIO

app = FastAPI()

# CORS (frontend support)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Token z Render
token = os.getenv("REPLICATE_API_TOKEN")

if not token:
    raise Exception("Missing REPLICATE_API_TOKEN")

client = replicate.Client(api_token=token)

# Test endpoint
@app.get("/")
def home():
    return {"message": "🚀 Photo Enhancer běží"}

# Hlavní endpoint
@app.post("/enhance/")
async def enhance(file: UploadFile = File(...)):
    try:
        # načtení obrázku
        image_bytes = await file.read()
        image_file = BytesIO(image_bytes)

        # 1️⃣ Face enhancement (CodeFormer)
        face_output = client.run(
            "sczhou/codeformer:latest",
            input={
                "image": image_file,
                "face_upsample": True,
                "background_enhance": False,
                "codeformer_fidelity": 0.6
            }
        )

        # stáhnout výstup
        face_data = requests.get(face_output, timeout=60).content
        face_file = BytesIO(face_data)

        # 2️⃣ Upscale (Real-ESRGAN)
        upscale_output = client.run(
            "nightmareai/real-esrgan:latest",
            input={
                "image": face_file,
                "scale": 4
            }
        )

        # stáhnout finální obrázek
        final_data = requests.get(upscale_output, timeout=60).content

        # uložit
        filename = f"{uuid.uuid4()}.png"
        with open(filename, "wb") as f:
            f.write(final_data)

        return FileResponse(filename)

    except Exception as e:
        return {"error": str(e)}
