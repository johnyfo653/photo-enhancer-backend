from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import replicate
import requests
import os
import uuid

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

token = os.getenv("REPLICATE_API_TOKEN")

if not token:
    raise Exception("❌ Missing REPLICATE_API_TOKEN")

client = replicate.Client(api_token=token)

@app.get("/")
def home():
    return {"message": "🚀 Photo Enhancer API běží"}

@app.post("/enhance/")
async def enhance(file: UploadFile = File(...)):
    try:
        # načtení obrázku
        image_bytes = await file.read()

        # STEP 1 — face enhance
        face_output = client.run(
            "sczhou/codeformer",
            input={
                "image": image_bytes,
                "face_upsample": True,
                "background_enhance": False,
                "codeformer_fidelity": 0.6
            }
        )

        face_data = requests.get(face_output).content

        # STEP 2 — upscale
        upscale_output = client.run(
            "nightmareai/real-esrgan",
            input={
                "image": face_data,
                "scale": 4
            }
        )

        final_data = requests.get(upscale_output).content

        filename = f"{uuid.uuid4()}.png"

        with open(filename, "wb") as f:
            f.write(final_data)

        return FileResponse(filename)

    except Exception as e:
        return {"error": str(e)}
