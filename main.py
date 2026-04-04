from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import replicate

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
    print("❌ CHYBÍ REPLICATE_API_TOKEN")

client = replicate.Client(api_token=token) if token else None


@app.get("/")
def home():
    return {"message": "Photo Enhancer API běží 🚀"}


@app.post("/enhance/")
async def enhance(file: UploadFile = File(...)):
    try:
        if not client:
            return JSONResponse({"error": "Missing API token"}, status_code=500)

        image_bytes = await file.read()

        # STEP 1 — FACE AI
        face_output = client.run(
            "sczhou/codeformer",
            input={
                "image": image_bytes,
                "face_upsample": True,
                "background_enhance": False,
                "codeformer_fidelity": 0.7
            }
        )

        face_data = requests.get(face_output).content

        # STEP 2 — UPSCALE
        upscale_output = client.run(
            "nightmareai/real-esrgan",
            input={
                "image": face_data,
                "scale": 4
            }
        )

        final_data = requests.get(upscale_output).content

        with open("final.png", "wb") as f:
            f.write(final_data)

        return FileResponse("final.png")

    except Exception as e:
        print("ERROR:", str(e))
        return JSONResponse({"error": str(e)}, status_code=500)
