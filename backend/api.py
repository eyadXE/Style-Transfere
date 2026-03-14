# backend/api.py
from fastapi import FastAPI, UploadFile
from PIL import Image
import io
import base64
import traceback
from backend.style_transfer import run_style_transfer
from torchvision.transforms import ToPILImage

app = FastAPI()

@app.post("/style-transfer")
async def style_transfer(content: UploadFile, style: UploadFile):
    try:
        # Read images from request
        content_img = Image.open(io.BytesIO(await content.read())).convert("RGB")
        style_img   = Image.open(io.BytesIO(await style.read())).convert("RGB")

        # Run NST
        result_tensor = run_style_transfer(content_img, style_img)

        # Convert to PIL
        result_img = ToPILImage()(result_tensor.cpu())

        # Encode as base64
        buffer = io.BytesIO()
        result_img.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode()

        return {"image": img_base64}

    except Exception as e:
        traceback.print_exc()
        return {"error": str(e)}