import uvicorn
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
import tempfile
import shutil
from ..inference import inference
from pathlib import Path
from io import BytesIO
import zipfile

app = FastAPI()

@app.post("/inference")
async def upload(model: str, file: UploadFile = File(...)):
    '''Endpoint to perform inference on an uploaded image using a specified model.'''

    # Save the uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp_path = tmp.name

        with file.file as upload:
            shutil.copyfileobj(upload, tmp)

    # Determine the path to the model file
    base_dir = Path(__file__).parent.parent.parent.resolve()
    print(base_dir)
    model_path = base_dir / "models" / f"{model}.pth"

    # Perform inference
    img, mask = inference(tmp_path, model_path)

    # Convert images to zip
    def image_to_bytes(image):
        b = BytesIO()
        image.save(b, format="PNG")
        return b.getvalue()

    # Create a zip file in memory containing the image and mask
    buffer = BytesIO()
    with zipfile.ZipFile(buffer, "w") as z:
        z.writestr("image.png", image_to_bytes(img))
        z.writestr("mask.png", image_to_bytes(mask))
    buffer.seek(0)

    return StreamingResponse(
        buffer,
        media_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=response.zip"}
    )

def run(host: str, port: int, reload: bool):
    '''Run the FastAPI application using Uvicorn.'''
    uvicorn.run("background_remover.api.main:app", host=host, port=port, reload=reload)