import os
import shutil
import tempfile
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from model import model_manager

app = FastAPI(
    title="TruthLens AI Service",
    description="Deepfake and Synthetic Media Detection Microservice",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "ai-detection-engine",
        "version": "1.0.0",
        "device": str(model_manager.device),
        "model_loaded": model_manager.is_model_loaded,
        "model_metadata": model_manager.model_metadata
    }

@app.post("/detect/image")
async def detect_image(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"Unsupported format '{ext}'.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp_path = tmp.name
        shutil.copyfileobj(file.file, tmp)

    try:
        result = model_manager.predict_image(tmp_path)
        return {
            "success": True,
            "filename": file.filename,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "raw_scores": result["raw_scores"],
            "processing_time": result["processing_time"]
        }
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
