from fastapi import APIRouter, UploadFile, File, HTTPException
from .ectraction_service import process_and_embed_rules
from .data_request import ExtractionResponse
from config import settings
import os
import shutil

router = APIRouter()

@router.post("/embed-rules", response_model=ExtractionResponse)
async def upload_and_embed_rules(file: UploadFile = File(...)):
    """
    Uploads the Pilot Rulebook PDF.
    Deletes the old database and creates a new one split by 'Section' (#).
    """
    temp_path = os.path.join(settings.TEMP_DATA_DIR, file.filename)
    
    try:
        # Save uploaded file temporarily
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # Process
        chunks = await process_and_embed_rules(temp_path)
        
        # Clean up
        os.remove(temp_path)
        
        return {
            "status": "success", 
            "message": "Rules successfully embedded and saved locally.",
            "chunks_processed": chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))