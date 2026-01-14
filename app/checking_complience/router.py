from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from typing import Optional, List
from .llm_service import analyze_compliance
from .checking_request import ComplianceReport
from config import settings
import os
import shutil

router = APIRouter()

@router.post("/check-compliance", response_model=ComplianceReport)
async def check_compliance(
    query: Optional[str] = Form(None),
    files: List[UploadFile] = File(None) # <--- UPDATED: Accepts a List of files
):
    """
    Checks compliance for:
    1. Text Query Only
    2. Uploaded Schedule (Single or Multiple Images/PDFs)
    3. Text Query + Uploaded Schedule
    """
    temp_files_data = []
    
    # Validation: Must have at least one input
    if not query and not files:
        raise HTTPException(status_code=400, detail="Must provide either a text query, a file, or both.")

    try:
        # 1. Process Multiple Files
        if files:
            # Optional: Limit to 5 files to prevent overload
            if len(files) > 5:
                raise HTTPException(status_code=400, detail="Maximum 5 files allowed per request.")

            for idx, file in enumerate(files):
                # Determine extension and save path
                file_ext = file.filename.split('.')[-1]
                temp_path = os.path.join(settings.TEMP_DATA_DIR, f"upload_{idx}.{file_ext}")
                
                # Save to disk
                with open(temp_path, "wb") as buffer:
                    shutil.copyfileobj(file.file, buffer)
                
                # Store path AND mime_type (Critical for PDF vs Image logic)
                temp_files_data.append({
                    "path": temp_path,
                    "mime_type": file.content_type
                })

        # 2. Run Analysis (Pass query + List of files)
        result = await analyze_compliance(query_text=query, uploaded_files=temp_files_data)
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 3. Cleanup Temp Files
        for f_data in temp_files_data:
            if os.path.exists(f_data["path"]):
                os.remove(f_data["path"])