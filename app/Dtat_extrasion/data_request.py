from pydantic import BaseModel

class ExtractionResponse(BaseModel):
    status: str
    message: str
    chunks_processed: int