from fastapi import FastAPI
from app.checking_complience.router import router as compliance_router
from app.Dtat_extrasion.router import router as extraction_router

app = FastAPI(title="Pilot Compliance AI System")

# Endpoint 1: Checking
app.include_router(compliance_router, prefix="/api/v1/compliance", tags=["Compliance Checking"])

# Endpoint 2: Embedding (Admin)
app.include_router(extraction_router, prefix="/api/v1/admin", tags=["Rules Ingestion"])

if __name__ == "__main__":
    import uvicorn
    print("Starting Pilot Compliance Server...")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)