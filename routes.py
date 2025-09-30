from fastapi import APIRouter, UploadFile, File
from services import extract_from_pdf

router = APIRouter()

@router.post("/extract")
async def extract_pdf(file: UploadFile = File(...)):
    pdf_bytes = await file.read()
    extracted = extract_from_pdf(pdf_bytes)
    return {"filename": file.filename, "extracted": extracted}
