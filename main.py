from fastapi import FastAPI
from routes import router

app = FastAPI(title="PDF Extraction Service with Whisperer")

app.include_router(router, prefix="/api")
