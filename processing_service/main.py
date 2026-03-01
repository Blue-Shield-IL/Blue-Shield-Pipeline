from fastapi import FastAPI

app = FastAPI(
    title="Blue Shield - Processing Service",
    description="Backend infrastructure for data ingestion and analysis",
    version="1.0.0"
)

# Root Endpoint
@app.get("/")
def read_root():
    return {
        "status": "Online",
        "message": "Welcome to the 'Blue Shield' Processing Service",
        "docs": "/docs"
    }

# Health Check
@app.get("/health")
def health_check():
    return {"status": "healthy"}