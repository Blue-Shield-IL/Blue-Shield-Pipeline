# Blue Shield - Processing Service

This repository contains the **Processing Service** for the Blue Shield project. It is a FastAPI-based Python backend responsible for the end-to-end data pipeline, including ingestion, filtering, and analysis.

---

## Tech Stack
* Framework: FastAPI
* Server: Uvicorn
* Language: Python 3.10+
* Environment Management: venv

---

## Getting Started

###  Navigate to Directory
```bash
cd processing_service

---

 ###Setup Virtual Environment

# Create the environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate

### Install Dependencies 
pip install -r requirements.txt

---

## Running the Server
uvicorn main:app --reload

---

## API Documentation
Once the server is running, you can access the interactive API documentation (Swagger UI):
* Swagger UI: http://127.0.0.1:8000/docs
* ReDoc: http://127.0.0.1:8000/redoc

---

## Project Structure
* main.py - Application entry point and route definitions.
* requirements.txt - Project dependencies.
* .gitignore - Prevents unnecessary files (like venv/) from being committed.
