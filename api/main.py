#!/usr/bin/env python3
"""
SpeechScan API - FastAPI server for dysarthria detection
Provides REST API and WebSocket endpoints for real-time audio analysis
"""

import os
import json
import time
import asyncio
import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from celery.result import AsyncResult

from api.celery_app import app as celery_app
from api.tasks import predict_task
from api.config import settings
from api.logging_config import setup_logging

# Initialize structured logging
setup_logging()
logger = logging.getLogger(__name__)

# Global loading state
loading_progress = {
    "loaded": False,
    "progress": 0,
    "status": "Initializing...",
    "details": "",
    "step_number": 0,
    "total_steps": 8,
    "estimated_time_remaining": "Unknown"
}
websocket_connections = set()

class LoadingStatusResponse(BaseModel):
    loaded: bool
    progress: int
    status: str
    details: str
    step_number: int
    total_steps: int
    estimated_time_remaining: str

class PredictionResponse(BaseModel):
    status: str
    clarityScore: str
    speechRate: str
    acousticNotes: str
    disclaimer: str = settings.MEDICAL_DISCLAIMER

# Note: Predictor loading moved to Celery worker in Phase 2

async def load_models_async():
    """
    NOTE: In the new async architecture, models are loaded by the Celery worker,
    not the API process. The API process only handles metadata and job submission.
    This function is kept for showing status but won't hold a 'predictor'.
    """
    global loading_progress
    loading_progress.update({
        "loaded": True,
        "progress": 100,
        "status": "API Ready",
        "details": "ML inference moved to background workers",
        "step_number": 1,
        "total_steps": 1
    })
    await broadcast_loading_update()

async def broadcast_loading_update():
    """Broadcast loading update to all connected WebSocket clients"""
    global websocket_connections
    if websocket_connections:
        message = {
            "type": "progress_update",
            **loading_progress
        }
        disconnected = set()
        for websocket in websocket_connections:
            try:
                await websocket.send_text(json.dumps(message))
            except:
                disconnected.add(websocket)
        
        # Remove disconnected clients
        websocket_connections -= disconnected

from api.database import db

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("🚀 Starting SpeechScan API...")
    
    # Initialize persistence (though API mainly reads/polls, 
    # it's good to ensure connectivity)
    db.initialize()
    
    # Start model loading in background
    asyncio.create_task(load_models_async())
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down SpeechScan API...")

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    description="Dysarthria Detection and Gender Classification API",
    version=settings.APP_VERSION,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory=str(settings.BASE_DIR / "web_interface")), name="static")

@app.get("/", response_class=HTMLResponse)
async def serve_web_interface():
    """Serve the web interface"""
    web_interface_path = settings.BASE_DIR / "web_interface" / "index.html"
    if web_interface_path.exists():
        with open(web_interface_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="<h1>SpeechScan API</h1><p>Web interface not found</p>")

@app.get("/loading-status", response_model=LoadingStatusResponse)
async def get_loading_status():
    """Get current model loading status"""
    return LoadingStatusResponse(**loading_progress)

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    websocket_connections.add(websocket)
    
    try:
        # Send initial status
        await websocket.send_text(json.dumps({
            "type": "connection_established",
            **loading_progress
        }))
        
        # Keep connection alive
        while True:
            await websocket.receive_text()
            
    except WebSocketDisconnect:
        websocket_connections.discard(websocket)
        logger.info("WebSocket client disconnected")



def format_prediction_response(prediction_result: Dict[str, Any]) -> PredictionResponse:
    """Format prediction result for web interface"""
    try:
        # Extract health prediction
        health_pred = prediction_result.get("health_prediction", "Unknown")
        health_prob = prediction_result.get("health_probability", 0.0)
        health_confidence = prediction_result.get("health_confidence", "Unknown")
        
        # Format status based on health prediction
        if "Normal" in health_pred:
            status = "Normal"
        elif "Abnormal" in health_pred:
            if health_prob > 0.8:
                status = "Moderate"
            elif health_prob > 0.6:
                status = "Mild"
            else:
                status = "Mild"
        else:
            status = "Unknown"
        
        # Format clarity score
        if not np.isnan(health_prob) and health_prob is not None:
            clarity_score = f"{health_prob:.2f}"
        else:
            clarity_score = "N/A"
        
        # Extract gender prediction for speech rate estimation
        gender_pred = prediction_result.get("gender_prediction", "Unknown")
        f0_mean = prediction_result.get("f0_mean_hz", 0.0)
        
        # Estimate speech rate based on gender and F0
        if not np.isnan(f0_mean) and f0_mean > 0:
            if "Female" in gender_pred or f0_mean > 165:
                speech_rate = "High-pitched"
            elif "Male" in gender_pred or f0_mean < 100:
                speech_rate = "Low-pitched"
            else:
                speech_rate = "Normal-pitched"
        else:
            speech_rate = "Unknown"
        
        # Generate acoustic notes
        acoustic_notes = []
        if not np.isnan(health_prob) and health_prob is not None:
            if health_prob > 0.7:
                acoustic_notes.append("High confidence in analysis")
            elif health_prob > 0.5:
                acoustic_notes.append("Moderate confidence in analysis")
            else:
                acoustic_notes.append("Low confidence in analysis")
        
        if not np.isnan(f0_mean) and f0_mean > 0:
            acoustic_notes.append(f"Fundamental frequency: {f0_mean:.1f} Hz")
        
        if health_confidence != "Unknown":
            acoustic_notes.append(f"Confidence level: {health_confidence}")
        
        processing_time = prediction_result.get("processing_time", 0.0)
        acoustic_notes.append(f"Processing time: {processing_time:.2f}s")
        
        acoustic_notes_str = "; ".join(acoustic_notes) if acoustic_notes else "No specific notes provided."
        
        return PredictionResponse(
            status=status,
            clarityScore=clarity_score,
            speechRate=speech_rate,
            acousticNotes=acoustic_notes_str,
            disclaimer=settings.MEDICAL_DISCLAIMER
        )
        
    except Exception as e:
        logger.error(f"Error formatting prediction response: {e}")
        return PredictionResponse(
            status="Error",
            clarityScore="N/A",
            speechRate="N/A",
            acousticNotes=f"Error processing results: {str(e)}"
        )

class JobResponse(BaseModel):
    job_id: str
    status: str
    message: str

@app.post("/predict", response_model=JobResponse)
async def predict_audio(file: UploadFile = File(...)):
    """Submit audio file for background analysis"""
    
    # Validate file type
    allowed_types = ["audio/wav", "audio/mp3", "audio/mpeg", "audio/ogg", "audio/x-m4a", "audio/webm"]
    if not file.content_type or file.content_type not in allowed_types:
        if not (file.content_type and "webm" in file.content_type):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type: {file.content_type}. Supported: wav, mp3, ogg, webm"
            )
    
    try:
        # Save uploaded file temporarily with UUID
        file_ext = file.filename.split('.')[-1] if file.filename else "webm"
        if len(file_ext) > 5 or not file_ext.isalnum():
            file_ext = "webm"
            
        temp_dir = settings.BASE_DIR / "temp_audio"
        temp_dir.mkdir(exist_ok=True)
            
        temp_filename = f"speech_sample_{uuid.uuid4()}.{file_ext}"
        # Save to temp_audio directory
        temp_file_path = temp_dir / temp_filename
        
        with open(temp_file_path, "wb") as buffer:
            while content := await file.read(1024 * 1024):
                buffer.write(content)
        
        logger.info(f"Received file: {file.filename} -> {temp_file_path}")
        
        # Submit task to Celery
        task = predict_task.delay(temp_file_path)
        logger.info(f"Task submitted: {task.id}")
        
        return JobResponse(
            job_id=task.id,
            status="pending",
            message="Audio submitted for analysis"
        )
        
    except Exception as e:
        logger.error(f"Error submitting job: {e}")
        # Cleanup if it failed before submission
        if 'temp_file_path' in locals() and os.path.exists(temp_file_path):
            os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Job submission failed: {str(e)}")

class StatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[PredictionResponse] = None
    error: Optional[str] = None

@app.get("/results/{job_id}", response_model=StatusResponse)
async def get_results(job_id: str):
    """Check status and retrieve results of a prediction job"""
    task_result = AsyncResult(job_id, app=celery_app)
    
    if task_result.status == 'PENDING':
        return StatusResponse(job_id=job_id, status="pending")
    elif task_result.status == 'SUCCESS':
        result_data = task_result.result
        if "error" in result_data:
            return StatusResponse(job_id=job_id, status="failed", error=result_data["error"])
            
        formatted_result = format_prediction_response(result_data)
        return StatusResponse(job_id=job_id, status="completed", result=formatted_result)
    elif task_result.status == 'FAILURE':
        return StatusResponse(job_id=job_id, status="failed", error=str(task_result.info))
    else:
        return StatusResponse(job_id=job_id, status=task_result.status.lower())

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": loading_progress["loaded"],
        "timestamp": time.time()
    }

if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    )
