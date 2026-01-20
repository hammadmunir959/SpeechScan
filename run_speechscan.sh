#!/bin/bash

# SpeechScan Unified Launch Script
# This script automates the setup and execution of Redis, Celery Workers, and the FastAPI Server.

# Colors for better visibility
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting SpeechScan Production-Ready Environment...${NC}"

# 1. Check for Redis
if ! command -v redis-server &> /dev/null; then
    echo -e "${YELLOW}⚠️ Redis not found. Attempting to start via Docker...${NC}"
    if ! command -v docker &> /dev/null; then
        echo -e "\033[0;31m❌ Error: Redis is required but neither redis-server nor Docker is installed.\033[0m"
        exit 1
    fi
    docker run -p 6379:6379 -d --name speechscan-redis-bk redis:alpine &> /dev/null
    echo -e "${GREEN}✅ Redis started via Docker container 'speechscan-redis-bk'${NC}"
else
    # Check if redis is already running
    if ! redis-cli ping &> /dev/null; then
        echo -e "${BLUE}🔄 Starting local Redis server...${NC}"
        redis-server --daemonize yes
    fi
    echo -e "${GREEN}✅ Redis is running.${NC}"
fi

# 2. Setup Virtual Environment (Local .venv)
VENV_PATH="./.venv"

if [ ! -d "$VENV_PATH" ]; then
    echo -e "${YELLOW}⚠️ Local virtual environment not found. Creating it...${NC}"
    python3 -m venv "$VENV_PATH"
    echo -e "${BLUE}🐍 Activating new venv and installing dependencies...${NC}"
    source "$VENV_PATH/bin/activate"
    pip install --upgrade pip setuptools wheel
    if [ -f "api/requirements.txt" ]; then
        pip install -r api/requirements.txt
    else
        echo -e "${YELLOW}⚠️ api/requirements.txt not found. Skipping auto-installation.${NC}"
    fi
else
    echo -e "${BLUE}🐍 Activating local virtual environment: $VENV_PATH${NC}"
    source "$VENV_PATH/bin/activate"
fi

# 3. Production Environment Prep
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Check if port 8000 is already in use
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null ; then
    echo -e "${YELLOW}⚠️ Warning: Port 8000 is already in use. Attempting to kill the process...${NC}"
    fuser -k 8000/tcp &> /dev/null
    sleep 1
fi

# 4. Start Celery Worker in background
echo -e "${BLUE}🤖 Starting Celery Inference Worker...${NC}"
python3 -m celery -A api.celery_app worker --loglevel=info > worker.log 2>&1 &
WORKER_PID=$!
echo -e "${GREEN}✅ Worker started (PID: $WORKER_PID). Logs: worker.log${NC}"

# 5. Start FastAPI Server
echo -e "${BLUE}🌐 Starting SpeechScan API & Web Interface...${NC}"
echo -e "${YELLOW}Dashboard available at: http://localhost:8000${NC}"

# Robust Cleanup on exit
function cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down SpeechScan components...${NC}"
    kill $WORKER_PID 2>/dev/null
    # Kill any other leftover processes on port 8000
    fuser -k 8000/tcp &> /dev/null
    echo -e "${GREEN}✅ Cleanup complete.${NC}"
    exit
}

trap cleanup SIGINT SIGTERM

python3 -m api.main
