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

# 2. Setup Virtual Environment
if [ ! -d "venv" ]; then
    echo -e "${BLUE}📦 Creating virtual environment...${NC}"
    python3 -m venv venv
fi

echo -e "${BLUE}🐍 Activating virtual environment and installing dependencies...${NC}"
source venv/bin/activate
pip install --upgrade pip
pip install -r api/requirements.txt

# 3. Start Celery Worker in background
echo -e "${BLUE}🤖 Starting Celery Inference Worker...${NC}"
python3 -m celery -A api.celery_app worker --loglevel=info > worker.log 2>&1 &
WORKER_PID=$!
echo -e "${GREEN}✅ Worker started (PID: $WORKER_PID). Logs: worker.log${NC}"

# 4. Start FastAPI Server
echo -e "${BLUE}🌐 Starting SpeechScan API & Web Interface...${NC}"
echo -e "${YELLOW}Dashboard available at: http://localhost:8000${NC}"

# Cleanup on exit
trap "kill $WORKER_PID; echo -e '\n${YELLOW}🛑 Shutting down...${NC}'; exit" SIGINT SIGTERM

python3 api/main.py
