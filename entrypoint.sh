#!/bin/bash

# Start FastAPI in background on port 8000
echo "Starting SchedulrX Engine (FastAPI)..."
uvicorn app:app --host 0.0.0.0 --port 8000 &

# Start Streamlit on the public port 7860
echo "Starting SchedulrX Dashboard (Streamlit)..."
streamlit run dashboard.py --server.port 7860 --server.address 0.0.0.0
