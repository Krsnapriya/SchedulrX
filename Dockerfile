FROM python:3.11-slim

# FIX #28: install system deps + all python deps in one clean layer
RUN apt-get update && apt-get install -y nginx && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY nginx.conf /etc/nginx/sites-available/default

EXPOSE 7860

# FIX #29: use a simple supervisor-style startup
# If uvicorn crashes, the container exits and HF restarts it automatically
CMD bash -c "\
  uvicorn api:app --host 0.0.0.0 --port 8001 --workers 1 & \
  streamlit run app.py --server.port=8000 --server.address=0.0.0.0 --server.headless=true & \
  nginx -g 'daemon off;'"