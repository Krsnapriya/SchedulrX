FROM --platform=linux/amd64 python:3.11-slim-bookworm

RUN apt-get update && apt-get install -y nginx && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
COPY nginx.conf /etc/nginx/sites-available/default

EXPOSE 7860

# Harden startup: Ensure backend starts before Nginx to minimize 502 window
CMD bash -c "uvicorn api:app --host 0.0.0.0 --port 8001 --workers 1 & \
  streamlit run app.py --server.port=8000 --server.address=0.0.0.0 --server.headless=true & \
  sleep 5 && nginx -g 'daemon off;'"