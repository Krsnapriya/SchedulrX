FROM --platform=linux/amd64 python:3.11-slim-bookworm

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and dashboard
COPY . .
RUN chmod +x entrypoint.sh

# Expose port 7860 for HF Spaces
EXPOSE 7860

# Run entrypoint script
CMD ["./entrypoint.sh"]