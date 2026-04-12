FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency management
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

# Copy project files
COPY pyproject.toml README.md ./
COPY uv.lock ./
COPY schedulrx/ ./schedulrx/

# Install the project and dependencies
RUN uv pip install --system .

# Metadata
ENV PYTHONUNBUFFERED=1
EXPOSE 7860

# Launch using the new entry point
CMD ["server"]