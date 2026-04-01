FROM python:3.11-slim

WORKDIR /app

# System Build Dependencies for Scikit-Learn / Joblib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pre-Install PyTorch with CUDA 12.4 capability to exploit Host Passthrough via WSL2/NVIDIA drivers
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# The volume maps over this dynamically for hot-reloading context later
COPY . .

# Scrub the requirements.txt file of the hardcoded CPU extra-index-url flag safely
# This avoids accidentally ruining the bare-metal local development environment
RUN sed -i '/extra-index-url.*cpu/d' requirements.txt && \
    sed -i '/torch$/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
