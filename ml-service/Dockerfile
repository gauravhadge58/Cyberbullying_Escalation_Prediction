FROM python:3.11-slim

WORKDIR /app

# System Build Dependencies for Scikit-Learn / Joblib
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch (Render has no GPU — saves ~2.5GB vs CUDA build)
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

COPY . .

# Install remaining dependencies (strip any leftover cpu/torch lines from requirements.txt)
RUN sed -i '/extra-index-url/d' requirements.txt && \
    sed -i '/torch$/d' requirements.txt && \
    pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
EXPOSE 7860

# Production: no --reload flag. 
# Defaults to 7860 for HuggingFace Spaces. Render will dynamically override this with its own PORT env var.
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}
