FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .

# Download model files directly from GitHub LFS
RUN curl -L -o attention_fusion_best.pth \
    "https://media.githubusercontent.com/media/fayyazayesha576-gif/i-Scanner/main/backend/attention_fusion_best.pth"

RUN curl -L -o meta_scaler.pkl \
    "https://media.githubusercontent.com/media/fayyazayesha576-gif/i-Scanner/main/backend/meta_scaler.pkl"

RUN curl -L -o best_thresholds.npy \
    "https://media.githubusercontent.com/media/fayyazayesha576-gif/i-Scanner/main/backend/best_thresholds.npy"

RUN ls -lh && echo "Model files downloaded"

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]