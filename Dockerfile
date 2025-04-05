FROM python:3.11-slim

WORKDIR /app

# Cài đặt các dependencies và thư viện hỗ trợ
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-vie \
    libreoffice \
    git \
    libgl1-mesa-glx \
    ffmpeg \
    libsm6 \
    libxext6 \
    libmagic1 \
    wget \
    unzip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt huggingface_hub tqdm

# Cài đặt extra models cho NLTK
RUN python -m nltk.downloader punkt

# Tạo thư mục cho mô hình LLM
ARG LLM_MODEL_PATH="/models"
RUN mkdir -p ${LLM_MODEL_PATH}

RUN mkdir -p /tmp/uploads /tmp/knowledge /tmp/chat_uploads \
    && chmod -R 777 /tmp/uploads /tmp/knowledge /tmp/chat_uploads

# Copy source code của ứng dụng
COPY . .

# Tạo các thư mục cần thiết
RUN mkdir -p /tmp/uploads /tmp/knowledge

# Làm cho entrypoint script có thể thực thi
RUN chmod +x /app/docker/app/entrypoint.sh

# Expose port FastAPI sẽ chạy
EXPOSE 8000

# Chạy entrypoint script
ENTRYPOINT ["/app/docker/app/entrypoint.sh"]