FROM python:3.11-slim

WORKDIR /app

# Cài đặt các dependencies và thư viện hỗ trợ
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    curl \
    poppler-utils \
    tesseract-ocr \
    libreoffice \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Cài đặt Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt huggingface_hub tqdm

# Tạo thư mục cho mô hình LLM
ARG LLM_MODEL_PATH="/models"
RUN mkdir -p ${LLM_MODEL_PATH}

# Copy source code của ứng dụng
COPY . .

# Làm cho entrypoint script có thể thực thi
RUN chmod +x /app/docker/app/entrypoint.sh

# Expose port FastAPI sẽ chạy
EXPOSE 8000

# Chạy entrypoint script
ENTRYPOINT ["/app/docker/app/entrypoint.sh"]