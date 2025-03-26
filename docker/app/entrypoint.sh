#!/bin/bash
set -e

LLM_MODEL_PATH=${LLM_MODEL_PATH:-"/models/llama-2-7b-chat.Q4_K_M.gguf"}
MODEL_NAME=${MODEL_NAME:-"llama-2-7b-chat-gguf"}

DATABASE_URL=${DATABASE_URL:-postgresql+asyncpg://postgres:password@db:5432/cv_analyzer}

echo "Kiểm tra kết nối database..."
python -c "
import asyncio
import asyncpg
from urllib.parse import urlparse
from time import sleep

async def check_db():
    db_url = '$DATABASE_URL'.replace('postgresql+asyncpg', 'postgresql')
    parsed = urlparse(db_url)
    user = parsed.username
    password = parsed.password
    host = parsed.hostname
    port = parsed.port or 5432
    dbname = parsed.path.lstrip('/')

    max_retries = 10
    retry_count = 0

    while retry_count < max_retries:
        try:
            conn = await asyncpg.connect(
                user=user,
                password=password,
                host=host,
                port=port,
                database=dbname
            )
            await conn.close()
            print('Database connection successful')
            return True
        except Exception as e:
            retry_count += 1
            print(f'Error connecting to database (attempt {retry_count}/{max_retries}): {str(e)}')
            if retry_count < max_retries:
                print('Retrying in 5 seconds...')
                sleep(5)
            else:
                print('Max retries reached, exiting...')
                return False

asyncio.run(check_db())
"

echo "Kiểm tra mô hình LLM..."
if [ ! -f "$LLM_MODEL_PATH" ]; then
    echo "Mô hình LLM không tìm thấy tại $LLM_MODEL_PATH"
    echo "Bắt đầu tải mô hình $MODEL_NAME..."
    python -m app.utils.download_model --model $MODEL_NAME --output-dir $(dirname "$LLM_MODEL_PATH")
else
    echo "Mô hình đã tồn tại tại $LLM_MODEL_PATH"
fi

if [ ! -d "/app/LLM_Web_search" ]; then
    echo "Cài đặt LLM_Web_search để tìm kiếm web nâng cao..."
    cd /tmp
    git clone https://github.com/mamei16/LLM_Web_search.git
    cd LLM_Web_search
    pip install -e .
    cd /app
    ln -s /tmp/LLM_Web_search /app/LLM_Web_search
fi


echo "Chạy migrations database..."
alembic upgrade head

echo "Khởi động ứng dụng..."
exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload