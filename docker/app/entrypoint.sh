#!/bin/bash
set -e

LLM_MODEL_PATH=${LLM_MODEL_PATH:-"/models/llama-2-7b-chat.Q4_K_M.gguf"}
MODEL_NAME=${MODEL_NAME:-"llama-2-7b-chat-gguf"}
USE_OLLAMA=${USE_OLLAMA:-"false"}

DATABASE_URL=${DATABASE_URL:-postgresql+asyncpg://postgres:postgres@db:5432/cv_analyzer}

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

if [ "$USE_OLLAMA" = "true" ]; then
    echo "Kiểm tra kết nối với Ollama..."
    OLLAMA_BASE_URL=${OLLAMA_BASE_URL:-"http://ollama:11434"}

    max_retries=20
    retry_count=0

    while [ $retry_count -lt $max_retries ]; do
        if curl -s "$OLLAMA_BASE_URL/api/tags" > /dev/null; then
            echo "Kết nối với Ollama thành công!"
            break
        else
            retry_count=$((retry_count+1))
            echo "Đợi Ollama khởi động (lần thử $retry_count/$max_retries)..."
            sleep 5
        fi
    done

    if [ $retry_count -eq $max_retries ]; then
        echo "Không thể kết nối với Ollama sau $max_retries lần thử"
        echo "Tiếp tục khởi động ứng dụng, nhưng Ollama có thể không khả dụng"
    fi
else
    echo "Kiểm tra mô hình LLM..."
    if [ ! -f "$LLM_MODEL_PATH" ]; then
        echo "Mô hình LLM không tìm thấy tại $LLM_MODEL_PATH"
        echo "Bắt đầu tải mô hình $MODEL_NAME..."
        python -m app.utils.download_model --model $MODEL_NAME --output-dir $(dirname "$LLM_MODEL_PATH")
    else
        echo "Mô hình đã tồn tại tại $LLM_MODEL_PATH"
    fi
fi

echo "Chạy migrations database..."
alembic upgrade head

echo "Khởi động ứng dụng..."
exec uvicorn main:app --host 0.0.0.0 --port 8000 --reload