import os
import sys
import requests
import logging
from pathlib import Path
from tqdm import tqdm
from huggingface_hub import hf_hub_download


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("model-downloader")


HF_MODELS = {
    "llama-2-7b-chat-gguf": {
        "repo_id": "TheBloke/Llama-2-7B-Chat-GGUF",
        "filename": "llama-2-7b-chat.Q4_K_M.gguf",
        "type": "gguf"
    },
    "mistral-7b-instruct-gguf": {
        "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
        "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "type": "gguf"
    },
    "gemma-7b-it-gguf": {
        "repo_id": "TheBloke/Gemma-7B-it-GGUF",
        "filename": "gemma-7b-it.Q4_K_M.gguf",
        "type": "gguf"
    }
}


def download_model_from_hf(model_name: str, output_dir: str):
    if model_name not in HF_MODELS:
        logger.error(f"Mô hình {model_name} không được hỗ trợ. Các mô hình hỗ trợ: {', '.join(HF_MODELS.keys())}")
        return False

    model_info = HF_MODELS[model_name]
    repo_id = model_info["repo_id"]
    filename = model_info["filename"]

    logger.info(f"Bắt đầu tải mô hình {model_name} từ {repo_id}")

    try:
        output_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        logger.info(f"Tải mô hình thành công: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình: {str(e)}")
        return False


def download_direct_url(url: str, output_path: str):
    try:
        logger.info(f"Bắt đầu tải từ {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024

        with open(output_path, 'wb') as f, tqdm(
                desc=output_path,
                total=total_size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
        ) as progress_bar:
            for data in response.iter_content(block_size):
                size = f.write(data)
                progress_bar.update(size)

        logger.info(f"Tải thành công: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Lỗi khi tải mô hình: {str(e)}")
        return False


def check_and_download_model(model_path: str, model_name: str = "llama-2-7b-chat-gguf"):
    if os.path.exists(model_path):
        logger.info(f"Mô hình đã tồn tại tại: {model_path}")
        return True

    model_dir = os.path.dirname(model_path)
    os.makedirs(model_dir, exist_ok=True)

    return download_model_from_hf(model_name, model_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Tải mô hình LLM từ Hugging Face Hub")
    parser.add_argument("--model", type=str, default="llama-2-7b-chat-gguf",
                        help=f"Tên mô hình để tải. Hỗ trợ: {', '.join(HF_MODELS.keys())}")
    parser.add_argument("--output-dir", type=str, default="/models",
                        help="Thư mục lưu mô hình")

    args = parser.parse_args()

    if args.model in HF_MODELS:
        download_model_from_hf(args.model, args.output_dir)
    else:
        logger.error(f"Mô hình {args.model} không được hỗ trợ. Các mô hình hỗ trợ: {', '.join(HF_MODELS.keys())}")