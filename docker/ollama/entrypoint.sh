#!/bin/bash
set -e

echo "Waiting for Ollama to initialize..."
sleep 5

echo "Pulling required models..."
ollama pull llama3:7b

echo "Ollama initialization completed"

# Keep container running
tail -f /dev/null