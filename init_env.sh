#!/bin/bash

# Environment Initialization Script
# This script sets up the environment and downloads required models

set -e

HF_TOKEN=$(grep '^auth_token' config.toml 2>/dev/null | grep -v 'hf_xxx' | cut -d'"' -f2)
if [ -z "$HF_TOKEN" ]; then
    echo "Please enter your HuggingFace token for login: "
    read -s HF_TOKEN
    echo ""
fi
if [ -z "$HF_TOKEN" ]; then
    echo "Error: HuggingFace token cannot be empty!"
    exit 1
fi

if [ -f "/etc/network_turbo" ]; then
    source /etc/network_turbo
fi
export HF_ENDPOINT=https://hf-mirror.com

echo "Creating conda environment at .conda with Python 3.12..."
conda create -p .conda python=3.12 -y

echo "Activating conda environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ./.conda

get_torch_index_url() {
    if ! command -v nvidia-smi &> /dev/null; then
        echo "https://download.pytorch.org/whl/cpu"
        return
    fi
    
    local CUDA_VERSION_FULL=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+' | head -1)
    if [ -z "$CUDA_VERSION_FULL" ]; then
        echo "https://download.pytorch.org/whl/cpu"
        return
    fi
    
    local CUDA_MAJOR=$(echo "$CUDA_VERSION_FULL" | cut -d'.' -f1)
    echo "Detected CUDA version: $CUDA_VERSION_FULL" >&2
    
    local EXACT_VERSION="cu${CUDA_MAJOR}$(echo "$CUDA_VERSION_FULL" | cut -d'.' -f2)"
    if curl -s -I "https://download.pytorch.org/whl/$EXACT_VERSION/" | head -1 | grep -q "200\|301\|302"; then
        echo "https://download.pytorch.org/whl/$EXACT_VERSION"
        return
    fi
    
    case $CUDA_MAJOR in
        13) echo "https://download.pytorch.org/whl/cu130" ;;
        12) echo "https://download.pytorch.org/whl/cu121" ;;
        11) echo "https://download.pytorch.org/whl/cu118" ;;
        *) echo "https://download.pytorch.org/whl/cpu" ;;
    esac
}

TORCH_INDEX_URL=$(get_torch_index_url)
echo "Installing PyTorch ($TORCH_INDEX_URL) and other dependencies..."
pip install torch --index-url "$TORCH_INDEX_URL"
pip install -r <(grep -v "^torch" requirements.txt) && pip install jury --no-deps && pip install modelscope

echo "Logging in to HuggingFace CLI..."
huggingface-cli login --token "$HF_TOKEN"
huggingface-cli whoami

echo "Downloading models..."
mkdir -p ./models/llama ./models/deberta ./models/self_rag
modelscope download --model shakechen/Llama-2-7b-chat-hf --local_dir ./models/llama
modelscope download --model microsoft/deberta-v2-xlarge-mnli --local_dir ./models/deberta
huggingface-cli download selfrag/selfrag_llama2_7b --local-dir ./models/self_rag
