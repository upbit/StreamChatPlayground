#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <gemma-7b-it|CapybaraHermes-2.5-Mistral-7B|qwen1_5-14b-chat.gguf|qwen1_5-32b-chat.gguf>"
    exit 1
fi

case "$1" in
    gemma-7b-it)
        USE_BACKEND="vllm"
        MODEL_NAME="gemma-7b-it"
        MODEL_PATH="/root/workspace/gemma-7b-it" # or google/gemma-7b-it
        EXTRA_PARAMS=""
        ;;
    CapybaraHermes-2.5-Mistral-7B)
        USE_BACKEND="vllm"
        MODEL_NAME="CapybaraHermes-2.5-Mistral-7B-AWQ"
        MODEL_PATH="/root/workspace/CapybaraHermes-2.5-Mistral-7B-AWQ"
        EXTRA_PARAMS="--quantization awq --dtype half"
        ;;
    qwen1_5-14b-chat.gguf)
        USE_BACKEND="llama.cpp"
        MODEL_NAME="qwen1_5-14b-chat"
        MODEL_PATH="/root/workspace/qwen1_5-14b-chat-q5_0.gguf"
        EXTRA_PARAMS="--n-gpu-layers 99"
        ;;
    qwen1_5-32b-chat.gguf)
        USE_BACKEND="llama.cpp"
        MODEL_NAME="qwen1_5-32b-chat"
        MODEL_PATH="/root/workspace/qwen1_5-32b-chat-q4_k_m.gguf"
        EXTRA_PARAMS="--n-gpu-layers 99"
        ;;
    *)
        echo "Unsupported: $1"
        exit 1
        ;;
esac

if [ "$USE_BACKEND" == "vllm" ]; then
    # vllm
    VLLM_COMMAND="python -m vllm.entrypoints.openai.api_server --enforce-eager"
    echo "$ $VLLM_COMMAND --served-model-name $MODEL_NAME --model $MODEL_PATH $EXTRA_PARAMS"
    $VLLM_COMMAND --served-model-name $MODEL_NAME --model $MODEL_PATH $EXTRA_PARAMS

elif [ "$USE_BACKEND" == "llama.cpp" ]; then
    # llama.cpp
    LLAMACPP_COMMAND="/root/workspace/llama.cpp/server --port 8000"
    echo "$ $LLAMACPP_COMMAND --alias $MODEL_NAME --model $MODEL_PATH $EXTRA_PARAMS"
    $LLAMACPP_COMMAND --alias $MODEL_NAME --model $MODEL_PATH $EXTRA_PARAMS

else
    echo "Unknown backend: $USE_BACKEND"
    exit 1
fi

exit 0
