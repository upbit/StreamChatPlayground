#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <gemma-7b-it|CapybaraHermes-2.5-Mistral-7B>"
    exit 1
fi

case "$1" in
    gemma-7b-it)
        MODEL_NAME="gemma-7b-it"
        MODEL_PATH="/root/workspace/gemma-7b-it" # or google/gemma-7b-it
        EXTRA_PARAMS=""
        ;;
    CapybaraHermes-2.5-Mistral-7B)
        MODEL_NAME="CapybaraHermes-2.5-Mistral-7B-AWQ"
        MODEL_PATH="/root/workspace/CapybaraHermes-2.5-Mistral-7B-AWQ"
        EXTRA_PARAMS="--quantization awq --dtype half"
        ;;
    *)
        echo "Unsupported: $1"
        exit 1
        ;;
esac

# Start vllm
VLLM_COMMAND="python -m vllm.entrypoints.openai.api_server --enforce-eager"
echo "> $VLLM_COMMAND --served-model-name $MODEL_NAME --model $MODEL_PATH $EXTRA_PARAMS"
$VLLM_COMMAND --served-model-name $MODEL_NAME --model $MODEL_PATH $EXTRA_PARAMS

exit 0
