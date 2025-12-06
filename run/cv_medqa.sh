
#!/bin/bash
set -e 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=$(pwd)/src

DATASET=cv_medqa
TAG=auq

declare -A LAYER_MAP=(
    ["meta-llama/Llama-3.1-8B-Instruct"]=32
    ["Qwen/Qwen2.5-7B-Instruct"]=9
    ["BioMistral/BioMistral-7B"]=30
    ["ContactDoctor/Bio-Medical-Llama-3-8B"]=11
)

for MODEL in \
    Qwen/Qwen2.5-7B-Instruct \
    meta-llama/Llama-3.1-8B-Instruct \
    ContactDoctor/Bio-Medical-Llama-3-8B \
    BioMistral/BioMistral-7B 
do
    echo "==============================="
    echo "Evaluating $MODEL (dataset=$DATASET)"
    echo "==============================="

    LAYER=${LAYER_MAP[$MODEL]}

    echo "Using hidden_state_layer=$LAYER"

    OVERRIDES=(
        "dataset=./CV-MedBench/${DATASET}/test"
        "model.path=$MODEL"
        "subsample_eval_dataset=-1"
        "tag=$TAG"
        "hidden_state_layer=$LAYER"
        "report_to_wandb=false"
    )

    CUDA_VISIBLE_DEVICES=0 ./scripts/polygraph_eval \
        --config-dir=./examples/configs/ \
        --config-name=eval_${DATASET}.yaml \
        "${OVERRIDES[@]}"

    sleep 5

done