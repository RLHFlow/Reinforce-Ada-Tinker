#!/bin/bash

# Note:
# Tinker considers one update of the actor as a step, while verl consider one update of the reference model as a step. 
# total_steps, max_steps_off_policy and groups_per_batch has been set accordingly to match the verl version for training 400 steps.

export TINKER_API_KEY=
export WANDB_API_KEY=

model="Qwen/Qwen3-4B-Instruct-2507"
dataset_name="RLHFlow/reinforce_ada_hard_prompt"
save_dir="./outputs_grpo"

mkdir -p ${save_dir}

python -m tinker_cookbook.recipes.reinforce_ada.train \
        model_name=${model} \
        dataset_name=${dataset_name} \
        total_steps=6400 \
        max_steps_off_policy=16 \
        group_size=4 \
        groups_per_batch=32 \
        lora_rank=32 \
        learning_rate=5e-6 \
        log_path=${save_dir} \
        global_stat_est=False \
        multiround_adaptive_downsampling=False \
        max_tokens=2048 \
        wandb_project="Reinforce-Ada" \
        wandb_name="GRPO-Qwen3-4B-Instruct-2507"