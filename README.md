<div align="center">

# Reinforce-Ada: An Adaptive Sampling Framework for Reinforce-Style LLM Training
[![Paper](https://img.shields.io/badge/paper-A42C25?style=for-the-badge&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2510.04996) 
[![Github](https://img.shields.io/badge/verl%20version-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/RLHFlow/Reinforce-Ada)
[![Github](https://img.shields.io/badge/Tinker%20version-000000?style=for-the-badge&logo=github&logoColor=000&logoColor=white)](https://github.com/RLHFlow/Reinforce-Ada-Tinker)
[![Dataset on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/dataset-on-hf-sm.svg)](https://huggingface.co/collections/RLHFlow/reinforce-ada-68e3a8a10fc69dc56d9d86fe)
</div>

## 🚨 News
- [2025.10.15] We release the Tinker version for Reinforce-Ada.
- [2025.10.07] We release the [verl version](https://github.com/RLHFlow/Reinforce-Ada) (main version) for Reinforce-Ada.

## 🛠️ Reinforce-Ada Meets Tinker
This repository contains the official implementation for Reinforce-Ada with [Tinker](https://github.com/thinking-machines-lab/tinker), an adaptive sampling framework designed to resolve the ``signal collapse'' problem in Reinforce-style algorithm with group baselines such as GRPO, making training more efficient and effective.

All results shown in our paper are from the runnings on verl. Here we further validate the effectiveness of Reinforce-Ada on another easy-to-use training API, [Tinker](https://github.com/thinking-machines-lab/tinker).

<p align="center">
  <img src="figures/Tinker-Qwen3-4B-Instruct-2507.png" width="83%" />
</p>

<i><b>Figure 1:</b> Training dynamics with Qwen3-4B-Instruct-2507.</i>
</p>

We can observe that Reinforce-Ada achieves a significantly higher reward (from iid samples w/o adaptive sampling) than GRPO on Tinker with LoRA finetuning.

### Note:
1. We apply full finetuning in our paper, while Tinker only supports finetuning with LoRA. The above results from LoRA are slightly worse than full finetuning on verl, which might be able to improve by adjusting the LoRA rank and learning rate. 
2. For full finetuning, we utilizes a small lr=1e-6. This small lr makes the learning really slow with LoRA. [Tinker suggests](https://tinker-docs.thinkingmachines.ai/supervised-learning/sl-hyperparams) using a 40 times lr (i.e. lr=4e-5) for LoRA on Qwen3-4B-Instruct-2507, which results in reward collapse (increase first, then decrease to 0) for both GRPO and Reinforce-Ada. Here we set lr=5e-6. A dedicated tuning might result in better performance.


## 🌍 Environment Setup
1. Create a new environment.
   ```bash
   python -m venv ~/.python/reinforce_ada_tinker
   source ~/.python/reinforce_ada/bin/activate

   # You can also use conda 
   #conda create -n reinforce_ada_tinker python==3.10
   #conda activate reinforce_ada_tinker
   ```
2. Install dependencies
   ```bash
   pip install pip --upgrade
   pip install uv
   python -m uv pip install tinker
   git clone https://github.com/RLHFlow/Reinforce-Ada-Tinker
   cd Reinforce-Ada-Tinker
   python -m uv pip install -e .
   python -m uv pip install wandb
   ```

## 🧪 Experiment Running
1. Prepare the training and test datasets

    If you have custom dataset, or want to train a new LLM, please go to the [Reinforce-Ada verl version]((https://github.com/RLHFlow/Reinforce-Ada)) for this step. Otherwise, you can use our open-sourced training sets in the following.
2. Start the training

   ```bash
   # Set your key in this file
   bash scripts/run_reinforce_ada.sh
   ```

   The key hyperparameters from Reinforce-Ada are:
   - ``multiround_adaptive_downsampling=True``: Use adaptive sampling.
   - ``reinforce_ada_choice=balanced``: How to balance the positive and negative prompts within a batch, could be one of [balanced, positive-focused].
   - ``global_stat_est=True``: Use global statistics to calculate the mean and std.

   Note: Tinker considers one update of the actor as a step, while verl consider one update of the reference model as a step. total_steps, max_steps_off_policy and groups_per_batch has been set accordingly to match the verl version for training 400 steps.

3. Evaluate
   
   We don't offer the evaluation code with Tinker. It's suggested to [download the weights](https://tinker-docs.thinkingmachines.ai/download-weights) and evaluate them with our [Reinforce-Ada verl version]((https://github.com/RLHFlow/Reinforce-Ada)). You can evaluate the checkpoint every 800 steps.


## 🤗 Processed Training Sets
We offer the processed/selected training prompts in [huggingface](https://huggingface.co/collections/RLHFlow/reinforce-ada-68e3a8a10fc69dc56d9d86fe).

Note: Not all LLMs are supported by Tinker, you can check the supported models [here](https://github.com/thinking-machines-lab/tinker-cookbook/blob/main/tinker_cookbook/model_info.py). 

  | Model to train | Prompt level | Algorithm | Training set |
  | --- | --- | --- | --- | 
  | ```Qwen/Qwen2.5-Math-1.5B``` | easy | Reinforce-Ada-balance | [```RLHFlow/reinforce_ada_easy_prompt_1.5b```](https://huggingface.co/datasets/RLHFlow/reinforce_ada_simple_prompt_1-5b) 
  | ```Qwen/Qwen2.5-Math-1.5B``` | hard | Reinforce-Ada-balance | [```RLHFlow/reinforce_ada_hard_prompt_1.5b```](https://huggingface.co/datasets/RLHFlow/reinforce_ada_hard_prompt_1-5b)
  | ```Qwen/Qwen2.5-Math-7B``` | easy | Reinforce-Ada-balance | [```RLHFlow/reinforce_ada_easy_prompt```](https://huggingface.co/datasets/RLHFlow/reinforce_ada_easy_prompt) 
  | ```Qwen/Qwen2.5-Math-7B``` | hard | Reinforce-Ada-balance | [```RLHFlow/reinforce_ada_hard_prompt```](https://huggingface.co/datasets/RLHFlow/reinforce_ada_hard_prompt) 
  | ```Qwen/Qwen3-4B-Instruct-2507``` | hard | Reinforce-Ada-balance | [```RLHFlow/reinforce_ada_hard_prompt```](https://huggingface.co/datasets/RLHFlow/reinforce_ada_hard_prompt)
  | ```meta-llama/Llama-3.2-3B-Instruct``` | hard | Reinforce-Ada-balance | [```RLHFlow/reinforce_ada_hard_prompt_llama```](https://huggingface.co/datasets/RLHFlow/reinforce_ada_hard_prompt_llama) 


## 🙏 Acknowledgement
We thank [Tinker](https://github.com/thinking-machines-lab/tinker) for providing this awesome training API.

## 📝 Citation
If you find our paper or code helpful, feel free to give us a citation.
```bibtex
@misc{xiong2025reinforceada,
      title={Reinforce-Ada: An Adaptive Sampling Framework for Reinforce-Style LLM Training}, 
      author={Wei Xiong and Chenlu Ye and Baohao Liao and Hanze Dong and Xinxing Xu and Christof Monz and Jiang Bian and Nan Jiang and Tong Zhang},
      year={2025},
      eprint={2510.04996},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.04996}, 
}
```
