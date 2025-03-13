set -x

export VLLM_ATTENTION_BACKEND=XFORMERS

MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct  # replace it with your local file path

SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
 The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""

# SYSTEM_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer.
#  The reasoning process MUST BE enclosed within <think> </think> tags. You are encouraged to use "wait", "but", "alternatively" to correct the reasoning process if needed. The final answer MUST BE put in \boxed{}."""

python3 -m verl.trainer.main \
    config=examples/grpo_example.yaml \
    data.train_files=shangzhu/vqa-rad@train \
    data.val_files=shangzhu/vqa-rad@test \
    data.system_prompt="${SYSTEM_PROMPT}" \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.enable_chunked_prefill=false \
    trainer.experiment_name=qwen2_5_vl_7b_vqa \
    trainer.n_gpus_per_node=8
