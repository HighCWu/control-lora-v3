import os
import sys
main_dir = os.path.abspath(os.path.dirname(__file__) + "/..")
os.chdir(main_dir)
sys.path.insert(0, main_dir)


task = "canny"
dataset_name = "../../data/sd-generated/sd1_5_canny_pair_data"
proportion_empty_prompts = 0.1
rank = 16
base_lr = 1e-4
mini_batch_size = 4
total_batch_size = 64
gradient_accumulation_steps = total_batch_size // mini_batch_size
mixed_precision = "no" if gradient_accumulation_steps > 16 else "fp16"
learning_rate = 1e-4 * gradient_accumulation_steps
save_steps = 5000
finish_steps = 50000
real_save_steps = save_steps // gradient_accumulation_steps
real_finish_steps = real_save_steps * finish_steps // save_steps
loraplus_lr_ratio = 1.0
half_or_full_lora = "half_skip_attn"
init_lora_weights = "gaussian"
extra_lora_rank_modules = ["conv_in"]
extra_lora_ranks = [64]
extra_suffix = "-".join([f"""{n}-rank{r * 2 if "pissa" in init_lora_weights else r}""" for n, r in zip(extra_lora_rank_modules, extra_lora_ranks)])
lora_adapter_name = f"""sd-control-lora-v3-{task}"""
repo_id = f"""sd-control-lora-v3-{task}-{half_or_full_lora}-rank{rank * 2 if "pissa" in init_lora_weights else rank}-{extra_suffix}"""
cmd = f"""\
accelerate launch train.py \
 --pretrained_model_name_or_path="runwayml/stable-diffusion-v1-5" \
 --output_dir="out/{repo_id}" \
 --tracker_project_name="{repo_id}" \
 --dataset_name={dataset_name} \
 --proportion_empty_prompts={proportion_empty_prompts} \
 --conditioning_image_column=guide \
 --image_column=image \
 --caption_column=text \
 --rank={rank} \
 --lora_adapter_name={lora_adapter_name} \
 --init_lora_weights={init_lora_weights} \
 --loraplus_lr_ratio={loraplus_lr_ratio} \
 --half_or_full_lora={half_or_full_lora} \
 --extra_lora_rank_modules {" ".join(extra_lora_rank_modules)} \
 --extra_lora_ranks {" ".join([str(r) for r in extra_lora_ranks])} \
 --resolution=512 \
 --learning_rate={learning_rate} \
 --seed=42 \
 --validation_image "./imgs/canny1.jpg" "./imgs/canny2.jpg" "./imgs/canny3.jpg" "./imgs/canny4.jpg" \
 --validation_prompt \
"portrait of a beautiful winged goddess with horns, long wavy black hair, long black dress with silver jewels by tom bagshaw" \
"an emo portrait painting. short dark brown messy pixie haircut, large black eyes, antichrist eyes, slightly rounded face, pointed chin, thin lips, small nose, black tank top, black leather jacket, black knee - length skirt, black choker, gold earring, by peter mohrbacher, by rebecca guay, by ron spencer" \
"a photograph of a futuristic street scene, brutalist style, straight edges, finely detailed oil painting, impasto brush strokes, soft light, 8 k, dramatic composition, dramatic lighting, sharp focus, octane render, masterpiece, by adrian ghenie and jenny saville and zhang jingna" \
"portrait of a dancing eagle woman, beautiful blonde haired lakota sioux goddess, intricate, highly detailed art by james jean, ray tracing, digital painting, artstation, concept art, smooth, sharp focus, illustration, artgerm and greg rutkowski and alphonse mucha, vladimir kush, giger, roger dean, 8 k" \
 --train_batch_size={mini_batch_size} \
 --gradient_accumulation_steps={gradient_accumulation_steps} \
 --mixed_precision={mixed_precision} \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps={real_save_steps} \
 --validation_steps={real_save_steps} \
 --max_train_steps={real_finish_steps} \
 --resume_from_checkpoint=latest \
 --report_to wandb \
 --push_to_hub
"""

print(cmd)
os.system(cmd)
