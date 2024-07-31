import os
import sys
main_dir = os.path.abspath(os.path.dirname(__file__) + "/..")
os.chdir(main_dir)
sys.path.insert(0, main_dir)


task = "tile"
dataset_script_path = "./exps/sd1_5_tile_pair_data.py"
proportion_empty_prompts = 0.1
rank = 16
base_lr = 1e-4
mini_batch_size = 4
total_batch_size = 64
gradient_accumulation_steps = total_batch_size // mini_batch_size
mixed_precision = "no" if gradient_accumulation_steps > 16 else "fp16"
learning_rate = 1e-4 * gradient_accumulation_steps
save_steps = 5000
finish_steps = 20000
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
 --dataset_script_path={dataset_script_path} \
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
 --validation_image "./imgs/tile1.jpg" "./imgs/tile2.jpg" "./imgs/tile3.jpg" "./imgs/tile4.jpg" \
 --validation_prompt \
"portrait of asian female humanoid, crew cut colored hair, very details, elegant, cyber neon lights, highly detailed, digital illustration, trending in artstation, trending in pinterest, glamor pose, concept art, smooth, sharp focus, art by artgerm and greg rutkowski" \
"A tin can man performance" \
"smiling, happy, beautiful, intelligent, powerful ww 1 housewife, 2 8 years old, loving eyes, fully clothed, wise, beautiful, dramatic lighting, sharp focus, by stanley artgerm, dramatic lighting, trending on artstation, flat colour, geometric curves, gradient filter, art deco patterns" \
"portrait of a man by greg rutkowski, dan skywalker from star wars expanded universe, wearing tactical gear of the triunvirate of the galactic alliance, he is about 3 0 years old, highly detailed portrait, digital painting, artstation, concept art, smooth, sharp foccus ilustration, artstation hq" \
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
