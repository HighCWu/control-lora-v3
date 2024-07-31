import os
import sys
main_dir = os.path.abspath(os.path.dirname(__file__) + "/..")
os.chdir(main_dir)
sys.path.insert(0, main_dir)


task = "depth"
dataset_name = "../../data/sd-generated/sd1_5_depth_pair_data"
proportion_empty_prompts = 0.1
rank = 4
learning_rate = 1e-4
half_or_full_lora = "half"
init_lora_weights = "pissa"
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
 --half_or_full_lora={half_or_full_lora} \
 --extra_lora_rank_modules {" ".join(extra_lora_rank_modules)} \
 --extra_lora_ranks {" ".join([str(r) for r in extra_lora_ranks])} \
 --resolution=512 \
 --learning_rate={learning_rate} \
 --seed=42 \
 --validation_image "./imgs/depth1.jpg" "./imgs/depth2.jpg" "./imgs/depth3.jpg" "./imgs/depth4.jpg" \
 --validation_prompt \
"a gorgeous woman with long light-blonde hair wearing a low cut tanktop, standing in the rain on top of a mountain, highly detailed, artstation, concept art, sharp focus, illustration, art by artgerm and alphonse mucha" \
"a colored colorized portrait of andy warhol, hyperrealistic, extremely realistic, highly realistic, hd quality, 4 k resolution, 8 k resolution, detailed, very detailed, highly detailed, extremely detailed, intricate details, real, very real, oil painting, digital painting, painting, trending on deviantart, trending on artstation, in the style of david levine, in the style of greg rutkowski" \
"lofi bus from school portrait, Pixar style, by Tristan Eaton Stanley Artgerm and Tom Bagshaw." \
"( cyberpunk 2 0 7 7, bladerunner 2 0 4 9 ), a complex thick bifurcated robotic cnc surgical arm cybernetic symbiosis hybrid mri 3 d printer machine making a bio chemical lab, art by artgerm and greg rutkowski and alphonse mucha, biomechanical, lens orbs, global illumination, octane render, architectural, f 3 2," \
 --train_batch_size=4 \
 --mixed_precision=fp16 \
 --enable_xformers_memory_efficient_attention \
 --checkpointing_steps=5000 \
 --validation_steps=5000 \
 --max_train_steps=50000 \
 --resume_from_checkpoint=latest \
 --report_to wandb \
 --push_to_hub
"""

print(cmd)
os.system(cmd)
