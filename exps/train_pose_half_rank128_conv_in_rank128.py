import os
import sys
main_dir = os.path.abspath(os.path.dirname(__file__) + "/..")
os.chdir(main_dir)
sys.path.insert(0, main_dir)


task = "pose"
dataset_name = "../../data/sd-generated/sd1_5_pose_pair_data"
proportion_empty_prompts = 0.1
rank = 64
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
 --validation_image "./imgs/pose1.jpg" "./imgs/pose2.jpg" "./imgs/pose3.jpg" "./imgs/pose4.jpg" \
 --validation_prompt \
"medieval rich kingpin sitting in a tavern with his thugs, drinking and cheering, few subjecs, elegant, close frontal shot, digital painting, concept art, smooth, sharp focus, illustration, from d & d by ruan jia and mandy jurgens and artgerm and william - adolphe bouguerea" \
"sensual good looking pale young indian doctors wearing jeans in celebrating after passing an exam, portrait, elegant, intricate, digital painting, artstation, concept art, smooth, sharp focus, illustration, art by artgerm and greg rutkowski and alphonse mucha" \
"a psychedelic detailed gorgeous acid trip painting of an extremely sexy elegant and attractive female character, wearing a tight-fitting tan detective jacket, detective had on her head, beautiful [[long red hair]] in loose curls, slender woman, very curvy, noir, smoking a fancy long french cigarette, in the rain in the early evening, cinematic, dramatic lighting, full body view, cool pose, artwork by Artgerm, Rutkowski, Dale Keown and Van Sciver, featured on artstation, cgsociety, behance hd" \
"wow! fanart, ork from warhammer 4 0 k with, d & d, high fantasy, detailed, digital art, artstation, smooth, sharp focus, art by artgerm, greg rutkowski, alphonse mucha" \
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
