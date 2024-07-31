import os
import sys
main_dir = os.path.abspath(os.path.dirname(__file__) + "/..")
os.chdir(main_dir)
sys.path.insert(0, main_dir)


task = "segmentation"
dataset_name = "../../data/sd-generated/sd1_5_segmentation_pair_data"
proportion_empty_prompts = 0.1
rank = 128
learning_rate = 1e-4
half_or_full_lora = "half_skip_attn"
init_lora_weights = "gaussian"
extra_lora_rank_modules = ["conv_in"]
extra_lora_ranks = [128]
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
 --validation_image "./imgs/segmentation1.jpg" "./imgs/segmentation2.jpg" "./imgs/segmentation3.jpg" "./imgs/segmentation4.jpg" \
 --validation_prompt \
"realistic painting of a tardigrade kaiju, with 6 legs in a desert storm, by james gurney, slime, big globule eye, godzilla, vintage, concept art, oil painting, tonalism, crispy" \
"portrait of a beautiful cute strong brave realistic! female gnome engineer, textured undercut black hair, d & d, micro detail, intricate, elegant, highly detailed, centered, rule of thirds, artstation, sharp focus, illustration, artgerm, tomasz alen kopera, donato giancola, wlop" \
"beautiful digital painting of a stylish asian female forest with high detail, 8 k, stunning detail, works by artgerm, greg rutkowski and alphonse mucha, unreal engine 5, 4 k uhd" \
"duotone noir scifi concept dynamic illustration of 3 d mesh of robotic cats inside box floating zero gravity glowing 3 d mesh portals futuristic, glowing eyes, octane render, surreal atmosphere, volumetric lighting. accidental renaissance. by sachin teng and sergey kolesov and ruan jia and heng z. graffiti art, scifi, fantasy, hyper detailed. trending on artstation" \
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
