import os
import sys
main_dir = os.path.abspath(os.path.dirname(__file__) + "/..")
os.chdir(main_dir)
sys.path.insert(0, main_dir)


task = "normal"
dataset_name = "../../data/sd-generated/sd1_5_normal_pair_data"
proportion_empty_prompts = 0.1
rank = 16
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
 --validation_image "./imgs/normal1.jpg" "./imgs/normal2.jpg" "./imgs/normal3.jpg" "./imgs/normal4.jpg" \
 --validation_prompt \
"portrait of young black super hero girl, short blonde Afro,sexy red lips, tall and slim figured, Brooklyn background, highly detailed and rendered gold jewelry, digital art, intricate, sharp focus, Trending on Artstation, HQ, unreal engine 5, 4K UHD image, by brom, artgerm, face by Otto Schmidt" \
"beautiful feathered beauty portrait, artgerm, peter mohrbacher, radiant light, swirling feathers" \
"d & d fantasy art, a huge human mouth with large flat teeth, large dorsal fins swimming through a dark ocean, pink skin, sinew, concept art, character art, horror" \
"brutalist architecture by Le Corbusier, abandoned and empty streetscapes, surrounded by lush green vegetation, ground-level view, stunning volumetric lighting, sunset, rusted steel, solid concrete, glass, stunning skies, trending on Artstation, 8k, photorealistic, hyper detailed, unreal engine 5, IMAX quality, cinematic, epic lighting, in the style of DOOM and Greg Rutkowski" \
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
