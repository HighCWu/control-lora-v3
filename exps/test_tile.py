import os
import sys
main_dir = os.path.abspath(os.path.dirname(__file__) + "/..")
os.chdir(main_dir)
sys.path.insert(0, main_dir)



import cv2
import torch
import numpy as np

from diffusers.utils import load_image
from diffusers import AutoencoderKL, UniPCMultistepScheduler
from PIL import Image

from model import UNet2DConditionModelEx
from pipeline import StableDiffusionControlLoraV3Pipeline


model_id = "SG161222/Realistic_Vision_V4.0_noVAE" # "ckpt/anything-v3.0" # "runwayml/stable-diffusion-v1-5" # 
vae_model_path = "stabilityai/sd-vae-ft-mse" # "ckpt/anything-v3.0" # "runwayml/stable-diffusion-v1-5" # 
vae_subfolder = None # "vae" # "vae" # 

vae = AutoencoderKL.from_pretrained(vae_model_path, subfolder=vae_subfolder, torch_dtype=torch.float16)

unet: UNet2DConditionModelEx = UNet2DConditionModelEx.from_pretrained(model_id, subfolder="unet", torch_dtype=torch.float16)
unet = unet.add_extra_conditions("tile")

pipe: StableDiffusionControlLoraV3Pipeline = StableDiffusionControlLoraV3Pipeline.from_pretrained(
    model_id, vae=vae, unet=unet, safety_checker=None, torch_dtype=torch.float16
)
# load attention processors
# pipe.load_lora_weights("out/sd-control-lora-v3-tile-half_skip_attn-rank16-conv_in-rank64")
pipe.load_lora_weights("HighCWu/control-lora-v3", subfolder="sd-control-lora-v3-tile-half_skip_attn-rank16-conv_in-rank64")
pipe = pipe.to(device="cuda", dtype=torch.float16)

pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows * cols

    w, h = imgs[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    grid_w, grid_h = grid.size

    for i, img in enumerate(imgs):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid


images = [(path, load_image(path).resize([1024,1024])) for path in ["./imgs/tile1.jpg", "./imgs/tile2.jpg", "./imgs/tile3.jpg", "./imgs/tile4.jpg"]]
prompts = [
    "portrait of asian female humanoid, crew cut colored hair, very details, elegant, cyber neon lights, highly detailed, digital illustration, trending in artstation, trending in pinterest, glamor pose, concept art, smooth, sharp focus, art by artgerm and greg rutkowski",
    "A tin can man performance",
    "smiling, happy, beautiful, intelligent, powerful ww 1 housewife, 2 8 years old, loving eyes, fully clothed, wise, beautiful, dramatic lighting, sharp focus, by stanley artgerm, dramatic lighting, trending on artstation, flat colour, geometric curves, gradient filter, art deco patterns",
    "portrait of a man by greg rutkowski, dan skywalker from star wars expanded universe, wearing tactical gear of the triunvirate of the galactic alliance, he is about 3 0 years old, highly detailed portrait, digital painting, artstation, concept art, smooth, sharp foccus ilustration, artstation hq"
]

os.makedirs("./out/grids", exist_ok=True)
for (path, image), prompt in zip(images, prompts):
    prompt = "best quality, extremely detailed, " + prompt
    generator = [torch.Generator(device="cuda").manual_seed(i) for i in range(3)]

    output = pipe(
        [prompt]*3,
        [image]*3,
        negative_prompt=["monochrome, lowres, bad anatomy, worst quality, low quality"] * 3,
        num_inference_steps=20,
        generator=generator,
    )

    grid = image_grid([image] + output.images, 2, 2)
    grid.save("./out/grids/%s.png" % os.path.basename(path).split(".")[0])
