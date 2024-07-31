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
unet = unet.add_extra_conditions("sd-control-lora-v3-normal")

pipe: StableDiffusionControlLoraV3Pipeline = StableDiffusionControlLoraV3Pipeline.from_pretrained(
    model_id, vae=vae, unet=unet, safety_checker=None, torch_dtype=torch.float16
)
# load attention processors
# pipe.load_lora_weights("out/sd-control-lora-v3-normal-half-rank32-conv_in-rank128")
pipe.load_lora_weights("HighCWu/control-lora-v3", subfolder="sd-control-lora-v3-normal-half-rank32-conv_in-rank128")
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


images = [(path, load_image(path)) for path in ["./imgs/normal1.jpg", "./imgs/normal2.jpg", "./imgs/normal3.jpg", "./imgs/normal4.jpg"]]
prompts = [
    "portrait of young black super hero girl, short blonde Afro,sexy red lips, tall and slim figured, Brooklyn background, highly detailed and rendered gold jewelry, digital art, intricate, sharp focus, Trending on Artstation, HQ, unreal engine 5, 4K UHD image, by brom, artgerm, face by Otto Schmidt",
    "beautiful feathered beauty portrait, artgerm, peter mohrbacher, radiant light, swirling feathers",
    "d & d fantasy art, a huge human mouth with large flat teeth, large dorsal fins swimming through a dark ocean, pink skin, sinew, concept art, character art, horror",
    "brutalist architecture by Le Corbusier, abandoned and empty streetscapes, surrounded by lush green vegetation, ground-level view, stunning volumetric lighting, sunset, rusted steel, solid concrete, glass, stunning skies, trending on Artstation, 8k, photorealistic, hyper detailed, unreal engine 5, IMAX quality, cinematic, epic lighting, in the style of DOOM and Greg Rutkowski"
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
